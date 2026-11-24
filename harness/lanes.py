"""The four lanes: A/B (LLM reads the media), C (convert to text, then LLM), D (Decisions API).

Every call returns the same record shape, so run.py treats all lanes alike.
No retries anywhere: an error is recorded as an error.
"""
import json, pathlib, random, subprocess, tempfile, time
import pytesseract
from PIL import Image
import common as C

TESS = f'--tessdata-dir "{C.HERE / "tessdata"}"'
SYSTEM = "Answer every question about the item. Use only the allowed values for each question."
DESCRIBE_IMAGE = "Describe this image in one factual paragraph: what it shows, and any problem visible. Do not guess what you cannot see."
DESCRIBE_FRAMES = ("These are frames from one screen recording, in order, one every {s} seconds. "
                   "Describe what happens across them in one factual paragraph, including every message that appears on screen, even briefly.")


def _record(**kw):
    base = dict(answers={}, confidence={}, probabilities={}, invalid=[], tokens_in=0, tokens_out=0,
                latency_ms=None, error=None, model_version=None)
    base.update(kw)
    return base


class Lanes:
    def __init__(self, client, dry_truth=None, seed=0):
        self.client = client
        self.dry = dry_truth is not None      # dry run: no API, fake answers to test the plumbing
        self.truth = dry_truth or {}
        self.rng = random.Random(seed)

    # ------------------------------------------------------------ LLM lanes
    def _interact(self, lane, blocks, system=SYSTEM, schema=None):
        cfg = C.CFG["lanes"][lane]
        gen = {"thinking_level": cfg["thinking"]}
        if C.CFG.get("temperature") is not None:
            gen["temperature"] = C.CFG["temperature"]
        kw = dict(model=cfg["model"], input=blocks, system_instruction=system, generation_config=gen, store=False)
        if schema:
            kw["response_format"] = {"type": "text", "mime_type": "application/json", "schema": schema}
        t0 = time.perf_counter()
        it = self.client.interactions.create(**kw)
        ms = (time.perf_counter() - t0) * 1000
        u = it.usage
        out = (u.total_output_tokens or 0) + (u.total_thought_tokens or 0)  # thinking is billed as output
        return it, ms, (u.total_input_tokens or 0), out

    def llm(self, lane, item_input, questions, item_id=None):
        """Lanes A, B and C's final call. item_input: text, a content block, or a list of blocks."""
        if self.dry:
            return self._fake(lane, item_id, questions)
        blocks = item_input if isinstance(item_input, list) else [item_input]
        blocks = [{"type": "text", "text": b} if isinstance(b, str) else b for b in blocks]
        blocks.append({"type": "text", "text": "\n\nQuestions:\n" + C.render_questions(questions)})
        try:
            it, ms, tin, tout = self._interact(lane, blocks, schema=C.json_schema(questions))
        except Exception as e:                                   # noqa: BLE001  recorded, not retried
            return _record(error=f"{type(e).__name__}: {e}"[:300])
        rec = _record(latency_ms=ms, tokens_in=tin, tokens_out=tout, model_version=getattr(it, "model", None))
        if it.status != "completed":
            rec["error"] = f"status:{it.status}"
        try:
            parsed = json.loads(it.output_text or "")
        except (json.JSONDecodeError, TypeError):
            parsed = {}
        for qid, spec in questions.items():
            v = parsed.get(qid)
            if v is not None and str(v) in C.allowed(spec):
                rec["answers"][qid] = str(v)
            else:
                rec["invalid"].append(qid)
        return rec

    # ------------------------------------------------------------ lane C conversion
    def convert(self, path):
        """Turn a photo, recording or PDF into text. Returns (text, ms, cost, tokens, steps)."""
        p = pathlib.Path(path)
        t0 = time.perf_counter()
        cost, tokens, steps, parts = 0.0, 0, [], []
        if self.dry:
            return f"[dry-run conversion of {p.name}]", 900.0, 0.0002, 600, ["dry"]
        if p.suffix in (".jpg", ".jpeg", ".png", ".webp"):
            parts.append("Text found in the image (OCR):\n" + self._ocr(Image.open(p)))
            steps.append("ocr")
            d, c, t = self._describe([C.media_block(p)], DESCRIBE_IMAGE)
            parts.append("Description of the image:\n" + d); cost += c; tokens += t; steps.append("describe")
        elif p.suffix == ".mp4":
            every = C.CFG["video_frame_every_s"]
            with tempfile.TemporaryDirectory() as td:
                subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(p), "-vf", f"fps=1/{every}",
                                f"{td}/f%03d.png"], check=True)
                frames = sorted(pathlib.Path(td).glob("f*.png"))
                ocr = [f"[{i * every}s] {self._ocr(Image.open(f))}" for i, f in enumerate(frames)]
                parts.append("Text on screen, per frame (OCR):\n" + "\n".join(ocr))
                steps += ["frames", "ocr"]
                d, c, t = self._describe([C.media_block(f) for f in frames], DESCRIBE_FRAMES.format(s=every))
            parts.append("Description of the recording:\n" + d); cost += c; tokens += t; steps.append("describe")
        elif p.suffix == ".pdf":
            from pypdf import PdfReader
            text = "\n".join((pg.extract_text() or "") for pg in PdfReader(p).pages)
            if len(text.strip()) >= 50:
                steps.append("text_layer")
            else:                                                   # scanned: no text layer
                import pypdfium2 as pdfium
                pdf = pdfium.PdfDocument(str(p))
                text = "\n".join(self._ocr(pdf[i].render(scale=2.5).to_pil()) for i in range(len(pdf)))
                steps += ["render", "ocr"]
            parts.append("Text of the document:\n" + text)
        return "\n\n".join(parts), (time.perf_counter() - t0) * 1000, cost, tokens, steps

    def _ocr(self, img):
        """Standard cleanup first (upright, grayscale, 2x): without it Tesseract misses
        white-on-red text, e.g. the V1 error toast."""
        from PIL import ImageOps
        g = ImageOps.exif_transpose(img).convert("L")
        g = g.resize((g.width * 2, g.height * 2))
        return pytesseract.image_to_string(g, lang="jpn+eng", config=TESS).strip()

    def _describe(self, blocks, prompt):
        it, _, tin, tout = self._interact("C", blocks + [{"type": "text", "text": prompt}],
                                          system="You describe business documents and screens accurately.")
        return (it.output_text or "").strip(), C.cost(C.CFG["lanes"]["C"], tin, tout), tin + tout

    # ------------------------------------------------------------ lane D
    def decide(self, item_input, questions, item_id=None):
        if self.dry:
            return self._fake("D", item_id, questions)
        t0 = time.perf_counter()
        try:
            r = self.client.decisions.create(model=C.CFG["lanes"]["D"]["model"], input=item_input,
                                             questions=C.decision_questions(questions))
        except Exception as e:                                   # noqa: BLE001
            return _record(error=f"{type(e).__name__}: {e}"[:300])
        ms = (time.perf_counter() - t0) * 1000
        rec = _record(latency_ms=ms, tokens_in=(r.usage.total_input_tokens if r.usage else 0) or 0)
        if r.prompt_feedback and r.prompt_feedback.block_reason:
            rec["error"] = f"blocked:{r.prompt_feedback.block_reason}"
            return rec
        for qid, spec in questions.items():
            a = (r.answers or {}).get(qid)
            if a is None:
                rec["invalid"].append(qid); continue
            if a.type == "noul":
                rec["answers"][qid] = "yes" if a.noul >= 0.5 else "no"
                rec["probabilities"][qid] = {"yes": a.noul}
            elif a.type == "choice":
                rec["answers"][qid] = a.choice
                rec["probabilities"][qid] = dict(a.probabilities)
            else:   # score: the most likely level (the expected value is kept too)
                rec["answers"][qid] = max(a.probabilities, key=a.probabilities.get)
                rec["probabilities"][qid] = {**a.probabilities, "expected": a.score}
            rec["confidence"][qid] = a.confidence
        return rec

    # ------------------------------------------------------------ dry run
    def _fake(self, lane, item_id, questions):
        """Plausible fake answers so the whole pipeline can be tested without a key or money."""
        rec = _record(latency_ms=self.rng.uniform(250, 2500), tokens_in=self.rng.randint(300, 5000),
                      tokens_out=0 if lane == "D" else self.rng.randint(40, 120), model_version="dry-run")
        for qid, spec in questions.items():
            right = self.truth.get(item_id, {}).get(qid, {}).get("answer")
            ok = self.rng.random() < 0.85
            rec["answers"][qid] = right if ok and right else self.rng.choice(C.allowed(spec))
            if lane == "D":
                rec["confidence"][qid] = self.rng.uniform(0.9, 0.99) if ok else self.rng.uniform(0.4, 0.95)
        return rec
