"""Run the decisions test. See the README at the repository root.

  uv run python run.py --dry-run                      # whole pipeline, no API, no cost
  uv run python run.py --confirm-paid                 # the real thing: 3 runs, all lanes, both tests
  uv run python run.py --confirm-paid --lanes D --types image --resolution medium --tag medium

Needs GEMINI_API_KEY in harness/.env, from a project on the PAID tier
(the free tier may use the data, and its rate limits distort the timings).
"""
import argparse, csv, datetime as dt, json, pathlib, platform, random, sys
import common as C
from lanes import Lanes

LANES_FOR = {"text": "ABD", "image": "ABCD", "video": "ABCD", "pdf": "ABCD"}
PUBLIC_KIND = {"message": "text", "receipt": "image", "parcel": "image", "contract": "pdf"}


def kumo_items():
    qs, labels = C.load_questions(C.ROOT / "questions.yaml"), C.load_labels(C.ROOT / "labels.yaml")
    items = []
    for f in sorted((C.ROOT / "kumo").glob("*/*")):
        iid = f.stem
        if iid not in labels:
            continue
        typ = C.TYPE_BY_PREFIX[iid[0]]
        items.append(dict(id=iid, test="kumo", type=typ, kind=typ, path=f,
                          questions=C.questions_for(qs, typ), labels=labels[iid]))
    missing = set(labels) - {i["id"] for i in items}
    if missing:
        print(f"WARNING: no file for {sorted(missing)}: those items are skipped", file=sys.stderr)
    return items


def public_items():
    pub = C.ROOT / "public"
    if not (pub / "items.csv").exists():
        print("WARNING: public/items.csv missing: run public/fetch_public.py first", file=sys.stderr)
        return []
    qs = C.load_questions(pub / "questions.yaml")
    items = []
    for row in csv.DictReader(open(pub / "items.csv")):
        path = pub / "cache" / row["file"]
        if not path.exists():
            print(f"WARNING: {path} missing: run public/fetch_public.py", file=sys.stderr); continue
        lab = {k: ({"answer": C.norm(v["answer"]), "also_ok": [C.norm(x) for x in v.get("also_ok", [])]}
                   if isinstance(v, dict) else {"answer": C.norm(v), "also_ok": []})
               for k, v in json.loads(row["labels"]).items()}
        items.append(dict(id=row["item_id"], test="public", type=row["type"], kind=PUBLIC_KIND[row["type"]],
                          path=path, questions=qs[row["type"]], labels=lab))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="both", choices=["kumo", "public", "both"])
    ap.add_argument("--lanes", default="ABCD")
    ap.add_argument("--types", default=None, help="comma list: text,image,video,pdf (or public types)")
    ap.add_argument("--runs", type=int, default=C.CFG["runs"])
    ap.add_argument("--resolution", default=C.CFG["decision_resolution"], help="lane D media resolution")
    ap.add_argument("--tag", default="", help="suffix for the output folder, e.g. 'medium'")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--confirm-paid", action="store_true", help="I confirm the key is on the paid tier")
    a = ap.parse_args()

    items = (kumo_items() if a.test in ("kumo", "both") else []) + (public_items() if a.test in ("public", "both") else [])
    if a.types:
        want = set(a.types.split(","))
        items = [i for i in items if i["type"] in want or i["kind"] in want]

    if a.dry_run:
        client, truth = None, {i["id"]: i["labels"] for i in items}
    else:
        if not a.confirm_paid:
            sys.exit("Refusing to run: pass --confirm-paid once the key's project is on the paid tier.")
        C.load_env()
        from google import genai
        client, truth = genai.Client(http_options={"api_version": "v1beta"}), None
    lanes = Lanes(client, dry_truth=truth, seed=7)

    day = dt.date.today().isoformat()
    name = f"decisions-test-{day}" + (f"-{a.tag}" if a.tag else "") + ("-dryrun" if a.dry_run else "")
    # dry runs never go to analytics/, so fake numbers can't end up in the article
    out = (C.HERE / "dryrun-output" if a.dry_run else C.ROOT / "results") / name
    out.mkdir(parents=True, exist_ok=True)
    import google.genai as g
    meta = dict(date=day, started=dt.datetime.now().isoformat(timespec="seconds"), dry_run=a.dry_run,
                sdk="google-genai " + g.__version__, python=platform.python_version(), machine=platform.platform(),
                lanes={k: C.CFG["lanes"][k] for k in a.lanes}, decision_resolution=a.resolution,
                temperature=C.CFG.get("temperature"), prices_checked=C.CFG["checked"], runs=a.runs,
                items=[i["id"] for i in items], model_versions={})
    raw = open(out / "raw.jsonl", "a")

    spent, by_run, stop = 0.0, {}, None
    for run in range(1, a.runs + 1):
        if stop:
            break
        if run == 2 and not a.dry_run:                          # pilot check: project the full bill from run 1
            r1, b1 = by_run.get(1, 0.0), by_run.get("B1", 0.0)
            proj = spent + (r1 - b1) * (a.runs - 1)
            print(f"pilot: run 1 cost ${r1:.4f}, projected total ${proj:.4f}")
            if proj > C.CFG["budget_projection_usd"]:
                stop = f"pilot projection ${proj:.4f} over budget"; break
        order = items[:]
        random.Random(1000 + run).shuffle(order)                 # fixed seed, different order per run
        for lane in a.lanes:
            if stop or run > C.CFG.get("runs_per_lane", {}).get(lane, a.runs):
                continue
            todo = [i for i in order if lane in LANES_FOR[i["kind"]]]
            for w in todo[: C.CFG["warmup_calls"]]:              # warm-up, not recorded (but paid)
                wr, wp = call(lanes, lane, w, a.resolution)
                spent += C.cost(C.CFG["lanes"][lane], wr["tokens_in"], wr["tokens_out"]) + (wp["cost"] or 0)
            n_ok = n_err = 0
            for it in todo:
                if spent >= C.CFG["budget_usd"]:
                    stop = f"budget cap ${C.CFG['budget_usd']} reached"; break
                rec, pre = call(lanes, lane, it, a.resolution)
                if rec.get("model_version"):
                    meta["model_versions"][lane] = rec["model_version"]
                lc = C.CFG["lanes"][lane]
                c = C.cost(lc, rec["tokens_in"], rec["tokens_out"])
                spent += c + (pre["cost"] or 0)
                by_run[run] = by_run.get(run, 0.0) + c + (pre["cost"] or 0)
                if lane == "B":
                    by_run[f"B{run}"] = by_run.get(f"B{run}", 0.0) + c
                for qid in it["questions"]:
                    ans = rec["answers"].get(qid)
                    strict, lenient = C.score(ans, it["labels"][qid])
                    raw.write(json.dumps(dict(
                        run=run, test=it["test"], lane=lane, type=it["type"], item_id=it["id"], question_id=qid,
                        answer=ans, label=it["labels"][qid]["answer"], confidence=rec["confidence"].get(qid),
                        probabilities=rec["probabilities"].get(qid), correct_strict=strict, correct_lenient=lenient,
                        invalid=qid in rec["invalid"], latency_ms=rec["latency_ms"], input_tokens=rec["tokens_in"],
                        output_tokens=rec["tokens_out"], cost_usd=c, preprocess_ms=pre["ms"],
                        preprocess_cost_usd=pre["cost"], preprocess_steps=pre["steps"],
                        jev_tokens=rec["tokens_in"] if lane == "C" else None, error=rec["error"],
                    ), ensure_ascii=False) + "\n")
                raw.flush()
                n_err += bool(rec["error"]); n_ok += sum(C.score(rec['answers'].get(q), it['labels'][q])[0] for q in it['questions'])
                if rec["error"] and n_err <= 2:
                    print(f"  error {lane} {it['id']}: {rec['error'][:160]}")
            print(f"run {run} lane {lane}: {len(todo)} items, {n_ok} correct answers, {n_err} errors, spent so far ${spent:.4f}")
    meta["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    meta["spent_usd"], meta["stopped"] = round(spent, 5), stop
    print(f"total spent ${spent:.4f}" + (f"; STOPPED: {stop}" if stop else ""))
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nWrote {out}")


def call(lanes, lane, it, resolution):
    pre = dict(ms=None, cost=None, steps=None)
    if lane in "AB":
        res = C.CFG.get("llm_resolution")
        return lanes.llm(lane, C.media_block(it["path"], res), it["questions"], it["id"]), pre
    if lane == "C":
        text, ms, cost, _, steps = lanes.convert(it["path"])
        pre = dict(ms=ms, cost=cost, steps=steps)
        return lanes.llm("C", text, it["questions"], it["id"]), pre
    res = None if it["kind"] in ("text", "pdf") else resolution
    return lanes.decide(C.media_block(it["path"], res), it["questions"], it["id"]), pre


if __name__ == "__main__":
    main()
