"""Shared loading, prompts and scoring for the decisions test. Used by every lane."""
import base64, json, mimetypes, os, pathlib, yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent                       # tests/decisions
CFG = yaml.safe_load((HERE / "config.yaml").read_text())

TYPE_BY_PREFIX = {"T": "text", "I": "image", "V": "video", "P": "pdf"}
YN = {True: "yes", False: "no", "yes": "yes", "no": "no"}


def norm(v):
    """YAML reads yes/no as booleans and scores as ints: make every answer a string."""
    if isinstance(v, bool):
        return YN[v]
    return str(v)


def load_questions(path):
    q = yaml.safe_load(pathlib.Path(path).read_text())
    for group in q.values():
        for spec in group.values():
            if spec["type"] == "choice":
                spec["options"] = {str(k): v for k, v in spec["options"].items()}
            if spec["type"] == "score":
                spec["levels"] = {int(k): v for k, v in spec["levels"].items()}
    return q


def load_labels(path):
    out = {}
    for item, qs in yaml.safe_load(pathlib.Path(path).read_text()).items():
        out[item] = {}
        for qid, v in qs.items():
            if qid == "hard":
                continue
            if isinstance(v, dict):
                out[item][qid] = {"answer": norm(v["answer"]), "also_ok": [norm(x) for x in v.get("also_ok", [])]}
            else:
                out[item][qid] = {"answer": norm(v), "also_ok": []}
    return out


def questions_for(qs, item_type, ids=None):
    """The question set for one item: shared + its type (kumo), or an explicit list (public)."""
    pool = {**qs.get("shared", {}), **qs.get(item_type, {})}
    if ids:
        return {k: pool[k] for k in ids}
    return pool


def allowed(spec):
    """Allowed answer values, in the order every lane sees them.
    The Decisions API sorts choice keys alphabetically, so every lane does the same."""
    if spec["type"] == "noul":
        return ["no", "yes"]
    if spec["type"] == "choice":
        return sorted(spec["options"])
    return [str(i) for i in sorted(spec["levels"])]


def render_questions(questions):
    """The question block for the LLM lanes. Same wording and option order as lane D."""
    lines = []
    for qid, s in questions.items():
        kind = {"noul": "yes or no", "choice": "choose one", "score": "score"}[s["type"]]
        lines.append(f"{qid} ({kind}): {s['instructions']}")
        if s["type"] == "choice":
            lines += [f"  - {k}: {s['options'][k]}" for k in allowed(s)]
        elif s["type"] == "score":
            lines += [f"  - {i}: {s['levels'][i]}" for i in sorted(s["levels"])]
    return "\n".join(lines)


def json_schema(questions):
    return {
        "type": "object",
        "properties": {qid: {"type": "string", "enum": allowed(s)} for qid, s in questions.items()},
        "required": list(questions),
        "propertyOrdering": list(questions),
    }


def decision_questions(questions):
    """The same questions in the Decisions API format."""
    out = {}
    for qid, s in questions.items():
        q = {"type": s["type"], "instructions": s["instructions"]}
        if s["type"] == "choice":
            q["criteria"] = {k: s["options"][k] for k in allowed(s)}
        elif s["type"] == "score":
            q["criteria"] = [s["levels"][i] for i in sorted(s["levels"])]
        out[qid] = q
    return out


MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp",
        ".mp4": "video/mp4", ".pdf": "application/pdf"}


def media_block(path, resolution=None):
    """One content block for a media file, inline base64. Text items return their text."""
    p = pathlib.Path(path)
    if p.suffix == ".txt":
        return p.read_text()
    mime = MIME.get(p.suffix.lower()) or mimetypes.guess_type(p.name)[0]
    kind = {"image": "image", "video": "video", "application": "document"}[mime.split("/")[0]]
    block = {"type": kind, "data": base64.b64encode(p.read_bytes()).decode(), "mime_type": mime}
    if resolution:
        block["resolution"] = resolution
    return block


def score(answer, label):
    if answer is None:
        return False, False
    strict = answer == label["answer"]
    return strict, strict or answer in label["also_ok"]


def cost(lane_cfg, tokens_in, tokens_out):
    return (tokens_in * lane_cfg["price_in"] + tokens_out * lane_cfg["price_out"]) / 1e6


def load_env():
    """Read harness/.env (never committed) into the environment."""
    f = HERE / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
