"""Fetch the 20 public items for the second decisions test, and write items.csv.

Only this script, the item IDs and the labels are kept in the repo. The files go to
public/cache/ (git-ignored), so every licence stays with its source.

  uv run --project ../harness python fetch_public.py messages receipts      # no credentials needed
  ROBOFLOW_API_KEY=... uv run --project ../harness python fetch_public.py parcels
  SEC_USER_AGENT="Name contact@example.com" uv run --project ../harness python fetch_public.py contracts
  uv run --project ../harness python fetch_public.py compose                 # items.csv from the parts

Labels made by a model (Claude, while preparing the test) rather than by a dataset are in
model_labels.yaml and marked label_source: model+rule. compose checks each one against a
keyword rule and lists every disagreement in label_check.md.
"""
import csv, html, io, json, os, pathlib, random, re, sys, time, urllib.parse, urllib.request, zipfile
import yaml

HERE = pathlib.Path(__file__).resolve().parent
CACHE, PARTS = HERE / "cache", HERE / "parts"
SEED = 20261124
HF_ROWS = "https://datasets-server.huggingface.co"


def get(url, headers=None, raw=False):
    req = urllib.request.Request(url, headers={"User-Agent": "decisions-test-fetch/1.0", **(headers or {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    return data if raw else json.loads(data)


def hf_sha(dataset):
    return get(f"https://huggingface.co/api/datasets/{dataset}")["sha"]


def save_part(name, rows):
    PARTS.mkdir(exist_ok=True)
    (PARTS / f"{name}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print(f"{name}: {len(rows)} items")


# ------------------------------------------------------------------ Bitext messages
BITEXT = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
# Bitext intent -> Kumo Supply team. Only intents with one obvious team are used.
INTENT_TEAM = {"get_refund": "billing", "payment_issue": "billing", "track_order": "shipping",
               "change_shipping_address": "shipping", "newsletter_subscription": "other"}


def messages():
    rng, sha, rows = random.Random(SEED), hf_sha(BITEXT), []
    for intent, team in INTENT_TEAM.items():
        where = urllib.parse.quote(f"\"intent\"='{intent}'")
        base = f"{HF_ROWS}/filter?dataset={BITEXT}&config=default&split=train&where={where}"
        total = get(base + "&offset=0&length=1")["num_rows_total"]
        got = get(base + f"&offset={rng.randrange(0, max(1, total - 100))}&length=100")
        # messy on purpose: rows flagged Z (typos) or Q (colloquial), with no {{placeholder}}
        pool = [r for r in got["rows"] if re.search("[ZQ]", r["row"]["flags"]) and "{{" not in r["row"]["instruction"]]
        pick = rng.choice(pool)
        r, idx = pick["row"], pick["row_idx"]
        iid = f"PM_{intent}_{idx}"
        CACHE.mkdir(exist_ok=True)
        (CACHE / f"{iid}.txt").write_text(r["instruction"])
        rows.append(dict(item_id=iid, type="message", file=f"{iid}.txt", source="Bitext customer support",
                         source_id=f"train row {idx} of intent {intent} @ {sha[:12]}", url=f"https://huggingface.co/datasets/{BITEXT}",
                         licence="CDLA-Sharing-1.0",
                         label_source="dataset (team, refund_request) + rule (personal_data) + model+rule (urgency)",
                         labels={"team": team, "refund_request": "yes" if intent == "get_refund" else "no",
                                 "personal_data": "no"},   # rows with placeholders were excluded, so no personal data
                         meta={"intent": intent, "flags": r["flags"], "text": r["instruction"]}))
    save_part("messages", rows)


URGENT = re.compile(r"\b(urgent|asap|immediately|right now|right away|now|today|charged twice|never arrived|still (?:haven'?t|not))\b", re.I)


def urgency_rule(text, team):
    if team == "other":
        return 0
    return 2 if URGENT.search(text) else 1


# ------------------------------------------------------------------ CORD receipts
CORD = "naver-clova-ix/cord-v2"


def receipts():
    rng, sha = random.Random(SEED), hf_sha(CORD)
    got = get(f"{HF_ROWS}/rows?dataset={CORD}&config=default&split=test&offset=0&length=100")
    above, below = [], []
    for r in got["rows"]:
        gt = json.loads(r["row"]["ground_truth"])["gt_parse"]
        t = (gt.get("total") or {}).get("total_price")
        if not isinstance(t, str) or not re.fullmatch(r"[\d.,]+", t.strip()):
            continue
        val = int(re.sub(r"\D", "", t))
        if val < 1000:                         # not a plain Rupiah amount: skip rather than guess
            continue
        (above if val > 100_000 else below).append((r["row_idx"], val, r["row"]["image"]["src"]))
    picks = rng.sample(above, 2) + rng.sample(below, 3)     # a mix on both sides of the threshold
    rows = []
    for idx, val, src in picks:
        iid = f"PR_cord_test_{idx}"
        CACHE.mkdir(exist_ok=True)
        (CACHE / f"{iid}.jpg").write_bytes(get(src, raw=True))
        rows.append(dict(item_id=iid, type="receipt", file=f"{iid}.jpg", source="CORD v2", source_id=f"test row {idx} @ {sha[:12]}",
                         url=f"https://huggingface.co/datasets/{CORD}", licence="CC-BY-4.0", label_source="dataset",
                         labels={"image_kind": "receipt", "total_above": "yes" if val > 100_000 else "no",
                                 "japanese_text": "no", "next_action": "file_expense"},
                         meta={"total": val}))
    save_part("receipts", rows)


# ------------------------------------------------------------------ Roboflow parcels (needs ROBOFLOW_API_KEY)
ROBOFLOW = [("ameesh", "damaged-package-detection-xjkol"), ("labels-kpsk0", "damaged-boxes")]


def parcels():
    key = os.environ.get("ROBOFLOW_API_KEY") or sys.exit("Set ROBOFLOW_API_KEY (free Roboflow account).")
    for ws, proj in ROBOFLOW:
        info = get(f"https://api.roboflow.com/{ws}/{proj}?api_key={key}")
        lic = (info.get("project") or {}).get("license") or info.get("license")
        print(f"{ws}/{proj}: licence {lic!r}, classes {list((info.get('project') or {}).get('classes', {}))}")
        if not lic or "CC BY 4.0" not in str(lic).upper().replace("-", " "):
            print("  not CC BY 4.0, skipped"); continue
        ver = max(v["id"].split("/")[-1] for v in info.get("versions", []))
        link = get(f"https://api.roboflow.com/{ws}/{proj}/{ver}/coco?api_key={key}")["export"]["link"]
        z = zipfile.ZipFile(io.BytesIO(get(link, raw=True)))
        out = CACHE / "roboflow" / proj
        out.mkdir(parents=True, exist_ok=True)
        z.extractall(out)
        print(f"  extracted to {out}: now pick items with pick_parcels() (needs a look at the photos)")
        return ws, proj, ver, str(lic)
    sys.exit("No CC BY 4.0 damaged-box dataset found: stop and report back (the plan says not to generate one).")


# ------------------------------------------------------------------ SEC EDGAR contracts (needs SEC_USER_AGENT)
RENEW = re.compile(r"automatically\s+(?:be\s+)?renew|successive\s+(?:\w+\s+){0,3}(?:term|period)s?", re.I)


def contracts():
    ua = os.environ.get("SEC_USER_AGENT") or sys.exit('Set SEC_USER_AGENT="Name email" (SEC requires a contact).')
    h = {"User-Agent": ua}
    rng = random.Random(SEED)
    found = {"yes": [], "no": []}
    for want, q in [("yes", '"automatically renew"'), ("no", '"supply agreement" "term"')]:
        res = get(f"https://efts.sec.gov/LATEST/search-index?q={urllib.parse.quote(q)}&forms=8-K,10-K,10-Q"
                  "&dateRange=custom&startdt=2024-01-01&enddt=2025-12-31", h)
        hits = [x for x in res["hits"]["hits"] if str(x["_source"].get("file_type", "")).startswith("EX-10")]
        rng.shuffle(hits)
        for x in hits:
            if len(found[want]) >= (3 if want == "yes" else 2):
                break
            adsh, fname = x["_id"].split(":")
            cik = x["_source"]["ciks"][0].lstrip("0")
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh.replace('-', '')}/{fname}"
            time.sleep(0.2)                                       # SEC asks for <= 10 requests per second
            text = html_to_text(get(url, h, raw=True).decode("utf-8", "replace"))
            words = len(text.split())
            has = bool(RENEW.search(text))
            if not 1500 <= words <= 4500 or has != (want == "yes"):   # ~3-8 pages; rule must agree
                continue
            found[want].append((adsh, fname, url, text, x["_source"].get("display_names", [""])[0]))
    rows = []
    for want, docs in found.items():
        for adsh, fname, url, text, who in docs:
            iid = f"PC_{adsh}"
            to_pdf(text, CACHE / f"{iid}.pdf")
            rows.append(dict(item_id=iid, type="contract", file=f"{iid}.pdf", source="SEC EDGAR", source_id=f"{adsh} {fname}",
                             url=url, licence="Public filing (U.S. government website)", label_source="rule (auto_renewal) + model+rule pending",
                             labels={"doc_type": "contract", "auto_renewal": want, "language": "english",
                                     "money_decision": {"answer": "yes", "also_ok": ["no"]}},   # Kumo is not a party: either is defensible
                             meta={"filer": who, "renewal_clause": (RENEW.search(text).group(0) if want == "yes" else None)}))
    save_part("contracts", rows)


def html_to_text(s):
    s = re.sub(r"(?is)<(script|style).*?</\1>", " ", s)
    s = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</tr>", "\n", s)
    s = html.unescape(re.sub(r"<[^>]+>", " ", s))
    return "\n".join(re.sub(r"[ \t\xa0]+", " ", ln).strip() for ln in s.splitlines() if ln.strip())


def to_pdf(text, path):
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate
    st = getSampleStyleSheet()["BodyText"]
    paras = [Paragraph(html.escape(p), st) for p in text.split("\n")]
    path.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(path), pagesize=LETTER).build(paras)


# ------------------------------------------------------------------ compose items.csv
def compose():
    model = yaml.safe_load((HERE / "model_labels.yaml").read_text()) if (HERE / "model_labels.yaml").exists() else {}
    rows, checks = [], []
    for part in sorted(PARTS.glob("*.json")):
        for r in json.loads(part.read_text()):
            m = (model or {}).get(r["item_id"], {})
            if r["type"] == "message":
                rule = urgency_rule(r["meta"]["text"], r["labels"]["team"])
                if "urgency" not in m:
                    sys.exit(f"{r['item_id']}: urgency not labeled yet in model_labels.yaml")
                r["labels"]["urgency"] = {"answer": m["urgency"], "also_ok": m.get("urgency_also_ok", [])}
                if rule != m["urgency"]:
                    checks.append(f"- {r['item_id']}: model says urgency {m['urgency']}, keyword rule says {rule}. Text: \"{r['meta']['text']}\"")
            rows.append(r)
    with open(HERE / "items.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "type", "file", "source", "source_id", "url", "licence", "label_source", "labels"])
        for r in rows:
            w.writerow([r[k] for k in ["item_id", "type", "file", "source", "source_id", "url", "licence", "label_source"]]
                       + [json.dumps(r["labels"], ensure_ascii=False)])
    (HERE / "label_check.md").write_text("# Model labels vs keyword rule\n\n" + ("\n".join(checks) or "No disagreements.") + "\n")
    print(f"items.csv: {len(rows)} items; {len(checks)} label disagreements (see label_check.md)")


if __name__ == "__main__":
    for step in sys.argv[1:] or ["messages", "receipts"]:
        globals()[step]()
