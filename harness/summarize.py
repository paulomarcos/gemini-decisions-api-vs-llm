"""Turn raw.jsonl into summary.csv, hard_cases.csv, confident_wrong.csv and the cover chart.

  uv run python summarize.py ../results/decisions-test-YYYY-MM-DD
"""
import collections, csv, json, pathlib, statistics, sys

HARD = ["T1_angry_bilingual_email", "T4_restaurant_newsletter", "I1_receipt_photo", "I4_shipping_label_photo",
        "V1_label_error_flash", "P1_carrier_invoice", "P3_scanned_delivery_note_ja"]
LANE_LABEL = {"A": "Cheap LLM", "B": "Stronger LLM", "C": "Convert first", "D": "Decisions"}


def p95(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(0.95 * (len(xs) - 1)))] if xs else None


def pct(n, d):
    return round(100 * n / d, 1) if d else None


def main(folder):
    out = pathlib.Path(folder)
    rows = [json.loads(l) for l in open(out / "raw.jsonl")]
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["test"], r["type"], r["lane"])].append(r)

    summary = []
    for (test, typ, lane), rs in sorted(groups.items()):
        calls = {}                                           # one entry per (run, item): the call
        for r in rs:
            calls[(r["run"], r["item_id"])] = r
        ok_calls = [c for c in calls.values() if not c["error"]]
        times = [(c["latency_ms"] or 0) + (c["preprocess_ms"] or 0) for c in ok_calls]
        costs = [c["cost_usd"] + (c["preprocess_cost_usd"] or 0) for c in ok_calls]
        answered = [r for r in rs if not r["error"]]
        # consistency: same answer in every run, per (item, question)
        by_q = collections.defaultdict(list)
        for r in rs:
            by_q[(r["item_id"], r["question_id"])].append(r["answer"])
        runs = len({r["run"] for r in rs})
        same = [len(set(v)) == 1 for v in by_q.values() if len(v) == runs]
        row = dict(test=test, type=typ, lane=lane, lane_name=LANE_LABEL[lane], decisions=len(rs),
                   calls=len(calls), errors=len(calls) - len(ok_calls),
                   accuracy_strict=pct(sum(r["correct_strict"] for r in answered), len(answered)),
                   accuracy_lenient=pct(sum(r["correct_lenient"] for r in answered), len(answered)),
                   invalid=sum(r["invalid"] for r in rs),
                   same_answer_all_runs=pct(sum(same), len(same)),
                   median_s=round(statistics.median(times) / 1000, 2) if times else None,
                   p95_s=round(p95(times) / 1000, 2) if times else None,
                   cost_per_item_usd=round(statistics.mean(costs), 6) if costs else None,
                   cost_per_1000_usd=round(1000 * statistics.mean(costs), 3) if costs else None,
                   preprocess_median_s=round(statistics.median([c["preprocess_ms"] for c in ok_calls]) / 1000, 2) if lane == "C" and ok_calls else None,
                   jev_cost_per_1000_usd=round(statistics.mean(c["jev_tokens"] for c in ok_calls) * 0.042 / 1e6 * 1000
                                               + statistics.mean(c["preprocess_cost_usd"] or 0 for c in ok_calls) * 1000, 3)
                                         if lane == "C" and ok_calls else None)
        if lane == "D":
            conf = [r for r in answered if r["confidence"] is not None and r["confidence"] >= 0.9]
            row["pct_conf_ge_0_9"] = pct(len(conf), len(answered))
            row["accuracy_conf_ge_0_9"] = pct(sum(r["correct_strict"] for r in conf), len(conf))
        summary.append(row)
    keys = list(dict.fromkeys(k for r in summary for k in r))
    write(out / "summary.csv", keys, summary)

    # Lane D's confident mistakes: the "what a wrong answer costs" section
    cw = [dict(test=r["test"], run=r["run"], item_id=r["item_id"], question_id=r["question_id"], answer=r["answer"],
               label=r["label"], confidence=round(r["confidence"], 3), correct_lenient=r["correct_lenient"])
          for r in rows if r["lane"] == "D" and r["confidence"] and r["confidence"] >= 0.9 and not r["correct_strict"]]
    write(out / "confident_wrong.csv", ["test", "run", "item_id", "question_id", "answer", "label", "confidence", "correct_lenient"], cw)

    # Hard cases: per item and lane, accuracy over all runs and the answers that were wrong
    hc = []
    for item in HARD:
        for lane in "ABCD":
            rs = [r for r in rows if r["item_id"] == item and r["lane"] == lane and not r["error"]]
            if not rs:
                continue
            wrong = collections.Counter(f"{r['question_id']}={r['answer']} (want {r['label']})" for r in rs if not r["correct_strict"])
            extra = ""
            if lane == "D" and item == "T1_angry_bilingual_email":
                ps = [r["probabilities"] for r in rs if r["question_id"] == "team" and r["probabilities"]]
                if ps:
                    extra = "team probabilities (mean over runs): " + ", ".join(
                        f"{k} {statistics.mean(p.get(k, 0) for p in ps):.2f}" for k in sorted(ps[0], key=lambda k: -ps[0][k])[:3])
            hc.append(dict(item_id=item, lane=lane, lane_name=LANE_LABEL[lane],
                           accuracy_strict=pct(sum(r["correct_strict"] for r in rs), len(rs)),
                           accuracy_lenient=pct(sum(r["correct_lenient"] for r in rs), len(rs)),
                           wrong_answers="; ".join(f"{k} x{v}" for k, v in wrong.most_common()), note=extra))
    write(out / "hard_cases.csv", ["item_id", "lane", "lane_name", "accuracy_strict", "accuracy_lenient", "wrong_answers", "note"], hc)
    chart(out, summary)
    print(f"Wrote summary.csv ({len(summary)} rows), hard_cases.csv, confident_wrong.csv ({len(cw)}), cover_chart.png in {out}")


def write(path, keys, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def chart(out, summary):
    """Cover chart: cost per 1,000 items and median time per item, by input type, cheap LLM vs Decisions.
    Two panels, never two y-axes. Series colors: slots 1 and 2 of the validated default palette."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    types = ["text", "image", "video", "pdf"]
    names = {"text": "Emails", "image": "Photos", "video": "Recordings", "pdf": "PDFs"}
    s = {(r["type"], r["lane"]): r for r in summary if r["test"] == "kumo"}
    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    color = {"A": "#2a78d6", "D": "#eb6834"}
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), facecolor=surface)
    for ax, (metric, title, fmt) in zip(axes, [("cost_per_1000_usd", "Cost per 1,000 items (USD, log scale)", "${:,.2f}"),
                                               ("median_s", "Median time per item (seconds)", "{:.1f}s")]):
        ax.set_facecolor(surface)
        w = 0.36
        for j, lane in enumerate("AD"):
            xs = [i + (j - 0.5) * (w + 0.03) for i in range(len(types))]
            vals = [(s.get((t, lane)) or {}).get(metric) for t in types]
            bars = ax.bar([x for x, v in zip(xs, vals) if v], [v for v in vals if v], width=w, color=color[lane],
                          label=LANE_LABEL[lane], edgecolor=surface, linewidth=2)
            for b, v in zip(bars, [v for v in vals if v]):
                ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom",
                            fontsize=9, color=ink2, xytext=(0, 3), textcoords="offset points")
        if metric == "cost_per_1000_usd":
            from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(range(len(types)), [names[t] for t in types], color=ink)
        ax.set_title(title, loc="left", color=ink, fontsize=12, pad=12)
        ax.tick_params(which="both", colors=ink2, length=0)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper right", ncol=2, labelcolor=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "cover_chart.png", dpi=200, facecolor=surface)


if __name__ == "__main__":
    main(sys.argv[1])
