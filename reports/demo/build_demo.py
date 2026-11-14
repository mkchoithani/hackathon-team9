"""Build the Team 9 demo page: one self-contained HTML file with the measured data embedded.

    python reports/demo/build_demo.py . reports/demo/index.html \
        [--pred-base results/eval/pred-base.jsonl] [--pred-ft results/eval/pred-finetuned.jsonl]

Predictions (pred-*.jsonl from scripts/07_predict.py) are optional; without them the explorer shows
labels only and the scorecard says the evaluation is pending.
"""
import argparse
import csv
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("repo")
ap.add_argument("out")
ap.add_argument("--pred-base")
ap.add_argument("--pred-ft")
ap.add_argument("--report", default=str(Path(__file__).resolve().parents[1] / "data"),
                help="dir with runs.csv / steps.csv / nccl.csv / load.csv (default: reports/data)")
a = ap.parse_args()
repo = Path(a.repo)
rep = Path(a.report)


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


passages = []
for line in open(repo / "data/sft/test.jsonl"):
    r = json.loads(line)
    lab = r["label"]
    passages.append({"id": r["id"], "ticker": r["ticker"], "company": r["company"], "date": r["filing_date"],
                     "item": r["item"], "title": r["item_title"], "text": r["text"],
                     "cls": lab["risk_class"], "sev": lab["severity"], "quote": lab["evidence_quote"],
                     "why": lab.get("rationale", "")})
gold = {}
for g in read_csv(repo / "data/sft/gold_review.csv"):
    if g.get("human_class"):
        gold[g["id"]] = {"cls": g["human_class"], "sev": g.get("human_severity") or None, "notes": g.get("notes") or ""}


def load_preds(p):
    if not p:
        return None
    out = {}
    for line in open(p):
        r = json.loads(line)
        out[r["id"]] = {"parsed": r.get("parsed"), "raw": (r.get("raw") or "")[:1500], "lat": r.get("latency_s")}
    return out


data = {
    "passages": passages,
    "gold": gold,
    "preds": {"base": load_preds(a.pred_base), "finetuned": load_preds(a.pred_ft)},
    "runs": read_csv(rep / "runs.csv"),
    "steps": read_csv(rep / "steps.csv"),
    "nccl": read_csv(rep / "nccl.csv"),
    "load": read_csv(rep / "load.csv"),
}
blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
html = Path(__file__).with_name("demo_template.html").read_text().replace("/*__DATA__*/null", blob)
Path(a.out).write_text(html)
print(f"wrote {a.out} ({len(html) / 1024:.0f} KB) | passages {len(passages)} | gold {len(gold)} | "
      f"base preds {len(data['preds']['base'] or {})} | finetuned preds {len(data['preds']['finetuned'] or {})}")
