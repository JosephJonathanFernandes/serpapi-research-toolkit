"""Create blind label-audit sheets for a human (and a second rater).

    python eval/audit/make_audit_sheet.py

Writes (agent labels are deliberately NOT included, to avoid anchoring):
  eval/audit/audit_sheet.csv        40 stratified pairs for rater 1 (fixed seed)
  eval/audit/audit_sheet_rater2.csv the first 20 of those, for a second rater

Raters fill `human_label` with one of: contradict | agree | unrelated | unclear.
Judge from the findings sentences shown (and open the link if unsure).
Then run: python eval/audit/score_audit.py
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "research_agent"))
from disagreement import extract_findings  # noqa: E402

AUDIT_DIR = Path(__file__).resolve().parent
DATA = ROOT / "eval" / "claim_pairs_labeled.json"
SEED = 0
PER_CLASS = {"contradict": 14, "agree": 13, "unrelated": 13}
N_RATER2 = 20


def claim_for(topic: str) -> str:
    sys.path.insert(0, str(ROOT / "eval"))
    from run_eval import TOPIC_CLAIMS

    return TOPIC_CLAIMS[topic]


def main() -> None:
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    rng = random.Random(SEED)
    chosen = []
    for label, n in PER_CLASS.items():
        pool = [p for p in payload["pairs"] if p["label"] == label]
        rng.shuffle(pool)
        chosen.extend(pool[:n])
    rng.shuffle(chosen)

    fields = ["pair_id", "split", "topic_claim", "title_a", "findings_a", "link_a",
              "title_b", "findings_b", "link_b", "human_label", "notes"]
    dev = set(payload["dev_topics"])

    def row(p):
        a, b = p["paper_a"], p["paper_b"]
        return {
            "pair_id": p["id"],
            "split": "dev" if p["topic"] in dev else "test",
            "topic_claim": claim_for(p["topic"]),
            "title_a": a["title"], "findings_a": extract_findings(a.get("abstract") or a.get("snippet", "")),
            "link_a": a.get("link", ""),
            "title_b": b["title"], "findings_b": extract_findings(b.get("abstract") or b.get("snippet", "")),
            "link_b": b.get("link", ""),
            "human_label": "", "notes": "",
        }

    for name, subset in (("audit_sheet.csv", chosen), ("audit_sheet_rater2.csv", chosen[:N_RATER2])):
        with (AUDIT_DIR / name).open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(row(p) for p in subset)
        print(f"wrote {AUDIT_DIR / name} ({len(subset)} pairs)")


if __name__ == "__main__":
    main()
