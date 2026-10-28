"""Score the human audit: label-change rate, Cohen's kappa, and the 15% verdict.

    python eval/audit/score_audit.py

Reads eval/audit/audit_sheet.csv (rater 1) and optionally audit_sheet_rater2.csv.
Pairs the rater marked 'unclear' are reported separately and excluded from kappa.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

AUDIT_DIR = Path(__file__).resolve().parent
DATA = AUDIT_DIR.parent / "claim_pairs_labeled.json"
CLASSES = ["contradict", "agree", "unrelated"]
THRESHOLD = 0.15


def read(path: Path) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {r["pair_id"]: r["human_label"].strip().lower() for r in csv.DictReader(f) if r["human_label"].strip()}


def kappa(a: list, b: list) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in CLASSES)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def main() -> None:
    agent = {p["id"]: p["label"] for p in json.loads(DATA.read_text(encoding="utf-8"))["pairs"]}
    r1_path = AUDIT_DIR / "audit_sheet.csv"
    if not r1_path.exists():
        print("Run make_audit_sheet.py first.")
        return
    r1 = read(r1_path)
    if not r1:
        print("audit_sheet.csv has no human_label values yet. Fill it in, then rerun.")
        return

    unclear = [k for k, v in r1.items() if v == "unclear"]
    ids = [k for k, v in r1.items() if v in CLASSES]
    changed = [k for k in ids if r1[k] != agent[k]]
    rate = len(changed) / len(ids) if ids else float("nan")
    print(f"Rater 1 labeled {len(r1)} pairs ({len(unclear)} unclear, {len(ids)} scored).")
    print(f"Agent label changed on {len(changed)}/{len(ids)} = {rate:.0%}")
    print(f"Agent-vs-human Cohen's kappa: {kappa([agent[k] for k in ids], [r1[k] for k in ids]):.2f}")
    for c in CLASSES:
        n = sum(agent[k] == c for k in ids)
        ch = sum(agent[k] == c and r1[k] != c for k in ids)
        print(f"  agent '{c}': {ch}/{n} changed")
    if rate > THRESHOLD:
        print(f"VERDICT: change rate {rate:.0%} > {THRESHOLD:.0%}. Treat every number in eval/results.md as noise.")
    else:
        print(f"VERDICT: change rate {rate:.0%} <= {THRESHOLD:.0%}. Numbers are provisional but usable; relabel the changed pairs.")

    r2_path = AUDIT_DIR / "audit_sheet_rater2.csv"
    if r2_path.exists():
        r2 = read(r2_path)
        both = [k for k in r1 if k in r2 and r1[k] in CLASSES and r2[k] in CLASSES]
        if both:
            print(f"Rater1-vs-rater2 kappa on {len(both)} shared pairs: {kappa([r1[k] for k in both], [r2[k] for k in both]):.2f}")


if __name__ == "__main__":
    main()
