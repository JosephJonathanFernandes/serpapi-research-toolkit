"""
Build phrase lexicon V3 by greedy selection on the DEV topics only.

Why: lexicon V2 was hand-written after reading snippets from all 10 topics, so
its test-split score is contaminated. V3 is chosen by a mechanical, reproducible
procedure that only ever looks at the dev split:

  1. Start from V1 (written before any evaluation).
  2. Candidate pool = the patterns V2 added on top of V1.
  3. Greedy forward selection: repeatedly add the single candidate (or, when no
     single one helps, the affirmative+null pair) that most increases
     contradiction F1 on dev snippets; stop when nothing improves it.

Remaining caveat (stated in the README): the candidate *pool* was written by
someone who had seen test snippets, so V3's test score is cleaner than V2's but
not fully clean. The genuinely unseen check is SciFact (eval/scifact/).

Usage:
    python eval/build_lexicon_v3.py          # writes examples/research_agent/lexicon_v3.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "research_agent"))

from disagreement import (  # noqa: E402
    AFFIRMATIVE_PATTERNS_V1,
    AFFIRMATIVE_PATTERNS_V2,
    NULL_OR_CONTRARY_PATTERNS_V1,
    NULL_OR_CONTRARY_PATTERNS_V2,
    _matches,
)

OUT = ROOT / "examples" / "research_agent" / "lexicon_v3.json"


def dev_pairs() -> List[Dict]:
    payload = json.loads((ROOT / "eval" / "claim_pairs_labeled.json").read_text(encoding="utf-8"))
    dev = set(payload["dev_topics"])
    return [p for p in payload["pairs"] if p["topic"] in dev]


def _text(paper: Dict) -> str:
    return f"{paper.get('title', '')} {paper.get('snippet', '')}".strip()


def f1(pairs: List[Dict], pos: List[str], neg: List[str]) -> Tuple[float, int, int, int]:
    tp = fp = fn = 0
    for p in pairs:
        a, b = _text(p["paper_a"]), _text(p["paper_b"])
        pa, na, pb, nb = _matches(pos, a), _matches(neg, a), _matches(pos, b), _matches(neg, b)
        pred = bool((pa and nb) or (pb and na))
        truth = p["label"] == "contradict"
        tp += pred and truth
        fp += pred and not truth
        fn += truth and not pred
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return (2 * prec * rec / (prec + rec) if prec + rec else 0.0), tp, fp, fn


def _apply(pos: List[str], neg: List[str], move) -> Tuple[List[str], List[str]]:
    new_pos = pos + [p for side, p in move if side == "pos"]
    new_neg = neg + [p for side, p in move if side == "neg"]
    return new_pos, new_neg


def main() -> int:
    pairs = dev_pairs()
    pos, neg = list(AFFIRMATIVE_PATTERNS_V1), list(NULL_OR_CONTRARY_PATTERNS_V1)
    pool = [("pos", p) for p in AFFIRMATIVE_PATTERNS_V2 if p not in pos] + [
        ("neg", p) for p in NULL_OR_CONTRARY_PATTERNS_V2 if p not in neg
    ]
    best, tp, fp, fn = f1(pairs, pos, neg)
    log = [f"start (V1): dev F1={best:.3f} tp={tp} fp={fp} fn={fn}"]
    while pool:
        # Try single additions first; if none helps, try one affirmative + one
        # null pattern together (a pair needs a cue on each side to fire).
        moves = [[c] for c in pool]
        scored = [(f1(pairs, *_apply(pos, neg, m)), m) for m in moves]
        if max(s[0][0] for s in scored) <= best:
            moves = [[a, b] for a in pool if a[0] == "pos" for b in pool if b[0] == "neg"]
            scored = [(f1(pairs, *_apply(pos, neg, m)), m) for m in moves]
        if not scored:
            break
        (score, tp, fp, fn), move = max(scored, key=lambda s: s[0][0])
        if score <= best:
            break
        best = score
        pos, neg = _apply(pos, neg, move)
        for c in move:
            pool.remove(c)
        desc = " & ".join(f"{side} {pat!r}" for side, pat in move)
        log.append(f"+ {desc}: dev F1={best:.3f} tp={tp} fp={fp} fn={fn}")

    print("\n".join(log))
    OUT.write_text(
        json.dumps(
            {
                "provenance": (
                    "Greedy forward selection on DEV topics only (eval/build_lexicon_v3.py). "
                    "Base = V1; candidate pool = V2 additions (pool authored with exposure to test snippets)."
                ),
                "dev_topics": sorted({p["topic"] for p in pairs}),
                "dev_f1": round(best, 4),
                "selection_log": log,
                "affirmative": pos,
                "null_or_contrary": neg,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {OUT.relative_to(ROOT)} ({len(pos)} affirmative, {len(neg)} null/contrary patterns)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
