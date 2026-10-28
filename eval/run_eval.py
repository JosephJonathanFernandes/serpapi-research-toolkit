"""
Reproducible evaluation benchmark for literature contradiction detection across tiers and inputs.

Tiers:
  - Tier 1: Phrase-matching lexicon (free, fast, CPU)
  - Tier 2: Cross-Encoder NLI transformer (nli-deberta-v3-small, CPU)
  - Tier 3: LLM Judge (OpenAI / Anthropic / custom chat model)

Inputs:
  - Snippet: Raw Google Scholar snippets (truncated, 100-250 chars)
  - Abstract: Full abstracts enriched from OpenAlex / arXiv (1,000-2,500 chars)

Datasets:
  - claim_pairs: 120 AI-labeled, unaudited paper pairs ('contradict', 'agree', 'unrelated')
  - legacy: 416 pairs across 10 topics from labels.json

Usage:
  python eval/run_eval.py                              # Tier 1 (Phrases V2) on full abstracts
  python eval/run_eval.py --input snippet              # Tier 1 on raw snippets
  python eval/run_eval.py --backend nli                # Tier 2 (Cross-Encoder NLI)
  python eval/run_eval.py --compare-inputs             # Compare Snippets vs Enriched Abstracts ablation
  python eval/run_eval.py --dataset legacy             # Legacy 416-pair benchmark
  python eval/run_eval.py --show-errors                # Print false positives / false negatives
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "research_agent"))

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from disagreement import (  # noqa: E402
    DEFAULT_NLI_THRESHOLD,
    LLMJudgeVerdict,
    extract_findings,
    make_llm_judge,
    make_nli_judge,
    make_stance_classifier,
    phrase_pair_disagrees,
    stances_oppose,
)
from sample_queries import FIXTURE_DIR  # noqa: E402

# Shared claim each topic's papers take a stance on (used by the stance backend).
TOPIC_CLAIMS = {
    "bilingual_advantage_executive_function_cognitive_control": "Bilingualism gives an advantage in executive function / cognitive control.",
    "ego_depletion_self_control_replication": "Ego depletion (a depleting self-control resource) is a real, replicable effect.",
    "growth_mindset_intervention_academic_achievement": "Growth mindset interventions improve academic achievement.",
    "intermittent_fasting_insulin_sensitivity_metabolic_health": "Intermittent fasting improves insulin sensitivity beyond continuous calorie restriction.",
    "large_language_models_theory_of_mind_reasoning": "LLMs show genuine, robust theory-of-mind reasoning.",
    "mindfulness_meditation_anxiety_randomized_trial": "Mindfulness meditation reduces anxiety effectively.",
    "omega_3_fatty_acids_cardiovascular_disease_prevention": "Omega-3 supplementation prevents cardiovascular disease.",
    "power_posing_hormones_behavior": "Power posing changes feelings of power, hormones or behavior.",
    "screen_time_adolescent_well_being": "Screen time harms adolescent well-being.",
    "vitamin_d_supplementation_depression": "Vitamin D supplementation reduces depressive symptoms.",
}

EVAL_DIR = Path(__file__).resolve().parent


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_legacy_texts(stem: str) -> List[str]:
    for directory in (FIXTURE_DIR, EVAL_DIR / "fixtures"):
        path = directory / f"{stem}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return [
                f"{r.get('title', '')} {r.get('snippet', '')}".strip()
                for r in payload["response"]["organic_results"]
            ]
    raise FileNotFoundError(f"No fixture for {stem!r}")


def evaluate_dataset(
    pairs: List[Dict[str, Any]],
    predict_fn: Callable[[str, str], bool],
    input_field: str = "abstract",
    show_errors: bool = False,
) -> Dict[str, Any]:
    """Evaluate binary contradiction detection on a list of labeled pairs."""
    tp = fp = fn = tn = 0
    errors = []
    per_topic: Dict[str, List[int]] = {}

    for item in pairs:
        topic = item["topic"]
        if topic not in per_topic:
            per_topic[topic] = [0, 0, 0, 0]  # TP, FP, FN, TN

        p_a = item["paper_a"]
        p_b = item["paper_b"]

        if input_field == "abstract":
            text_a = f"{p_a.get('title', '')} {p_a.get('abstract') or p_a.get('snippet', '')}".strip()
            text_b = f"{p_b.get('title', '')} {p_b.get('abstract') or p_b.get('snippet', '')}".strip()
        elif input_field == "findings":
            text_a = f"{p_a.get('title', '')} {extract_findings(p_a.get('abstract') or p_a.get('snippet', ''))}".strip()
            text_b = f"{p_b.get('title', '')} {extract_findings(p_b.get('abstract') or p_b.get('snippet', ''))}".strip()
        else:
            text_a = f"{p_a.get('title', '')} {p_a.get('snippet', '')}".strip()
            text_b = f"{p_b.get('title', '')} {p_b.get('snippet', '')}".strip()

        truth = item["label"] == "contradict"
        pred = bool(predict_fn(text_a, text_b, topic))

        if truth and pred:
            tp += 1
            per_topic[topic][0] += 1
        elif pred and not truth:
            fp += 1
            per_topic[topic][1] += 1
            if show_errors:
                errors.append(("FP", item["id"], text_a[:200], text_b[:200], item.get("rationale", "")))
        elif truth and not pred:
            fn += 1
            per_topic[topic][2] += 1
            if show_errors:
                errors.append(("FN", item["id"], text_a[:200], text_b[:200], item.get("rationale", "")))
        else:
            tn += 1
            per_topic[topic][3] += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    p_lo, p_hi = wilson(tp, tp + fp)
    r_lo, r_hi = wilson(tp, tp + fn)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": len(pairs),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ci_p": (p_lo, p_hi),
        "ci_r": (r_lo, r_hi),
        "per_topic": per_topic,
        "errors": errors,
    }


def evaluate_legacy(
    predict_fn: Callable[[str, str], bool],
    show_errors: bool = False,
) -> Dict[str, Any]:
    labels = json.loads((EVAL_DIR / "labels.json").read_text(encoding="utf-8"))
    tp = fp = fn = tn = skipped = 0
    errors = []
    per_topic = []
    for stem, spec in labels["topics"].items():
        texts = load_legacy_texts(stem)
        pos = {tuple(sorted(p)) for p in spec["disagree"]}
        unclear = {tuple(sorted(p)) for p in spec["unclear"]}
        t_tp = t_fp = t_fn = 0
        for i, j in itertools.combinations(range(len(texts)), 2):
            if (i, j) in unclear:
                skipped += 1
                continue
            truth = (i, j) in pos
            pred = bool(predict_fn(texts[i], texts[j]))
            if truth and pred:
                tp += 1
                t_tp += 1
            elif pred and not truth:
                fp += 1
                t_fp += 1
                if show_errors:
                    errors.append(("FP", stem, i, j, texts[i][:150], texts[j][:150]))
            elif truth and not pred:
                fn += 1
                t_fn += 1
                if show_errors:
                    errors.append(("FN", stem, i, j, texts[i][:150], texts[j][:150]))
            else:
                tn += 1
        per_topic.append((stem, len(pos), t_tp, t_fp, t_fn))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    p_lo, p_hi = wilson(tp, tp + fp)
    r_lo, r_hi = wilson(tp, tp + fn)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "skipped": skipped,
        "total": tp + fp + fn + tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ci_p": (p_lo, p_hi),
        "ci_r": (r_lo, r_hi),
        "per_topic": per_topic,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate literature contradiction detector.")
    parser.add_argument("--backend", choices=["phrases", "nli", "llm", "stance"], default="phrases")
    parser.add_argument("--patterns", choices=["v1", "v2"], default="v2", help="Phrase lexicon.")
    parser.add_argument("--input", choices=["snippet", "abstract", "findings"], default="abstract", help="Text input representation.")
    parser.add_argument("--split", choices=["all", "dev", "test"], default="all", help="Topic-level dev/test split (tune on dev only).")
    parser.add_argument("--dataset", choices=["claim_pairs", "legacy"], default="claim_pairs")
    parser.add_argument("--threshold", type=float, default=DEFAULT_NLI_THRESHOLD)
    parser.add_argument("--show-errors", action="store_true")
    parser.add_argument("--compare-inputs", action="store_true", help="Compare Snippets vs Enriched Abstracts ablation.")
    args = parser.parse_args()

    # Build prediction function: predict_fn(text_a, text_b, topic=None) -> bool
    try:
        if args.backend == "nli":
            judge = make_nli_judge()
            predict_fn = lambda a, b, t=None: judge(a, b) >= args.threshold  # noqa: E731
        elif args.backend == "llm":
            llm_judge = make_llm_judge()

            def predict_fn(a: str, b: str, t=None) -> bool:  # noqa: E306
                res = llm_judge(a, b)
                if isinstance(res, LLMJudgeVerdict):
                    return res.relationship == "CONTRADICT" or res.score >= args.threshold
                return float(res) >= args.threshold
        elif args.backend == "stance":
            classify = make_stance_classifier()
            _cache: dict = {}

            def _stance(topic: str, text: str) -> str:
                key = (topic, text)
                if key not in _cache:
                    _cache[key] = classify(TOPIC_CLAIMS[topic], text)
                return _cache[key]

            def predict_fn(a: str, b: str, t=None) -> bool:  # noqa: E306
                return stances_oppose(_stance(t, a), _stance(t, b))
        else:
            predict_fn = lambda a, b, t=None: phrase_pair_disagrees(a, b, args.patterns)  # noqa: E731
    except (ImportError, Exception) as exc:  # missing deps / API key
        print(f"NOT EVALUATED: backend '{args.backend}' could not start ({type(exc).__name__}: {str(exc)[:200]})")
        return 2

    if args.dataset == "legacy":
        res = evaluate_legacy(predict_fn, show_errors=args.show_errors)
        detail = f" ({args.patterns} lexicon)" if args.backend == "phrases" else f" (threshold {args.threshold})"
        print(f"Backend            : {args.backend}{detail}")
        print(f"Dataset            : Legacy fixture pairs (416 scored, {res['skipped']} unclear skipped)")
        print(f"TP / FP / FN / TN  : {res['tp']} / {res['fp']} / {res['fn']} / {res['tn']}")
        print(f"Precision          : {res['precision']:.2f}  (95% CI {res['ci_p'][0]:.2f}-{res['ci_p'][1]:.2f})")
        print(f"Recall             : {res['recall']:.2f}  (95% CI {res['ci_r'][0]:.2f}-{res['ci_r'][1]:.2f})")
        print(f"F1                 : {res['f1']:.2f}")
        return 0

    # Load 120 claim pairs dataset
    data_path = EVAL_DIR / "claim_pairs_labeled.json"
    if not data_path.exists():
        print(f"Dataset not found at {data_path}; run `python eval/generate_labeled_pairs.py` first.")
        return 1

    payload = json.loads(data_path.read_text(encoding="utf-8"))
    pairs = payload["pairs"]
    dev_topics = set(payload.get("dev_topics", []))
    if args.split == "dev":
        pairs = [p for p in pairs if p["topic"] in dev_topics]
    elif args.split == "test":
        pairs = [p for p in pairs if p["topic"] not in dev_topics]
    n_pos = sum(p["label"] == "contradict" for p in pairs)
    print(f"[split={args.split}] {len(pairs)} pairs, {n_pos} contradict. Labels: {payload.get('provenance', '')[:60]}...")

    if args.compare_inputs:
        print("=" * 70)
        print("ABLATION STUDY: Raw Google Scholar Snippets vs. Enriched Abstracts")
        print(f"Backend: {args.backend} | Dataset: {len(pairs)} AI-labeled (unaudited) pairs across 10 topics")
        print("=" * 70)
        res_snip = evaluate_dataset(pairs, predict_fn, input_field="snippet", show_errors=False)
        res_abs = evaluate_dataset(pairs, predict_fn, input_field="abstract", show_errors=args.show_errors)

        print("\n| Metric | Snippets (100-250 chars) | Enriched Abstracts (1k-2.5k chars) | \u0394 Delta |")
        print("|:---|:---:|:---:|:---:|")
        print(f"| **True Positives** | {res_snip['tp']} | {res_abs['tp']} | {res_abs['tp'] - res_snip['tp']:+d} |")
        print(f"| **False Positives** | {res_snip['fp']} | {res_abs['fp']} | {res_abs['fp'] - res_snip['fp']:+d} |")
        print(f"| **False Negatives** | {res_snip['fn']} | {res_abs['fn']} | {res_abs['fn'] - res_snip['fn']:+d} |")
        print(f"| **Precision (95% CI)** | {res_snip['precision']:.2f} [{res_snip['ci_p'][0]:.2f}, {res_snip['ci_p'][1]:.2f}] | {res_abs['precision']:.2f} [{res_abs['ci_p'][0]:.2f}, {res_abs['ci_p'][1]:.2f}] | {res_abs['precision'] - res_snip['precision']:+.2f} |")
        print(f"| **Recall (95% CI)** | {res_snip['recall']:.2f} [{res_snip['ci_r'][0]:.2f}, {res_snip['ci_r'][1]:.2f}] | {res_abs['recall']:.2f} [{res_abs['ci_r'][0]:.2f}, {res_abs['ci_r'][1]:.2f}] | {res_abs['recall'] - res_snip['recall']:+.2f} |")
        print(f"| **F1 Score** | **{res_snip['f1']:.2f}** | **{res_abs['f1']:.2f}** | **{res_abs['f1'] - res_snip['f1']:+.2f}** |")

        if args.show_errors:
            print("\nAbstract Errors:")
            for err in res_abs["errors"][:5]:
                print(f"  [{err[0]}] {err[1]}: {err[4]}")
        return 0

    res = evaluate_dataset(pairs, predict_fn, input_field=args.input, show_errors=args.show_errors)
    detail = f" ({args.patterns} lexicon)" if args.backend == "phrases" else f" (threshold {args.threshold})"
    print(f"Backend            : {args.backend}{detail}")
    print(f"Input Mode         : {args.input} (Scholar snippet vs OpenAlex full abstract)")
    print(f"Dataset            : {len(pairs)} AI-labeled (unaudited) pairs across 10 topics ({payload['counts']})")
    print(f"TP / FP / FN / TN  : {res['tp']} / {res['fp']} / {res['fn']} / {res['tn']}")
    print(f"Precision          : {res['precision']:.2f}  (95% CI {res['ci_p'][0]:.2f}-{res['ci_p'][1]:.2f})")
    print(f"Recall             : {res['recall']:.2f}  (95% CI {res['ci_r'][0]:.2f}-{res['ci_r'][1]:.2f})")
    print(f"F1 Score           : {res['f1']:.2f}")

    if args.show_errors and res["errors"]:
        print(f"\nSample Errors ({len(res['errors'])} total):")
        for kind, pair_id, ta, tb, rat in res["errors"][:6]:
            print(f"[{kind}] {pair_id}\n  Rationale: {rat}\n  Text A: {ta[:120]}...\n  Text B: {tb[:120]}...\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
