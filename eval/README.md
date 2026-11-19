# Stance & Disagreement Detector Evaluation

Evaluation of the research agent's stance and disagreement detectors across:
1. **External human-labeled benchmark (SciFact dev, 340 claim–abstract pairs)** — [`eval/scifact/results.md`](scifact/results.md).
2. **Internal pairwise ablation (120 AI-drafted, unaudited paper pairs across 10 topics)** — [`eval/results.md`](results.md).

---

## 1. External Validation on SciFact (Human Labels)

Because our internal 120-pair dataset was drafted by an AI assistant, we validate stance classification externally on the **SciFact** dev split (Wadden et al., EMNLP 2020; 300 expert-written claims, 340 claim–abstract pairs labeled `SUPPORT`, `CONTRADICT`, or `NEI` by human annotators). None of the SciFact topics were seen when writing any phrase lexicon.

Mapping from our classifier outputs to SciFact labels:
`SUPPORTS -> SUPPORT`, `NULL -> CONTRADICT`, `MIXED / UNCLEAR -> NEI`.

| Backend | n | Accuracy | Macro-F1 | CONTRADICT P [95% CI] | CONTRADICT R [95% CI] | CONTRADICT F1 | SUPPORT F1 |
|:---|---:|---:|---:|:---:|:---:|---:|---:|
| `majority` — always SUPPORT | 340 | 0.41 | 0.19 | 0.00 n/a (of 0) | 0.00 [0.00, 0.05] | **0.00** | 0.58 |
| `phrases-v1` — untuned phrase lexicon | 340 | 0.38 | 0.19 | 0.20 [0.04, 0.62] (of 5) | 0.01 [0.00, 0.08] | **0.03** | 0.00 |
| `phrases-v2` — expanded phrase lexicon | 340 | 0.41 | 0.30 | 0.23 [0.08, 0.50] (of 13) | 0.04 [0.01, 0.12] | **0.07** | 0.30 |
| `phrases-v3` — dev-selected phrase lexicon | 340 | 0.42 | 0.30 | 0.20 [0.04, 0.62] (of 5) | 0.01 [0.00, 0.08] | **0.03** | 0.33 |
| `nli` — `nli-deberta-v3-small` (claim as hypothesis) | 340 | **0.48** | **0.42** | **0.56** [0.43, 0.68] (of 59) | **0.46** [0.35, 0.58] | **0.51** | 0.17 |

Full confusion matrices: [`eval/scifact/results.md`](scifact/results.md). Reproduce with:

```bash
python eval/scifact/run_scifact.py --backend phrases-v1
python eval/scifact/run_scifact.py --backend phrases-v3
python eval/scifact/run_scifact.py --backend nli
python eval/scifact/run_scifact.py --backend llm   # uses OPENAI_API_KEY or local Ollama
```

### What SciFact shows
1. **Testing against an explicit claim works far better for `CONTRADICT` than claim-agnostic phrase matching.** On 340 human-labeled claim–abstract pairs, `cross-encoder/nli-deberta-v3-small` reaches **0.51 CONTRADICT F1** (precision 0.56, recall 0.46, and 0/71 true `CONTRADICT` abstracts misclassified as `SUPPORT`), whereas all three phrase lexicons score **0.03–0.07 CONTRADICT F1** because surface outcome cues cannot tell whether a finding supports or opposes a claim's direction.
2. **NLI is conservative on `SUPPORT` (recall 13/138 = 0.09, precision 13/16 = 0.81).** A multi-sentence abstract findings block often contains extra detail beyond the short claim, which an MNLI cross-encoder labels `neutral` (`UNCLEAR`). That makes NLI stance helpful for per-paper `NULL / CONTRADICT` flagging in a stance-count UI, but poor when multiplied across pairs (`SUPPORTS` × `NULL`).

---

## 2. Internal 120-Pair Benchmark & Limitations

### Limitations (read first)
- **Labels are AI-drafted and unaudited.** The 120 pair labels and rationales in `claim_pairs_labeled.json` were written by an AI coding assistant in one pass without reading all full abstracts. Use the blind human-audit workflow in [`eval/audit/`](audit/make_audit_sheet.py) before treating pairwise numbers as ground truth.
- **Lexicon V2 vs V3.** Phrase lexicon V2 was written after reading snippets from all 10 topics, so its test score (0.40 F1) is contaminated. Phrase lexicon V3 (`eval/build_lexicon_v3.py`, saved to `examples/research_agent/lexicon_v3.json`) was selected mechanically by greedy forward selection on the **dev split only** (caveat: its candidate pool came from V2).
- **Small n.** Test has 84 pairs (30 contradictions), dev has 36 (12 contradictions). One true positive shifts test recall by 0.033.

### Test-split pairwise results (precision / recall / F1)

| Detector | Snippet | Full abstract | Findings only |
|---|:---:|:---:|:---:|
| `phrases-v1` (untuned) | 0.00 / 0.00 / 0.00 | 0.50 / 0.03 / 0.06 | 0.00 / 0.00 / 0.00 |
| `phrases-v2` (tuned on same topics — upper bound) | 0.80 / 0.27 / **0.40** | 0.33 / 0.13 / 0.19 | 0.17 / 0.03 / 0.06 |
| `phrases-v3` (selected on dev only) | 0.75 / 0.10 / **0.18** | 0.44 / 0.13 / **0.21** | 0.20 / 0.03 / 0.06 |
| `nli-pair` (DeBERTa paper-vs-paper) | 0.58 / 0.23 / **0.33** | 0.00 / 0.00 / 0.00 | 0.44 / 0.13 / 0.21 |
| `nli-stance` (DeBERTa opposing stances) | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |
| `llm-stance` (`--llm`) | skipped on CPU | skipped on CPU | skipped on CPU |

Full dev and test counts: [`eval/results.md`](results.md).

### Reproduce
```bash
python eval/build_lexicon_v3.py         # greedy dev-only selection -> lexicon_v3.json
python eval/run_ablation.py             # Tiers 1-2 sweep -> eval/results.md, eval/results.json
python eval/run_ablation.py --llm       # include Tier 3 LLM stance (OpenAI key or local Ollama)
python eval/audit/make_audit_sheet.py   # generate blind 40-pair + 20-pair human audit CSVs
python eval/audit/score_audit.py        # compute label-change rate & Cohen's kappa
```
