# Disagreement Detector Evaluation

Provisional evaluation of the research agent's "possible disagreement" detector. Read the limitations first.

## Limitations (read first)

- **Labels are AI-drafted and unaudited.** The 120 pair labels and rationales were written by an AI coding assistant in one pass from titles, snippets and (for only some papers) abstracts, plus general knowledge. Rationales can mention details not in the stored abstracts. There is one labeler and no inter-annotator agreement (no Cohen's kappa). A human should re-label at least a subset before any number here is quoted.
- **Lexicon contamination.** Phrase lexicon V2 was written after reading snippets from these same topics. The dev/test split cannot remove that; V2 numbers are optimistic. V1 is the only untuned lexicon.
- **Small n.** Test has 84 pairs with 30 contradictions, dev has 36 with 12. Most differences between rows are within noise (one true positive moves recall by about 0.03).
- **Tier 3 / LLM stance is not evaluated.** No API key was available. The code and prompt exist and are unit-tested with a fake LLM only.

## Data and split

`claim_pairs_labeled.json`: 120 pairs, 10 topics, 42 contradict / 43 agree / 35 unrelated. Papers come from Google Scholar results enriched with OpenAlex/arXiv abstracts (`enriched_papers.json`, 85 of 100 papers found; the rest fall back to snippets).

Split rule: topics sorted alphabetically, dev = every 4th topic, test = the rest (dev 36 pairs, test 84). Tune on dev only; report test once. Thresholds are fixed (NLI 0.5).

## Reproduce

```bash
python eval/generate_labeled_pairs.py   # rebuild dataset (adds provenance + split)
python eval/run_ablation.py             # full sweep -> eval/results.md, eval/results.json
python eval/run_eval.py --split test --input findings --backend nli   # single row
```

## Test-split results (precision / recall / F1)

| Detector | Snippet | Full abstract | Findings only |
|---|:---:|:---:|:---:|
| Phrases V1 | 0.00 / 0.00 / 0.00 | 0.50 / 0.03 / 0.06 | 0.00 / 0.00 / 0.00 |
| Phrases V2 (contaminated) | 0.80 / 0.27 / 0.40 | 0.33 / 0.13 / 0.19 | 0.17 / 0.03 / 0.06 |
| NLI DeBERTa-small | 0.58 / 0.23 / 0.33 | 0.00 / 0.00 / 0.00 | 0.44 / 0.13 / 0.21 |
| LLM stance | not evaluated | not evaluated | not evaluated |

Counts and the dev split are in `results.md`. "Findings only" = structured Results/Conclusion section if present, else the last three sentences (`extract_findings`).

## What the numbers say

1. Full abstracts have a mixed effect: test F1 fell for phrases V2 (0.40 to 0.19) and NLI (0.33 to 0.00) but rose for phrases V1 (0.00 to 0.06, one true positive); on dev, V2 rose (0.15 to 0.25). The effect is within noise.
2. Findings-only NLI (0.21) vs full-abstract NLI (0.00) rests on 4 true positives of 30. Not evidence of an effect. It does not help phrase matching.
3. Phrases V2 on snippets (F1 0.40) is the best row, but its lexicon was **tuned on the same topics**, so it is an upper bound.
4. Best recall is 0.27, so the detector misses roughly three of four labeled contradictions.
5. The labels are unverified and were not checked against the abstracts the detectors read, so a miss may be a label error. These numbers do not support a claim about detector quality until the human audit (`eval/audit/`) is done.
6. NLI is the wrong task: papers on different cohorts are "neutral" under an MNLI model. The stance classifier targets that but is not yet measured.

## Next steps

1. Human re-label a subset (ideally two labelers, report Cohen's kappa).
2. Run the LLM stance classifier on dev, then test once.
3. Error analysis of false positives/negatives per row before explaining causes.
