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

1. Full abstracts did not help any detector on test; they hurt all of them. Our earlier expectation that better input would improve detection is not supported here.
2. Findings-only extraction recovers some NLI performance but not phrase matching, and no input beats raw snippets. The reason is untested; plausible causes are cue co-occurrence in long text and crude sentence extraction.
3. Best recall is 0.27, so the detector misses roughly three of four labeled contradictions. It is a triage hint, not evidence of disagreement.
4. NLI is the wrong task for this problem: two papers on different cohorts are "neutral" under an MNLI model even when they take opposing stances. The stance classifier (SUPPORTS vs NULL toward a shared claim) targets that, but is not yet measured.

## Next steps

1. Human re-label a subset (ideally two labelers, report Cohen's kappa).
2. Run the LLM stance classifier on dev, then test once.
3. Error analysis of false positives/negatives per row before explaining causes.
