# 🧪 Literature Contradiction Detection Evaluation

This directory contains the reproducible benchmark and empirical evaluation for the research agent's **tiered disagreement detector** and **abstract enrichment pipeline**.

---

## 🎯 Motivation

Most literature exploration tools either:
1. Present search results as a flat list with no synthesis, or
2. Make unverified claims about detecting "mixed findings" or "scientific consensus" using 150-character Google Scholar snippets.

Short snippets are inherently flawed for contradiction detection:
- **Mid-sentence truncation:** Snippets frequently cut off before the result clause.
- **Introductory background noise:** Snippets often quote previous literature reviews or problem definitions rather than the study's own findings.
- **Population & metric divergence:** Differences in phrasing often reflect different cohorts (e.g. pediatric vs elderly) rather than an actual scientific conflict.

To solve this, we:
1. **Fixed the input:** Enriched truncated snippets with full academic abstracts from **OpenAlex** (250M+ works) and **arXiv** (free, no API key).
2. **Built a tiered detector:**
   - **Tier 1:** Directional phrase matching (fast, deterministic, zero compute).
   - **Tier 2:** Transformer Cross-Encoder NLI (`cross-encoder/nli-deberta-v3-small` running locally on CPU).
   - **Tier 3:** LLM Judge (OpenAI / LangChain `BaseChatModel`).
3. **Audited the results:** Hand-labeled 120 claim pairs across 10 contentious literature domains to measure precision, recall, F1, and empirical failure modes.

---

## 📊 Dataset

The benchmark dataset ([`claim_pairs_labeled.json`](file:///eval/claim_pairs_labeled.json)) consists of **120 audited paper pairs** across 10 contentious scientific domains:
- **Bilingual Advantage** in cognitive control & executive function
- **Growth Mindset** interventions & academic GPA
- **Ego Depletion** replication & self-control strength model
- **Power Posing** hormonal & behavioral effects
- **Intermittent Fasting** vs continuous restriction on insulin sensitivity
- **Screen Time** & adolescent psychological well-being
- **Vitamin D** supplementation & depressive symptoms
- **Omega-3 Fatty Acids** & cardiovascular disease prevention
- **Mindfulness Meditation** vs pharmacotherapy (Escitalopram)
- **Large Language Models** & Theory of Mind reasoning benchmarks

### Class Distribution (120 pairs):
- **Contradict (`42` pairs):** Papers reaching directly opposing conclusions or failing to replicate primary effects.
- **Agree (`43` pairs):** Papers reporting consistent findings, successful replications, or supporting meta-analyses.
- **Unrelated (`35` pairs):** Papers examining orthogonal mechanisms, different populations, or non-comparable questions within the same topic.

Each pair contains:
- Paper title, original Google Scholar snippet, and full OpenAlex/arXiv abstract
- Ground-truth label (`contradict`, `agree`, `unrelated`)
- Human annotator domain rationale

---

## 🔬 Reproducible Benchmark Commands

All benchmarks run 100% offline from the cached fixture and abstract datasets:

```bash
# Tier 1 (Phrases V2) on full abstracts
python eval/run_eval.py

# Tier 1 on raw Google Scholar snippets
python eval/run_eval.py --input snippet

# Compare Snippets vs Enriched Abstracts ablation
python eval/run_eval.py --compare-inputs

# Tier 2 (Cross-Encoder NLI on CPU)
python eval/run_eval.py --backend nli

# Inspect False Positives and False Negatives
python eval/run_eval.py --show-errors
```

---

## 📈 Benchmark Results

### 1. The Input Ablation: Raw Snippets vs. Enriched Abstracts (Tier 1 Phrases)

| Input Representation | True Positives | False Positives | False Negatives | Precision (95% CI) | Recall (95% CI) | F1 Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Google Scholar Snippets** (100–250 chars) | 9 | 2 | 33 | **0.82** [0.52, 0.95] | **0.21** [0.12, 0.36] | **0.34** |
| **OpenAlex Enriched Abstracts** (1k–2.5k chars) | 6 | 10 | 36 | **0.38** [0.18, 0.61] | **0.14** [0.07, 0.28] | **0.21** |

### 2. Tier 2 Cross-Encoder NLI (`nli-deberta-v3-small`) on CPU

| Input Representation | True Positives | False Positives | False Negatives | Precision (95% CI) | Recall (95% CI) | F1 Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Snippets** | 8 | 8 | 34 | **0.50** [0.28, 0.72] | **0.19** [0.10, 0.33] | **0.28** |
| **Enriched Abstracts (Head Truncation)** | 1 | 12 | 41 | **0.08** [0.01, 0.33] | **0.02** [0.00, 0.12] | **0.04** |

*(Confidence intervals computed using the Wilson score interval at 95% confidence).*

---

## 💡 Key Empirical Takeaways & Error Analysis

1. **Why Naive Abstract Truncation Fails NLI:**
   Cross-encoders typically have a 512-token context window. Truncating long abstracts from the beginning (`text[:512]`) captures only the **introductory background** and literature review where authors introduce the common problem. The actual empirical findings, sample statistics, and conclusions reside in the **final 2–3 sentences** of the abstract. Feeding the introductory head causes the model to classify almost all pairs as `neutral` or `entailment`.

2. **Why General Sentence-NLI Struggles with Scientific Literature:**
   NLI cross-encoders are trained on MNLI/SNLI (strict sentence-level logical contradiction, e.g. *"A man is eating"* vs *"A man is sleeping"*). In scientific papers:
   - Paper A: *"In a sample of 4,524 children, we found no evidence of an executive function advantage."*
   - Paper B: *"In 64 adults under high task demands, bilinguals had significantly smaller switching costs."*
   Because both statements can be simultaneously true under strict formal logic (different samples, different age groups), standard NLI models assign `neutral`. However, in scientific meta-analysis, these represent competing stances on the generalizability of the bilingual advantage hypothesis.

3. **Why Phrase Matching is a Useful Free Baseline:**
   On raw snippets, Tier 1 Phrase Matching achieved **0.82 Precision** with zero network latency and zero GPU/CPU footprint. While its recall is limited (0.21), its high precision on snippets makes it an effective initial heuristic to surface candidate conflicts for human inspection.

4. **Product Decision:**
   Because no automated heuristic achieves 100% precision on scientific literature, the UI conservatively brands every detection as:
   > **⚠️ Possible disagreement — human review recommended.**  
   > *Snippets contain contrasting outcome wording. Always check the primary papers.*
