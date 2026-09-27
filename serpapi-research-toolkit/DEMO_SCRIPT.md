# 🔬 Research Agent — Live Demo Script & Walkthrough

This guide provides a step-by-step walkthrough for demoing the **serpapi-research-toolkit** during hackathon judging or presentation.

---

## 🚀 Pre-Flight Setup (1 Minute)

1. Ensure your `.env` contains your `SERPAPI_API_KEY`:
   ```bash
   SERPAPI_API_KEY=your_key_here
   ```
2. Launch the Streamlit application:
   ```bash
   cd examples/research_agent
   streamlit run app.py
   ```
3. The app will open at `http://localhost:8501`.

---

## 🎯 Demo Flow (3–4 Minutes)

### Step 1: The Core Problem (30s)
> *"When researchers or developers use search APIs in AI agents, they typically get raw JSON blobs with inconsistent schemas depending on whether they query Google Search, News, Jobs, or Scholar. Furthermore, during development, iterative agent loops burn expensive search credits rapidly on repeated calls.*
>
> *We built `langchain-serpapi-provider`: a lean, LangChain-native plugin that normalizes all four SerpApi engines into a unified Pydantic schema with built-in disk caching and heuristic engine routing."*

### Step 2: Live Query & Clustering (90s)
Click the first sample query button:
> **💡 "Do large language models possess theory of mind?"**
*(or "Does intermittent fasting improve insulin sensitivity in adults?")*

Click **🔬 Run Research Agent**.

**What to point out as it runs:**
1. **Live Retrieval:** Calls `SerpApiRetriever(engine="google_scholar")` to pull top academic papers with parsed citation counts and author metadata.
2. **Topic Clustering:** Rather than a flat list of 10 links, the agent embeds each paper's title + abstract and runs **KMeans** clustering.
3. **Cluster Labeling:** Point out the topic headers (e.g. `Cluster 1: Attention & Sequence Modeling`), dynamically generated via TF-IDF n-grams to identify what sub-domain each group represents.

### Step 3: The Divergence Detection (60s)
Point out the cluster with the **amber divergence badge**:
> ⚠️ **Possible divergent findings — needs human review.**

**Talking points:**
- *"Notice our framing here: we deliberately do not claim 'detected contradictions'. Scientific nuance cannot be reduced to a binary contradiction detector without hallucinations.*
- *Instead, we do a conservative heuristic pass looking for clusters where peer-reviewed abstracts contain both affirmative conclusion signals (e.g. 'outperforms', 'demonstrates capability') and cautionary/limitation signals (e.g. 'fails under perturbation', 'no evidence of genuine reasoning').*
- *We surface the exact keyword cues directly on the paper cards (green pill for affirmative cues, purple pill for cautionary cues) so the human researcher can immediately click through to the source papers and evaluate the evidence for themselves."*

### Step 4: The Developer Experience & Caching (30s)
Re-click **🔬 Run Research Agent** on the same query.
- Notice how it completes **instantly** (0.2 seconds).
- Point to the terminal logs showing `Cache hit for query`.
- *"The built-in `SerpApiCache` uses SHA-256 query+engine hashing with configurable TTL, guaranteeing that developers testing agent workflows don't burn their 250 free monthly credits on repeat runs."*

---

## 🛡️ Anticipated Q&A / Edge Cases

### Q: *"Why does a paper from 2020 show 2026 as the year?"*
> **Answer:** *"Google Scholar aggregates preprints and canonical versions under one cluster. If an author circulated an arXiv preprint in 2020 and it was republished in a 2026 textbook edition (e.g. Springer), Google Scholar indexes the canonical edition date (2026) while aggregating all citations since 2020. That is why our UI explicitly labels it **'Indexed: 2026'** to remain completely transparent about what the underlying metadata represents."*

### Q: *"How does the plugin avoid dependency bloat?"*
> **Answer:** *"The plugin core only depends on `langchain-core` (not the heavy full `langchain` package). It provides `BaseRetriever` and `Tool` interfaces with zero agent bloat. Full LangChain agent executors are kept as an optional `[agent]` extra."*

### Q: *"Does the engine selector use an LLM or ML classifier?"*
> **Answer:** *"No, by design. `select_engine` uses deterministic, word-boundary and compound phrase matching (e.g. 'literature review' → Scholar, 'press release' → News, 'remote work' → Jobs). It runs in microseconds, consumes 0 tokens, and is completely predictable."*
