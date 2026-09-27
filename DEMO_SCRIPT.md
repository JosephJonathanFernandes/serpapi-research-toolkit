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

## 🎯 Demo Flow (Timed Stopwatch: 2:15 Total — 45s Buffer Under 3:00 Cap)

### Step 1: The Core Problem (25s)
> *"When building AI agents that search the web, developers face two immediate headaches:
> 1. Inconsistent data shapes: Querying Google Scholar, Web, News, or Jobs returns completely different JSON schemas.
> 2. Expensive credit burn: Iterative agent development loops re-query the same prompts and rapidly exhaust API quotas.
>
> We built `langchain-serpapi-provider`: a lean, pip-installable LangChain plugin with normalized Pydantic schemas, automatic heuristic engine routing, and a built-in disk cache."*

### Step 2: Live Query & Clustering (50s)
Click the first sample query button:
> **💡 Bilingual Advantage**  
*(Query: `bilingual advantage executive function cognitive control`)*  

Click **🔬 Run Research Agent**.

**What to point out during the render (Measured UI latency: ~10 seconds total):**
1. **Live Retrieval:** Calls `SerpApiRetriever(engine="google_scholar")` to pull top peer-reviewed papers with citation counts and author metadata.
2. **Topic Clustering:** Rather than dumping a flat list of 10 links, the agent embeds each title + abstract and runs **KMeans** clustering.
3. **Dynamic Cluster Labeling:** Point out the topic headers (e.g. `Cluster 1: Cognitive Control & Control`, `Cluster 2: Executive & Bilingual`), generated via TF-IDF n-grams to immediately explain what sub-domain each group represents.

### Step 3: The Mixed Findings Signal (45s)
Scroll down to **Cluster 2 ("Executive & Bilingual")** with the **amber warning badge**:
> ⚠️ **Mixed findings signal — human review recommended.**  

**Talking points (Honest, Defensible Framing):**
- *"Notice our framing here: we deliberately do NOT claim to have built an ML 'contradiction detector'. Scientific nuance cannot be reduced to a binary contradiction label without severe hallucination risk.*
- *Instead, we provide a transparent heuristic filter: when a cluster contains abstracts reporting advantages/enhancements alongside papers reporting null findings or replication challenges, the agent flags it as a 'Mixed findings signal'.*
- *Look at the actual papers in Cluster 2: we have Emily Nichols' landmark 2020 population study of 11,000 people titled **'Bilingualism affords no general cognitive advantages'** (flagged with null cue pills: `no general cognitive advantages`, `affords no`), right alongside Arizmendi et al. reporting bilingual children **'outperforming'** monolinguals.*
- *This is a genuine, active debate in cognitive psychology. The agent immediately surfaces this tension so a researcher knows to inspect the methodologies rather than falsely assuming scientific consensus."*

### Step 4: Developer Experience & Sub-Millisecond Caching (20s)
Click **🔬 Run Research Agent** a second time on the same query.
- Notice the retrieval finishes **instantly**: terminal logs show a cache hit with sub-millisecond retrieval (`< 1 ms`).
- *"The built-in `SerpApiCache` uses SHA-256 query+engine hashing with configurable TTL, guaranteeing that developers testing agent workflows don't burn their 250 free monthly credits on repeat runs."*

---

## 🛡️ Anticipated Q&A / Edge Cases

### Q: *"Why does a paper from 2020 show 2026 as the year?"*
> **Answer:** *"Google Scholar aggregates preprints and canonical versions under one cluster. If an author circulated an arXiv preprint in 2020 and it was republished in a 2026 textbook edition (e.g. Springer), Google Scholar indexes the canonical edition date (2026) while aggregating all citations since 2020. That is why our UI explicitly labels it **'Indexed: 2026'** to remain completely transparent about what the underlying metadata represents."*

### Q: *"How does the plugin avoid dependency bloat?"*
> **Answer:** *"The plugin core only depends on `langchain-core` (not the heavy full `langchain` package). It provides `BaseRetriever` and `Tool` interfaces with zero agent bloat. Full LangChain agent executors are kept as an optional `[agent]` extra."*

### Q: *"Does the engine selector use an LLM or ML classifier?"*
> **Answer:** *"No, by design. `select_engine` uses deterministic, word-boundary and compound phrase matching (e.g. 'literature review' → Scholar, 'press release' → News, 'remote work' → Jobs). It runs in microseconds, consumes 0 tokens, and is completely predictable."*
