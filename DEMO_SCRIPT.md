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

## 🎯 Demo Flow (Estimated about 2:30; time yourself with a stopwatch before recording. Cap is 3:00)

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

**🎙️ Spoken Narration During the ~12s Cold Spinner (Fill the gap — zero dead air):**
> *"Right now, three pipeline stages are executing live:
> 1. First, our `SerpApiRetriever` is hitting Google Scholar via SerpApi, pulling 10 papers complete with author metadata, citation counts, and canonical URLs.
> 2. Next, rather than dumping a raw flat list, the agent embeds each title and abstract into vector space and runs unsupervised KMeans clustering to group related studies.
> 3. Finally, it uses TF-IDF n-grams to dynamically generate human-readable topic headers for each cluster — and here are the results."*

*(At ~11–12 seconds, the cluster cards appear on screen. Transition directly to Step 3 without pause).*

### Step 3: The Mixed Findings Signal (45s)
Scroll down to **Cluster 2 ("Executive & Bilingual")** with the **amber warning badge**:
> ⚠️ **Mixed findings signal — human review recommended.**  

**Talking points (Honest, Defensible Framing):**
- *"Notice our framing here: we deliberately do NOT claim to have built an ML 'contradiction detector'. Scientific nuance cannot be reduced to a binary contradiction label without severe hallucination risk.*
- *Instead, we provide a transparent heuristic filter: when a cluster contains abstracts reporting advantages/enhancements alongside papers reporting null findings or replication challenges, the agent flags it as a 'Mixed findings signal'.*
- *Look at Cluster 2. One paper is titled **'Bilingualism affords no general cognitive advantages'**, a 2020 study of 11,000 people, and it gets null-cue pills (`no general cognitive advantages`, `affords no`). Another paper in the same cluster has a snippet containing 'outperforming', which comes from a sentence defining the bilingual advantage, not from that paper's result. That is why the badge says human review: the tool matches phrases, it does not read conclusions.*
- *Whether bilingualism gives an executive-function advantage is widely discussed as contested. We are not adjudicating that. The tool just points the researcher to papers worth reading closely."*

### Step 4: Developer Experience & Millisecond-Level Caching (20s)
Click **🔬 Run Research Agent** a second time on the same query.
- Notice the dramatic visible difference: the first live run took **~12s** while SerpApi queried Google Scholar. The second run completes in **2.8s** with about 1 ms retrieval (`< 1.3 ms`).
- Point out: *"The built-in `SerpApiCache` uses SHA-256 query+engine hashing with configurable TTL, guaranteeing that developers testing agent workflows don't burn their 250 free monthly credits on repeat runs (in our test, retrieval went from about 7s live to about 1ms cached)."*

---

## 🛡️ Anticipated Q&A / Edge Cases

### Q: *"Why does a paper from 2020 show 2026 as the year?"*
> **Answer:** *"Google Scholar aggregates preprints and canonical versions under one cluster. If an author circulated an arXiv preprint in 2020 and it was republished in a 2026 textbook edition (e.g. Springer), Google Scholar indexes the canonical edition date (2026) while aggregating all citations since 2020. That is why our UI explicitly labels it **'Indexed: 2026'** to remain completely transparent about what the underlying metadata represents."*

### Q: *"How does the plugin avoid dependency bloat?"*
> **Answer:** *"The plugin core only depends on `langchain-core` (not the heavy full `langchain` package). It provides `BaseRetriever` and `Tool` interfaces with zero agent bloat. Full LangChain agent executors are kept as an optional `[agent]` extra."*

### Q: *"Does the engine selector use an LLM or ML classifier?"*
> **Answer:** *"No, by design. `select_engine` uses deterministic, word-boundary and compound phrase matching (e.g. 'literature review' → Scholar, 'press release' → News, 'remote work' → Jobs). It runs in microseconds, consumes 0 tokens, and is completely predictable."*
