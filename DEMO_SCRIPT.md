# 🔬 Research Agent — Live Demo Script & Walkthrough

This guide provides a step-by-step walkthrough for demoing the **serpapi-research-toolkit** during hackathon judging or presentation.

---

## 🚀 Pre-Flight Setup (1 Minute)

### Option A: Offline Demo Mode (Zero API Key Needed!)
1. Launch the Streamlit application:
   ```bash
   cd examples/research_agent
   streamlit run app.py
   ```
2. The app opens at `http://localhost:8501`.
3. In the sidebar, leave **Demo mode (recorded results, no API key)** checked. The full pipeline runs instantly using cached response fixtures without consuming any SerpApi credits.

### Option B: Live Mode with SerpApi
1. Ensure your `.env` contains your `SERPAPI_API_KEY`:
   ```bash
   SERPAPI_API_KEY=your_key_here
   ```
2. Launch the application:
   ```bash
   cd examples/research_agent
   streamlit run app.py
   ```
3. Untick **Demo mode** in the sidebar.

---

## 🎯 Demo Flow (Estimated about 2:30; time yourself with a stopwatch before recording. Cap is 3:00)

### Step 1: The Core Problem & Plugin Justification (30s)
> *"When building AI agents that search the web or conduct RAG over academic literature, developers face three immediate roadblocks:
> 1. Inconsistent data shapes: Querying Google Scholar, Web, News, or Jobs returns completely different JSON schemas.
> 2. Expensive credit burn: Iterative agent loops re-query the same prompts and rapidly exhaust API quotas.
> 3. Standard LangChain wrappers (like `langchain-community`) only provide basic string execution, not a first-class `BaseRetriever` with unified schemas or built-in caching.
>
> We built `langchain-serpapi-provider`: a lean, pip-installable LangChain plugin with normalized Pydantic schemas across all four engines, multi-parameter disk caching, and automated error retries."*

### Step 2: Query, Silhouette Clustering & Labeling (45s)
Click the first sample query button:
> **💡 Bilingual Advantage**  
*(Query: `bilingual advantage executive function cognitive control`)*  

Click **🔬 Run Research Agent**.

**🎙️ Spoken Narration During the Cold Spinner (Fill the gap — zero dead air):**
> *"Right now, three pipeline stages are executing:
> 1. First, our `SerpApiRetriever` retrieves 10 papers complete with author metadata, citation counts, and canonical URLs.
> 2. Next, rather than dumping a flat list, the agent embeds each title and snippet into vector space and automatically selects the optimal cluster count using silhouette scores.
> 3. Finally, it uses TF-IDF n-grams to dynamically generate topic headers for each cluster — showing the top 3 most-cited papers alongside so the researcher can verify cluster coherence."*

*(Cluster cards appear on screen. Transition directly to Step 3 without pause).*

### Step 3: Possible Disagreement Detection (50s)
Scroll down to the cluster with the **amber warning badge**:
> ⚠️ **Possible disagreement — human review recommended.**  

**Talking points (Honest, Defensible Framing):**
- *"Notice our careful framing here: we deliberately do NOT claim to have built an infallible ML 'contradiction detector'. Search snippets are truncated and cut mid-sentence, which makes automated conflict adjudication high-risk.*
- *Instead, we provide a transparent heuristic filter: when a cluster contains abstracts reporting advantages alongside papers reporting null findings or replication challenges, the agent flags it as 'Possible disagreement — human review recommended'.*
- *Look at this cluster. One paper is titled **'Bilingualism affords no general cognitive advantages'**, a study of 11,000 people, tagged with cautionary cue chips. Another paper has an affirmative cue chip. We even benchmarked this against a cross-encoder NLI model in our `eval/` suite: our transparent phrase heuristic achieved 53% recall and 32% precision, while transformer NLI struggled with snippet truncation at 12% precision.*
- *The tool does not claim to settle scientific disputes. It points researchers directly to papers worth reading closely."*

### Step 4: Export, Sorting & Developer Caching (25s)
1. Point to the **Export CSV** and **Export BibTeX** buttons:
   - *"Researchers can immediately export their clustered findings as CSV or BibTeX with sanitized keys and formula-injection defenses."*
2. Demonstrate Sorting:
   - In the sidebar, select **Citations (high → low)** — the clustered cards re-sort instantly without re-running the pipeline.
3. Click **🔬 Run Research Agent** a second time:
   - The query returns in milliseconds from the local disk cache (`~0.5 ms` retrieval vs seconds live), saving API credits.

---

## 🛡️ Anticipated Q&A / Edge Cases

### Q: *"How does this compare to `langchain-community`'s SerpAPI wrapper?"*
> **Answer:** *"The community wrapper is a basic utility that returns raw text or raw dicts. Our plugin is a full `BaseRetriever` drop-in for modern LCEL RAG chains (`retriever | prompt | llm`), provides normalized Pydantic schemas across Scholar, Web, News, and Jobs, includes multi-parameter disk caching (~1,400x speedup), and has native async and error retry support."*

### Q: *"Why does a paper from 2020 show 2026 as the year?"*
> **Answer:** *"Google Scholar aggregates preprints and canonical versions under one cluster. If an author circulated an arXiv preprint in 2020 and it was republished in a 2026 textbook edition (e.g. Springer), Google Scholar indexes the canonical edition date (2026) while aggregating all citations since 2020. That is why our UI explicitly labels it **'Indexed: 2026'** to remain completely transparent about what the underlying metadata represents."*

### Q: *"Can I plug in an LLM or custom classifier for engine selection or contradiction?"*
> **Answer:** *"Yes. `build_auto_engine_tool(engine_selector=...)` accepts any custom callable, including an LLM prompt. For disagreement, the app also supports an experimental NLI cross-encoder backend, fully documented with precision/recall benchmarks in `eval/`."*

### Q: *"How does the plugin avoid dependency bloat?"*
> **Answer:** *"The plugin core only depends on `langchain-core>=0.2.0`, `google-search-results`, `pydantic`, and `diskcache`. It is fully typed (`py.typed`) with zero agent bloat. Heavy demo dependencies (Streamlit, scikit-learn, transformers) are kept as optional extras."*
