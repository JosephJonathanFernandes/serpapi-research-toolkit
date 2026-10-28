# 🔬 serpapi-research-toolkit

[![CI](https://github.com/JosephJonathanFernandes/serpapi-research-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/JosephJonathanFernandes/serpapi-research-toolkit/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Typing: Typed](https://img.shields.io/badge/typing-typed-green.svg)](https://peps.python.org/pep-0561/)

A hardened, **LangChain-native provider for SerpApi** designed for production RAG pipelines, autonomous agents, and academic literature analysis.

Drop multi-engine search (Google Scholar, Google Web, Google News, and Google Jobs) into any LangChain application or LCEL chain with **one import**, persistent disk caching, unified result normalization, and automated error retries. Includes a **Streamlit research agent** that clusters literature by topic similarity, chooses cluster counts via silhouette scores, and flags potential disagreement for human review.

---

## Why this instead of `langchain-community`?

LangChain includes a basic `SerpAPIWrapper` in `langchain-community`, but it was designed as a simple text-in/text-out utility. In production RAG systems and autonomous research loops, it falls short:

| Feature | `langchain-community` SerpAPIWrapper | `langchain-serpapi-provider` (This repo) |
|---|:---:|:---:|
| **Interface** | Low-level utility / text runner | First-class `BaseRetriever` (`.invoke()` & `.ainvoke()`) |
| **LCEL RAG Chains** | ❌ Requires custom parsing wrappers | ✅ Native runnable: `retriever \| format_docs \| prompt \| llm` |
| **Cross-Engine Schema** | ❌ Raw JSON with engine-specific formats | ✅ Unified Pydantic `SerpApiResult` across Scholar, Web, News, Jobs |
| **Response Caching** | ❌ None (burns API credits on repeat queries) | ✅ Multi-parameter SHA-256 disk cache (~1,400x speedup) |
| **Cache Key Robustness** | N/A | ✅ Hashes query, engine, pagination (`start`), `language`, year ranges |
| **Engine Routing** | ❌ Fixed single-engine configuration | ✅ Built-in heuristic selector + pluggable custom/LLM routing |
| **Async Execution** | ⚠️ Blocking wrapper | ✅ Native non-blocking `_aget_relevant_documents` & async tools |
| **Error Handling** | ❌ Unclassified exceptions | ✅ Typed exception hierarchy + exponential backoff retry on 429/5xx |
| **Dependencies** | ⚠️ Pulls broad `langchain-community` | ✅ Minimal: `langchain-core` + `google-search-results` only (`py.typed`) |

---

## 3-Line Quick Start

```bash
# 1. Install directly from GitHub
pip install git+https://github.com/JosephJonathanFernandes/serpapi-research-toolkit.git

# 2. Set your API key
export SERPAPI_API_KEY="your_api_key_here"

# 3. Retrieve papers as LangChain Documents
python -c "from langchain_serpapi_provider import SerpApiRetriever; print(SerpApiRetriever().invoke('attention mechanism')[0].metadata['title'])"
```

---

## Table of Contents

- [Quick Start: Standalone Retriever](#quick-start-standalone-retriever)
- [Quick Start: Modern LCEL RAG Chain](#quick-start-modern-lcel-rag-chain)
- [Quick Start: LangChain Agents](#quick-start-langchain-agents)
- [Hardened Architecture & Features](#hardened-architecture--features)
- [Streamlit Research Demo (with Offline Demo Mode)](#streamlit-research-demo)
- [Disagreement Detection & Benchmark](#disagreement-detection--benchmark)
- [Running Tests & Verifiability](#running-tests--verifiability)
- [Environment Variables](#environment-variables)
- [Supported Engines & Schema](#supported-engines--schema)

---

## Quick Start: Standalone Retriever

Use `SerpApiRetriever` anywhere you need LangChain `Document` objects. All metadata fields (`citation_count`, `publication_year`, `authors`, `url`) are normalized and accessible across engines.

```python
from dotenv import load_dotenv
from langchain_serpapi_provider import SerpApiRetriever

load_dotenv()

# Google Scholar (academic papers with citation data)
retriever = SerpApiRetriever(
    engine="google_scholar",
    num_results=5,
    year_from=2020,         # Restrict to papers published >= 2020
    language="en",
)

docs = retriever.invoke("bilingual advantage executive function cognitive control")

for doc in docs:
    print(f"Title    : {doc.metadata['title']}")
    print(f"Citations: {doc.metadata.get('citation_count')}")
    print(f"Year     : {doc.metadata.get('publication_year')}")
    print(f"URL      : {doc.metadata.get('url')}")
    print(f"Snippet  : {doc.page_content[:120]}...\n")
```

### Async Retrieval

`SerpApiRetriever` implements `_aget_relevant_documents`, running network calls in non-blocking worker threads:

```python
import asyncio

async def main():
    docs = await retriever.ainvoke("large language model alignment")
    print(f"Retrieved {len(docs)} documents asynchronously.")

asyncio.run(main())
```

---

## Quick Start: Modern LCEL RAG Chain

Because `SerpApiRetriever` is a standard `BaseRetriever`, it composes cleanly into LangChain Expression Language (LCEL) chains:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_serpapi_provider import SerpApiRetriever

retriever = SerpApiRetriever(engine="google_scholar", num_results=4)

def format_docs(docs):
    return "\n\n".join(
        f"Title: {d.metadata['title']} ({d.metadata.get('publication_year')})\n"
        f"Citations: {d.metadata.get('citation_count')}\n"
        f"Excerpt: {d.page_content}"
        for d in docs
    )

prompt = ChatPromptTemplate.from_template(
    "Synthesize the following academic findings to answer the question:\n\n"
    "{context}\n\n"
    "Question: {question}\n\n"
    "Synthesized Answer:"
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = chain.invoke("What does recent literature say about bilingual cognitive advantages?")
print(answer)
```

---

## Quick Start: LangChain Agents

Use `build_serpapi_tool()` to wrap a retriever into a tool for ReAct or tool-calling agents.

```python
from langchain_serpapi_provider import build_serpapi_tool, build_auto_engine_tool

# Fixed-engine tool
scholar_tool = build_serpapi_tool(
    engine="google_scholar",
    name="scholar_search",
    num_results=5,
)

# Auto-routing tool: inspects the query and chooses scholar, news, jobs, or web search
auto_search_tool = build_auto_engine_tool(
    name="auto_search",
    num_results=5,
)
```

### Pluggable Engine Selection & Overrides

The default selector uses word-boundary keyword matching (fast, zero external calls). For complex prompts, you can pass your own classifier or ask an LLM:

```python
def my_llm_selector(query: str) -> str:
    # Custom logic or prompt to route the query
    return "google_scholar" if "paper" in query else "google"

custom_tool = build_auto_engine_tool(
    engine_selector=my_llm_selector,
    # Or force an override: engine="google_news"
)
```

The chosen engine and selection method are logged at `INFO` level under `langchain_serpapi_provider.tool`.

---

## Hardened Architecture & Features

### 1. Robust Disk Caching & Zero Key Collisions

The disk cache (`SerpApiCache`, backed by `diskcache`) prevents burning API credits during development or agent loops.

- **Parameter-Aware Keying:** Cache keys are SHA-256 hashes of `(query, engine, request_params)`. Paginating (`start=10`), changing `num_results`, filtering years (`as_ylo`), or changing language (`hl`) generates distinct cache entries.
- **Error Protection:** Network failures and error responses are never stored in cache.
- **Measured Speedup:** In real benchmark runs (`python scripts/bench_cache.py`), cached lookups execute in **~0.5 ms** compared to **~700–2,800 ms** for cold network requests (~1,400x speedup).
- **Control:** Pass `cache=None` to disable caching entirely, or set `SERPAPI_CACHE_TTL=0` in your environment.

### 2. Typed Error Hierarchy & Exponential Backoff

SerpApi errors are parsed into explicit exception classes (`SerpApiError`):
- `SerpApiAuthError` (invalid or missing API key; non-retryable)
- `SerpApiQuotaError` (plan limit exceeded; non-retryable)
- `SerpApiRateLimitError` (rate limit reached; retried automatically with backoff)
- `SerpApiNetworkError` (connection drop or timeout; retried automatically)
- `SerpApiRequestError` (bad query parameter; non-retryable)

Transient rate limits and network errors retry automatically up to `max_retries=3` with exponential backoff (`delay = backoff_factor * 2**attempt`).

### 3. Lightweight Packaging

- Zero dependency on the full `langchain` package (only `langchain-core>=0.2.0`).
- Fully typed with PEP 561 marker (`py.typed`).

---

## Streamlit Research Demo

The repository includes a decoupled research agent demo in `examples/research_agent/app.py`:

```
serpapi-research-toolkit/
├── langchain_serpapi_provider/     # Core pip package (BaseRetriever, tools, cache, schema)
├── examples/
│   └── research_agent/
│       ├── app.py                  # Streamlit application (demo mode, exports, filters)
│       ├── analysis.py             # Clustering, embeddings, silhouette k selection
│       ├── disagreement.py         # 3-tier detector (Phrases, NLI cross-encoder, LLM judge)
│       ├── enrichment.py           # OpenAlex & arXiv abstract enrichment service
│       ├── demo_data.py            # Offline fixture replay loader
│       ├── export.py               # CSV (injection-safe) and BibTeX generators
│       ├── fixtures/               # Recorded SerpApi responses for offline demo
│       └── requirements.txt        # Demo-only dependencies
├── eval/                           # 120-pair benchmark, ablation study, & run_eval.py
└── tests/                          # 120 unit & integration tests (100% offline passing)
```

### Launch the Demo in 2 Minutes (Zero API Key Needed!)

The demo includes an **Offline Demo Mode** that replays pre-recorded SerpApi responses for three sample questions:
1. *Bilingual Advantage & Executive Function*
2. *Intermittent Fasting & Insulin Sensitivity*
3. *LLM Theory of Mind & Reasoning*

```bash
# 1. Install the plugin in editable mode
pip install -e .

# 2. Install demo dependencies
pip install -r examples/research_agent/requirements.txt

# 3. Launch Streamlit
streamlit run examples/research_agent/app.py
```

Check the **Demo mode** box in the sidebar: the entire pipeline (abstract enrichment, embedding, KMeans clustering, silhouette scoring, disagreement detection, and BibTeX/CSV export) runs instantly with zero API key configuration.

### Demo Features

- **Abstract Enrichment:** Automatically enriches truncated search snippets with full academic abstracts via OpenAlex and arXiv APIs (free, no API key).
- **Automated Cluster Sizing:** Silhouette analysis automatically selects $k$ across candidates ($k=2\dots 6$) with a minimum cluster size guard.
- **Cluster Label Sanity Checking:** Shows the top 3 most-cited papers alongside each auto-generated TF-IDF cluster label so users can evaluate topic coherence.
- **Research Export:** One-click downloads for **CSV** (with spreadsheet formula injection defense) and sanitized **BibTeX** `@misc` entries.
- **Sorting Options:** Sort papers within any cluster by search relevance, citation count, or indexed year.

---

## 🧪 Disagreement Detection & Evaluation Benchmark

A core challenge in automated literature review is surfacing contradictory evidence across studies. We built a **3-tier disagreement detection pipeline** backed by a hand-labeled benchmark in [`eval/`](eval/):

### The 3 Detector Tiers
- **Tier 1 (Directional Phrase Matching):** Fast, deterministic pattern lexicon that scans for affirmative vs cautionary/contrary outcome language with zero compute footprint.
- **Tier 2 (Transformer Cross-Encoder NLI):** Runs `cross-encoder/nli-deberta-v3-small` locally on CPU to score sentence-level contradiction probabilities.
- **Tier 3 (LLM Judge):** Uses a structured LangChain chat model (`BaseChatModel`) to evaluate meta-analytic contradiction, agreement, or unrelatedness.

### Benchmark Dataset (`eval/claim_pairs_labeled.json`)
- **120 audited paper pairs** across 10 contentious literature domains (bilingual advantage, power posing, ego depletion replication, growth mindset, intermittent fasting, screen time, vitamin D, omega-3, mindfulness, LLM theory of mind).
- Explicit 3-class ground truth: **42 contradict**, **43 agree**, and **35 unrelated** pairs with annotator rationales.

### Input Ablation: Google Scholar Snippets vs. Enriched Abstracts
We empirically tested whether pulling full abstracts from OpenAlex improves contradiction detection compared to short search snippets:

| Detector | Input Mode | Precision (95% CI) | Recall (95% CI) | F1 Score |
|---|---|:---:|:---:|:---:|
| **Tier 1 (Phrases V2)** | Google Scholar Snippets (100–250 chars) | **0.82** [0.52, 0.95] | **0.21** [0.12, 0.36] | **0.34** |
| **Tier 1 (Phrases V2)** | OpenAlex Enriched Abstracts (1k–2.5k chars) | 0.38 [0.18, 0.61] | 0.14 [0.07, 0.28] | 0.21 |
| **Tier 2 (NLI DeBERTa)** | Snippets | **0.50** [0.28, 0.72] | **0.19** [0.10, 0.33] | **0.28** |
| **Tier 2 (NLI DeBERTa)** | Enriched Abstracts (Head Truncation) | 0.08 [0.01, 0.33] | 0.02 [0.00, 0.12] | 0.04 |

*(See [eval/README.md](eval/README.md) for full error analysis on why NLI cross-encoders fail when truncated on introductory background text rather than conclusion sentences).*

---

## ⚠️ Limitations & Scientific Integrity

1. **Abstract & Snippet Evidence:** Google Scholar snippets and OpenAlex abstracts summarize findings but do not replace full PDF inspection. Statistical power, exact covariates, and subgroup analyses cannot be verified from abstracts alone.
2. **Possible Disagreement ≠ Proven Conflict:** Two papers often reach differing directional conclusions because they study different populations (e.g. pediatric vs elderly cohorts) or different dosage protocols. The detector surfaces candidate disputes for human verification.
3. **Small-N Clustering:** Clustering 10–20 papers per query is subject to small-sample noise; cluster counts are selected via silhouette scoring with minimum cluster size constraints ($n \ge 2$).
4. **SerpApi Credit Usage:** Live searches consume 1 API credit per query. Repeat queries are served for free from the multi-parameter SHA-256 disk cache in < 1 ms.

---

## Running Tests & Verifiability

All **120 unit and integration tests** run offline with **zero API credits** consumed:

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run full test suite
pytest
```

### Smoke Tests & Benchmarks

```bash
# 1. Offline dry-run audit (verifies schema normalization across all 4 engines)
python scripts/smoke_test.py --dry-run

# 2. Offline end-to-end pipeline check (fixtures replay)
python scripts/test_demo_queries.py --offline

# 3. Live smoke test (requires SERPAPI_API_KEY; issues 1 minimal call per engine)
python scripts/smoke_test.py

# 4. Cache benchmark (measures cold fetch vs warm cache latency)
python scripts/bench_cache.py
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|:---:|:---:|---|
| `SERPAPI_API_KEY` | Optional* | — | SerpApi API key (*required for live searches; not needed for offline demo mode or pytest) |
| `SERPAPI_CACHE_DIR` | No | `~/.serpapi_cache` | Filesystem path for response disk cache |
| `SERPAPI_CACHE_TTL` | No | `86400` (24 h) | Cache TTL in seconds; set to `0` to disable caching globally |

---

## Supported Engines & Schema

| Engine | Parameter | What it retrieves |
|---|---|---|
| Google Scholar | `google_scholar` | Academic papers, citation metrics, author lists, publication info |
| Google Web | `google` | General web search results |
| Google News | `google_news` | News articles with publisher source and publication dates |
| Google Jobs | `google_jobs` | Job postings with direct application links and location |

### Normalized Result Schema (`SerpApiResult`)

```python
class SerpApiResult(BaseModel):
    title: str
    url: Optional[str]
    snippet: str
    source_engine: str          # "google_scholar", "google", etc.

    # Google Scholar fields (None on other engines)
    citation_count: Optional[int]
    publication_year: Optional[int]
    authors: Optional[str]
    publication_info: Optional[str]

    # Google News fields
    source_name: Optional[str]
    published_date: Optional[str]

    # Google Jobs fields
    company_name: Optional[str]
    location: Optional[str]
```

---

## Contributing

1. Clone repository: `git clone https://github.com/JosephJonathanFernandes/serpapi-research-toolkit.git`
2. Create environment and install in editable mode: `pip install -e ".[dev,demo]"`
3. Run `pytest` — all tests must pass.
4. Submit pull requests with clear test coverage.

License: [MIT](LICENSE)
