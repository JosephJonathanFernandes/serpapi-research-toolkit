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
│       ├── app.py                  # Streamlit application (stance counts, demo mode, exports)
│       ├── analysis.py             # Clustering, embeddings, silhouette k selection
│       ├── disagreement.py         # 3-tier stance & pair detector (Phrases V1-V3, NLI, LLM)
│       ├── llm_backends.py         # OpenAI + zero-dependency local Ollama chat adapter
│       ├── lexicon_v3.json         # Dev-only greedily selected phrase lexicon
│       ├── enrichment.py           # OpenAlex & arXiv abstract enrichment service
│       ├── demo_data.py            # Offline fixture replay loader
│       ├── export.py               # CSV (injection-safe, with stance) and BibTeX generators
│       ├── fixtures/               # Recorded SerpApi responses for offline demo
│       └── requirements.txt        # Demo-only dependencies
├── eval/                           # SciFact external validation, 120-pair ablation, human audit sheets
└── tests/                          # 130 unit & integration tests (offline)
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

## 🧪 Stance & Disagreement Detection: External SciFact Validation + Internal Ablation

Instead of only predicting noisy pairwise contradictions between papers, the research agent classifies each paper's **stance toward a stated hypothesis** (`supports`, `null / contrary`, `mixed`, `unclear`) and shows stance counts and top-cited papers under each stance per topic cluster.

### 1. External Validation on SciFact (Human-Labeled Claims & Abstracts)

To avoid grading only on our own labels, we evaluate the stance classifiers on the public **SciFact** dev benchmark (Wadden et al., EMNLP 2020; **340 human-annotated claim–abstract pairs** across 300 scientific claims; none of these topics were seen when building any phrase lexicon).

| Backend | n | Accuracy | Macro-F1 | CONTRADICT P [95% CI] | CONTRADICT R [95% CI] | CONTRADICT F1 | SUPPORT F1 |
|:---|---:|---:|---:|:---:|:---:|---:|---:|
| `majority` — always SUPPORT | 340 | 0.41 | 0.19 | 0.00 n/a | 0.00 [0.00, 0.05] | **0.00** | 0.58 |
| `phrases-v1` — untuned phrase cues | 340 | 0.38 | 0.19 | 0.20 [0.04, 0.62] | 0.01 [0.00, 0.08] | **0.03** | 0.00 |
| `phrases-v2` — expanded phrase cues | 340 | 0.41 | 0.30 | 0.23 [0.08, 0.50] | 0.04 [0.01, 0.12] | **0.07** | 0.30 |
| `phrases-v3` — dev-selected phrase cues | 340 | 0.42 | 0.30 | 0.20 [0.04, 0.62] | 0.01 [0.00, 0.08] | **0.03** | 0.33 |
| `nli` — `nli-deberta-v3-small` (claim as hypothesis) | 340 | **0.48** | **0.42** | **0.56** [0.43, 0.68] | **0.46** [0.35, 0.58] | **0.51** | 0.17 |

Full report and confusion matrices: [`eval/scifact/results.md`](eval/scifact/results.md). Reproduce with `python eval/scifact/run_scifact.py --backend nli`.

- **Why NLI stance beats phrase cues on SciFact:** Phrase matching ignores the claim and only looks for positive/negative outcome wording, reaching **0.03–0.07 CONTRADICT F1**. Passing the paper's findings as premise and the explicit claim as hypothesis to `cross-encoder/nli-deberta-v3-small` reaches **0.51 CONTRADICT F1** (precision 0.56, recall 0.46, with 0/71 true `CONTRADICT` abstracts misclassified as `SUPPORT`).
- **Why the UI shows per-cluster stance counts:** NLI is conservative on `SUPPORT` (precision 0.81, recall 0.09 — extra detail in multi-sentence findings often scores `neutral`). Showing per-cluster stance counts (`supports`, `null / contrary`, `mixed`, `unclear`) surfaces those high-precision detections directly, whereas requiring two papers in a pair to simultaneously cross `SUPPORTS` and `NULL` thresholds rarely fires.

### 2. Internal 120-Pair Pairwise Ablation (Provisional / Unaudited Labels)

We also include a 120-pair pairwise dataset across 10 topics ([`eval/claim_pairs_labeled.json`](eval/claim_pairs_labeled.json), split by topic into **dev** = 36 pairs / 12 contradict and **test** = 84 pairs / 30 contradict).
- **Provenance warning:** Those 120 labels were drafted by an AI assistant in one pass and have **not yet been human-audited** (blind human-audit sheets and Cohen's $\kappa$ scorer are ready in [`eval/audit/`](eval/audit/make_audit_sheet.py)).
- **Clean vs. contaminated lexicons:** `phrases-v2` was written after reading snippets from all 10 topics (upper bound). `phrases-v3` ([`eval/build_lexicon_v3.py`](eval/build_lexicon_v3.py)) was selected mechanically on the **dev split only**.

| Detector | Snippet (Test P/R/F1) | Full abstract (Test P/R/F1) | Findings only (Test P/R/F1) |
|---|:---:|:---:|:---:|
| `phrases-v1` (untuned) | 0.00 / 0.00 / 0.00 | 0.50 / 0.03 / 0.06 | 0.00 / 0.00 / 0.00 |
| `phrases-v2` (tuned on same topics — upper bound) | 0.80 / 0.27 / **0.40** | 0.33 / 0.13 / 0.19 | 0.17 / 0.03 / 0.06 |
| `phrases-v3` (selected on dev only) | 0.75 / 0.10 / **0.18** | 0.44 / 0.13 / **0.21** | 0.20 / 0.03 / 0.06 |
| `nli-pair` (DeBERTa paper-vs-paper) | 0.58 / 0.23 / **0.33** | 0.00 / 0.00 / 0.00 | 0.44 / 0.13 / 0.21 |
| `nli-stance` (DeBERTa opposing stances) | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |

Full dev/test breakdown: [`eval/results.md`](eval/results.md). Reproduce with `python eval/run_ablation.py` (add `--llm` to evaluate Tier 3 via `OPENAI_API_KEY` or a local Ollama server).

---

## ⚠️ Limitations & Scientific Integrity

1. **Abstract & Snippet Evidence:** Google Scholar snippets and OpenAlex abstracts summarize findings but do not replace full PDF inspection. Statistical power, exact covariates, and subgroup analyses cannot be verified from abstracts alone.
2. **Possible Disagreement ≠ Proven Conflict:** Two papers often reach differing directional conclusions because they study different populations (e.g. pediatric vs elderly cohorts) or different dosage protocols. The detector surfaces candidate disputes for human verification.
3. **Small-N Clustering:** Clustering 10–20 papers per query is subject to small-sample noise; cluster counts are selected via silhouette scoring with minimum cluster size constraints ($n \ge 2$).
4. **SerpApi Credit Usage:** Live searches consume 1 API credit per query. Repeat queries are served for free from the multi-parameter SHA-256 disk cache in < 1 ms.

---

## Running Tests & Verifiability

All **130 unit and integration tests** run offline with **zero API credits** consumed:

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
