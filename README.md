# 🔬 serpapi-research-toolkit

A **LangChain-compatible provider for SerpApi** that lets you drop academic paper
retrieval, news search, job search, and web search into any LangChain application
with one import — plus a polished **Streamlit research agent** demo that clusters
papers and flags possible divergent findings for human review.

---

## Table of Contents

- [What the plugin does](#what-the-plugin-does)
- [Repository layout](#repository-layout)
- [Install the plugin](#install-the-plugin)
- [Quick start: retriever standalone](#quick-start-retriever-standalone)
- [Quick start: inside a LangChain agent](#quick-start-inside-a-langchain-agent)
- [Auto engine selection](#auto-engine-selection)
- [Disk cache](#disk-cache)
- [Run the demo app](#run-the-demo-app)
- [Running tests](#running-tests)
- [Environment variables](#environment-variables)
- [Supported engines](#supported-engines)
- [Schema reference](#schema-reference)
- [Contributing](#contributing)

---

## What the plugin does

`langchain-serpapi-provider` wraps the [SerpApi](https://serpapi.com) Python SDK
and exposes it as a first-class LangChain component:

| Component | What it does |
|---|---|
| `SerpApiRetriever` | `BaseRetriever` subclass — call `.invoke(query)` and get `Document` objects |
| `SerpApiCache` | Disk-based cache (via `diskcache`) keyed by SHA-256(query + engine). Default TTL: 24 h |
| `SerpApiResult` | Normalized Pydantic schema for results across all four engines |
| `select_engine()` | Heuristic keyword-based engine picker — no ML, fast, predictable |
| `build_serpapi_tool()` | Wraps the retriever in a LangChain `Tool` for agent use |
| `build_auto_engine_tool()` | Same, but auto-selects the engine per query |

**Supported SerpApi engines:** `google_scholar`, `google`, `google_news`, `google_jobs`

Each engine returns different JSON. The plugin normalizes everything into a
consistent `SerpApiResult` Pydantic model so your application code doesn't need to
know which engine was used.

---

## Repository layout

```
serpapi-research-toolkit/
├── langchain_serpapi_provider/   # pip-installable plugin
│   ├── __init__.py               # public API surface
│   ├── retriever.py              # SerpApiRetriever (BaseRetriever)
│   ├── schema.py                 # normalized Pydantic models + engine normalizers
│   ├── cache.py                  # SerpApiCache (diskcache-backed)
│   └── tool.py                   # LangChain Tool wrapper + select_engine()
├── examples/
│   └── research_agent/
│       ├── app.py                # Streamlit demo app
│       └── requirements.txt      # demo-only deps (Streamlit, spaCy, FAISS, …)
├── tests/
│   ├── test_retriever.py         # retriever unit tests (mocked SerpApi SDK)
│   ├── test_cache.py             # cache unit tests (mocked diskcache)
│   ├── test_tool.py              # engine selector tests
│   └── test_schema.py            # schema normalization tests
├── pyproject.toml                # plugin package definition
├── .env.example                  # copy to .env and fill in SERPAPI_API_KEY
└── README.md
```

---

## Install the plugin

**From this repo (development / editable install):**

```bash
pip install -e ".[dev]"
```

**From PyPI (once published):**

```bash
pip install langchain-serpapi-provider
```

**Minimum requirements:** Python 3.10+, `langchain-core>=0.2`, `pydantic>=2.0`

Copy `.env.example` → `.env` and fill in your API key:

```bash
cp .env.example .env
# then edit .env
```

---

## Quick start: retriever standalone

```python
from dotenv import load_dotenv
from langchain_serpapi_provider import SerpApiRetriever

load_dotenv()  # reads SERPAPI_API_KEY from .env

# --- Google Scholar (academic papers) ---
retriever = SerpApiRetriever(
    engine="google_scholar",
    num_results=5,
)
docs = retriever.invoke("transformer attention mechanism")

for doc in docs:
    print(doc.metadata["title"])
    print(f"  Citations : {doc.metadata.get('citation_count')}")
    print(f"  Year      : {doc.metadata.get('publication_year')}")
    print(f"  URL       : {doc.metadata.get('url')}")
    print(f"  Snippet   : {doc.page_content[:120]}…")
    print()
```

**Expected output** (truncated):

```
Attention Is All You Need
  Citations : 95241
  Year      : 2017
  URL       : https://arxiv.org/abs/1706.03762
  Snippet   : We propose a new simple network architecture, the Transformer…
```

---

## Quick start: inside a LangChain agent

```python
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain import hub

from langchain_serpapi_provider import build_serpapi_tool

load_dotenv()

# Build the SerpApi tool (Scholar engine by default)
serpapi_tool = build_serpapi_tool(
    engine="google_scholar",
    num_results=5,
    name="scholar_search",
    description=(
        "Search Google Scholar for academic papers. "
        "Returns titles, URLs, abstracts, and citation counts. "
        "Use for any question about research, studies, or scientific papers."
    ),
)

# Wire up a ReAct agent
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm=llm, tools=[serpapi_tool], prompt=prompt)
executor = AgentExecutor(agent=agent, tools=[serpapi_tool], verbose=True)

result = executor.invoke({
    "input": "What are the most cited papers on large language model alignment?"
})
print(result["output"])
```

**Auto-selecting engine** (scholar / news / jobs / web based on query keywords):

```python
from langchain_serpapi_provider import build_auto_engine_tool

auto_tool = build_auto_engine_tool(num_results=5)
# auto_tool automatically routes:
#   "latest AI news"         → google_news
#   "ML engineer jobs"       → google_jobs
#   "attention paper 2017"   → google_scholar
#   "how to install python"  → google
```

---

## Auto engine selection

`select_engine(query: str) -> str` picks the engine using simple keyword matching —
no model, no external call, deterministic:

```python
from langchain_serpapi_provider import select_engine

select_engine("BERT paper citation study")   # → "google_scholar"
select_engine("breaking news OpenAI today")  # → "google_news"
select_engine("ML engineer remote hiring")   # → "google_jobs"
select_engine("best pizza recipe")           # → "google"
```

Keyword sets live in `tool.py` and are easy to extend.

---

## Disk cache

The cache is enabled by default and prevents burning API credits on repeated
queries during development.

```python
from langchain_serpapi_provider.cache import SerpApiCache

# Custom dir + 1-hour TTL
cache = SerpApiCache(
    cache_dir="/tmp/my_serpapi_cache",
    ttl=3600,
)

# Pass to the retriever
from langchain_serpapi_provider import SerpApiRetriever
retriever = SerpApiRetriever(engine="google_scholar", cache=cache)

# Disable caching (e.g. production)
retriever_nocache = SerpApiRetriever(engine="google", cache=None)
```

Cache keys are SHA-256 hashes of `(query.strip().lower(), engine)` so they're
stable across sessions. Set `SERPAPI_CACHE_TTL=0` in `.env` to disable globally.

---

## Run the demo app

The Streamlit app lives in `examples/research_agent/` and is **decoupled** from
the plugin — it just `pip install`s it as a dependency.

```bash
# 1. Install plugin (from repo root)
pip install -e .

# 2. Install demo-only deps
cd examples/research_agent
pip install -r requirements.txt

# 3. Download spaCy model (for embedding)
python -m spacy download en_core_web_md

# 4. Ensure .env has your SERPAPI_API_KEY (or paste it into the sidebar)
cp ../../.env.example ../../.env
# edit .env …

# 5. Launch
streamlit run app.py
```

**What the demo does:**

1. You enter a research question (e.g. *"Does intermittent fasting improve metabolic health?"*)
2. The agent calls `SerpApiRetriever` (scholar engine) for the top N papers
3. Each paper's title + abstract is embedded with **spaCy** (`en_core_web_md`)
   or falls back to **TF-IDF + TruncatedSVD** if spaCy isn't available
4. Papers are clustered by cosine similarity using **KMeans**
5. Within each cluster, a heuristic keyword pass checks for papers with opposing
   conclusion signals (e.g. "improves" vs "no significant effect")
6. Clusters with such signals are labeled **"Possible divergent findings — needs
   human review"** — deliberately conservative, not "detected contradictions"
7. Results display as dark-mode cards: cluster → papers → citations → divergence badge

> **Note on FAISS:** `faiss-cpu` is listed in `requirements.txt` for future
> nearest-neighbour queries. The current pipeline uses KMeans directly on the
> embedding matrix; FAISS would be the natural next step for large corpora
> or retrieval-augmented expansion.

---

## Running tests

All tests mock the SerpApi SDK — no live API calls, no credits consumed.

```bash
# From repo root
pip install -e ".[dev]"
pytest
```

Test coverage:

| File | What's tested |
|---|---|
| `test_retriever.py` | Construction validation, all 4 engines, `num_results` limiting, empty results, cache hit / miss / bypass |
| `test_cache.py` | Key generation (case / whitespace normalization, SHA-256 length), disabled cache no-ops, enabled cache set/get/invalidate/clear/TTL, graceful degradation without `diskcache` |
| `test_schema.py` | Normalization for scholar / news / jobs / google, field mapping, description truncation, unknown engine fallback |
| `test_tool.py` | `select_engine` for all four engines, tie-breaking, empty query, case insensitivity |

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `SERPAPI_API_KEY` | ✅ | — | Your SerpApi API key |
| `SERPAPI_CACHE_DIR` | ❌ | `~/.serpapi_cache` | Cache directory path |
| `SERPAPI_CACHE_TTL` | ❌ | `86400` (24 h) | Cache TTL in seconds; `0` = disabled |

---

## Supported engines

| Engine | SerpApi name | Best for |
|---|---|---|
| Google Scholar | `google_scholar` | Academic papers, citation data |
| Google Web | `google` | General search |
| Google News | `google_news` | Recent news articles |
| Google Jobs | `google_jobs` | Job postings |

---

## Schema reference

```python
class SerpApiResult(BaseModel):
    # All engines
    title: str
    url: Optional[str]
    snippet: str
    source_engine: str          # e.g. "google_scholar"

    # Scholar only
    citation_count: Optional[int]
    publication_year: Optional[int]
    authors: Optional[str]
    publication_info: Optional[str]

    # News only
    source_name: Optional[str]
    published_date: Optional[str]

    # Jobs only
    company_name: Optional[str]
    location: Optional[str]
```

Non-applicable fields are `None` (never missing keys), so you can always safely
access `doc.metadata.get("citation_count")` regardless of engine.

---

## Contributing

1. Fork the repo and create a feature branch
2. Run `pytest` — all tests must pass
3. Add tests for any new behavior
4. Open a PR with a clear description

License: MIT
