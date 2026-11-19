# Changelog

All notable changes to `langchain-serpapi-provider` are documented in this file.

## [0.1.0] - 2026-11-19

### Added
- **Core Package (`langchain-serpapi-provider`)**:
  - `SerpApiRetriever` implementing LangChain `BaseRetriever` with synchronous (`.invoke()`) and asynchronous (`.ainvoke()` / `_aget_relevant_documents`) execution.
  - Unified `SerpApiResult` Pydantic schema across `google_scholar`, `google`, `google_news`, and `google_jobs`.
  - Parameter-aware SHA-256 disk cache (`SerpApiCache`) hashing query, engine, pagination (`start`), `num_results`, language (`hl`), and year filters (`as_ylo`, `as_yhi`), with zero error caching.
  - Typed exception hierarchy (`SerpApiError`, `SerpApiAuthError`, `SerpApiQuotaError`, `SerpApiRateLimitError`, `SerpApiNetworkError`, `SerpApiRequestError`) and automatic exponential backoff retries on transient failures.
  - Agent tool builders (`build_serpapi_tool`, `build_auto_engine_tool`, `create_serpapi_tools`) with pluggable engine selectors.
  - Minimal runtime dependencies (`langchain-core>=0.2.0`, `google-search-results`, `pydantic>=2.0`, `diskcache`) and PEP 561 `py.typed` marker.
- **Research Agent Demo (`examples/research_agent/`)**:
  - Offline Demo Mode replaying recorded Google Scholar fixtures with no API key required.
  - Silhouette-guided KMeans topic clustering (`k = 2..6`) and OpenAlex / arXiv full-abstract enrichment.
  - Stance-toward-claim UI summarizing `supports`, `null / contrary`, `mixed`, and `unclear` counts per cluster across 3 tiers (Phrase cues, DeBERTa NLI cross-encoder, and LLM stance classifier with OpenAI or local Ollama).
  - Formula-injection-safe CSV and BibTeX `@misc` export.
- **Evaluation & External Validation (`eval/`)**:
  - External validation against the human-labeled **SciFact** dev benchmark (`eval/scifact/run_scifact.py`, 340 claim–abstract pairs).
  - Mechanical dev-only phrase lexicon builder (`eval/build_lexicon_v3.py`), 120-pair dev/test ablation runner (`eval/run_ablation.py`), and blind human-audit sheets (`eval/audit/`).
