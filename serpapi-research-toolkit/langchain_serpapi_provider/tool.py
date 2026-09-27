"""
LangChain Tool wrapper for SerpApiRetriever plus a lightweight engine selector.

Engine selector heuristics
--------------------------
The ``select_engine`` function uses simple keyword matching — it is intentionally
NOT an ML classifier. The goal is fast, predictable behavior with no extra
dependencies.

Usage::

    from langchain_serpapi_provider.tool import select_engine, build_serpapi_tool

    engine = select_engine("latest breakthroughs in quantum computing news")
    # → "google_news"

    tool = build_serpapi_tool(api_key="...")
    result = tool.run("transformer attention papers")
"""

from __future__ import annotations

import re
from typing import Any, Optional

from langchain_core.tools import Tool

from .retriever import SerpApiRetriever


# ---------------------------------------------------------------------------
# Engine selector
# ---------------------------------------------------------------------------

_SCHOLAR_KEYWORDS = frozenset(
    [
        "paper", "papers", "study", "studies", "research", "researchers",
        "journal", "abstract", "citation", "citations", "thesis", "dissertation",
        "preprint", "arxiv", "peer-reviewed", "peer reviewed", "academic",
        "conference", "proceedings", "literature review", "meta-analysis",
        "experiment", "dataset", "benchmark", "findings",
    ]
)

_NEWS_KEYWORDS = frozenset(
    [
        "news", "latest", "recent", "today", "yesterday", "breaking",
        "headline", "announced", "announcement", "press release", "report",
        "update", "updates", "current events", "this week", "trending",
    ]
)

_JOBS_KEYWORDS = frozenset(
    [
        "job", "jobs", "hiring", "career", "careers", "employment",
        "position", "vacancy", "vacancies", "recruit", "recruiter",
        "salary", "intern", "internship", "full-time", "part-time",
        "remote work", "work from home",
    ]
)


def select_engine(query: str) -> str:
    """Pick the best SerpApi engine for *query* using keyword heuristics.

    Returns one of: ``"google_scholar"``, ``"google_news"``,
    ``"google_jobs"``, or ``"google"``.

    Uses normalized word-boundary and multi-word phrase matching so that
    both single terms (e.g. 'paper', 'salary') and compound phrases
    (e.g. 'literature review', 'work from home', 'peer-reviewed')
    match reliably without token collisions.
    """
    # Normalize query: lowercased alphanumeric words separated by single spaces
    words = re.findall(r"[a-z0-9]+", query.lower())
    if not words:
        return "google"

    normalized_query = " " + " ".join(words) + " "

    def _score(keywords: frozenset[str]) -> int:
        count = 0
        for kw in keywords:
            norm_kw = " ".join(re.findall(r"[a-z0-9]+", kw.lower()))
            if f" {norm_kw} " in normalized_query:
                count += 1
        return count

    scores = {
        "google_scholar": _score(_SCHOLAR_KEYWORDS),
        "google_news": _score(_NEWS_KEYWORDS),
        "google_jobs": _score(_JOBS_KEYWORDS),
    }

    best_engine, best_score = max(scores.items(), key=lambda kv: kv[1])

    if best_score == 0:
        return "google"

    return best_engine


# ---------------------------------------------------------------------------
# Tool builder
# ---------------------------------------------------------------------------

def _make_retriever_fn(retriever: SerpApiRetriever):
    """Return a callable suitable for wrapping in a LangChain Tool."""

    def _run(query: str) -> str:
        docs = retriever.invoke(query)
        if not docs:
            return "No results found."

        lines = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            title = meta.get("title", "(no title)")
            url = meta.get("url", "")
            snippet = doc.page_content or ""
            line = f"{i}. **{title}**"
            if url:
                line += f"\n   URL: {url}"
            if snippet:
                line += f"\n   {snippet}"
            # Scholar extras
            if meta.get("citation_count") is not None:
                line += f"\n   Citations: {meta['citation_count']}"
            if meta.get("publication_year") is not None:
                line += f"\n   Year: {meta['publication_year']}"
            lines.append(line)

        return "\n\n".join(lines)

    return _run


def build_serpapi_tool(
    api_key: str = "",
    engine: str = "google_scholar",
    num_results: int = 5,
    name: str = "serpapi_search",
    description: str = (
        "Search the web using SerpApi. "
        "Input should be a plain-English search query. "
        "Returns titles, URLs, and snippets."
    ),
    **retriever_kwargs: Any,
) -> Tool:
    """Create a LangChain ``Tool`` wrapping ``SerpApiRetriever``.

    Parameters
    ----------
    api_key:
        SerpApi key (falls back to ``SERPAPI_API_KEY`` env var).
    engine:
        SerpApi engine (default ``google_scholar``).
    num_results:
        Max results (default 5).
    name:
        Tool name shown to the agent.
    description:
        Tool description shown to the agent.
    **retriever_kwargs:
        Any additional kwargs forwarded to ``SerpApiRetriever``.

    Returns
    -------
    Tool
        A ready-to-use LangChain ``Tool`` instance.
    """
    retriever = SerpApiRetriever(
        api_key=api_key,
        engine=engine,
        num_results=num_results,
        **retriever_kwargs,
    )
    return Tool(
        name=name,
        description=description,
        func=_make_retriever_fn(retriever),
    )


def build_auto_engine_tool(
    api_key: str = "",
    num_results: int = 5,
    name: str = "serpapi_auto",
    description: str = (
        "Search the web, academic papers, news, or jobs using SerpApi. "
        "Automatically selects the best search engine for the query. "
        "Input should be a plain-English search query."
    ),
    **retriever_kwargs: Any,
) -> Tool:
    """Like ``build_serpapi_tool`` but auto-selects the engine per query."""
    from .cache import SerpApiCache

    cache = SerpApiCache()

    def _auto_run(query: str) -> str:
        engine = select_engine(query)
        retriever = SerpApiRetriever(
            api_key=api_key,
            engine=engine,
            num_results=num_results,
            cache=cache,
            **retriever_kwargs,
        )
        return _make_retriever_fn(retriever)(query)

    return Tool(name=name, description=description, func=_auto_run)
