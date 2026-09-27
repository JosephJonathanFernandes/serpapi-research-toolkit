"""
SerpApiRetriever — LangChain BaseRetriever implementation.

Wraps the SerpApi Python SDK, normalizes results across engines, applies
disk-based caching, and returns LangChain ``Document`` objects.

Supported engines
-----------------
- ``google_scholar`` (default) — academic papers with citation data
- ``google``         — general web search
- ``google_news``    — recent news articles
- ``google_jobs``    — job postings

Quick start::

    from langchain_serpapi_provider import SerpApiRetriever

    retriever = SerpApiRetriever(
        api_key="sk-...",        # or set SERPAPI_API_KEY env var
        engine="google_scholar",
        num_results=5,
    )
    docs = retriever.invoke("transformer attention mechanism")
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, model_validator

from .cache import SerpApiCache
from .schema import SerpApiResult, normalize_response

logger = logging.getLogger(__name__)

SUPPORTED_ENGINES = {"google_scholar", "google", "google_news", "google_jobs"}


class SerpApiRetriever(BaseRetriever):
    """LangChain retriever backed by SerpApi.

    Parameters
    ----------
    api_key:
        Your SerpApi key. Falls back to the ``SERPAPI_API_KEY`` environment
        variable if not provided.
    engine:
        The SerpApi engine to query. One of ``google_scholar``, ``google``,
        ``google_news``, or ``google_jobs``.
    num_results:
        Maximum number of results to return per query (default: 10).
    cache:
        A ``SerpApiCache`` instance. Pass ``None`` to disable caching.
    extra_params:
        Additional parameters forwarded verbatim to the SerpApi SDK call
        (e.g. ``{"hl": "en", "gl": "us"}``).
    """

    api_key: str = Field(default="", description="SerpApi API key.")
    engine: str = Field(default="google_scholar", description="SerpApi engine name.")
    num_results: int = Field(default=10, ge=1, le=100, description="Max results per query.")
    cache: Optional[Any] = Field(default=None, description="SerpApiCache instance or None.", exclude=True)
    extra_params: Dict[str, Any] = Field(default_factory=dict, description="Extra SerpApi params.")

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="before")
    @classmethod
    def _resolve_api_key(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        key = values.get("api_key") or os.getenv("SERPAPI_API_KEY", "")
        if not key:
            raise ValueError(
                "SerpApiRetriever requires an API key. "
                "Pass `api_key=...` or set the SERPAPI_API_KEY environment variable."
            )
        values["api_key"] = key

        engine = values.get("engine", "google_scholar")
        if engine not in SUPPORTED_ENGINES:
            raise ValueError(
                f"Unsupported engine '{engine}'. Choose from: {sorted(SUPPORTED_ENGINES)}"
            )

        # Instantiate a default cache if the caller didn't pass one
        if values.get("cache") is None:
            values["cache"] = SerpApiCache()

        return values

    # ------------------------------------------------------------------
    # LangChain BaseRetriever interface
    # ------------------------------------------------------------------

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[Document]:
        """Retrieve documents for *query* from SerpApi."""
        raw = self._fetch(query)
        response = normalize_response(raw, engine=self.engine, query=query)
        return [self._to_document(r) for r in response.results[: self.num_results]]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch(self, query: str) -> Dict[str, Any]:
        """Return raw SerpApi response dict, consulting cache first."""
        if self.cache is not None:
            cached = self.cache.get(query, self.engine)
            if cached is not None:
                logger.info("Cache hit for query=%r engine=%s", query, self.engine)
                return cached

        raw = self._call_serpapi(query)

        if self.cache is not None:
            self.cache.set(query, self.engine, raw)

        return raw

    def _call_serpapi(self, query: str) -> Dict[str, Any]:
        """Make the live SerpApi network call and return the raw dict."""
        try:
            from serpapi import GoogleSearch  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The `google-search-results` package is required. "
                "Install it with: pip install google-search-results"
            ) from exc

        params: Dict[str, Any] = {
            "q": query,
            "engine": self.engine,
            "api_key": self.api_key,
            "num": self.num_results,
            **self.extra_params,
        }
        # Google Scholar uses `num` for pagination; `num` is not standard for
        # all engines — some use `num_results`. Keep it simple for now.
        if self.engine == "google_jobs":
            params.pop("num", None)

        logger.debug("SerpApiRetriever calling engine=%s q=%r", self.engine, query)
        search = GoogleSearch(params)
        raw = dict(search.get_dict())
        if "error" in raw:
            raise RuntimeError(f"SerpApi error ({self.engine}): {raw['error']}")
        return raw

    @staticmethod
    def _to_document(result: SerpApiResult) -> Document:
        """Convert a ``SerpApiResult`` into a LangChain ``Document``."""
        metadata: Dict[str, Any] = {
            "title": result.title,
            "url": result.url,
            "source_engine": result.source_engine,
        }
        # Include non-None optional fields
        for field in (
            "citation_count",
            "publication_year",
            "authors",
            "publication_info",
            "source_name",
            "published_date",
            "company_name",
            "location",
        ):
            value = getattr(result, field, None)
            if value is not None:
                metadata[field] = value

        return Document(
            page_content=result.snippet or result.title,
            metadata=metadata,
        )
