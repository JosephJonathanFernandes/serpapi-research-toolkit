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

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, model_validator

from .cache import SerpApiCache
from .errors import (
    SerpApiError,
    SerpApiNetworkError,
    SerpApiRequestError,
    classify_error_message,
    is_empty_results_message,
)
from .schema import SerpApiResult, normalize_response

logger = logging.getLogger(__name__)

SUPPORTED_ENGINES = {"google_scholar", "google", "google_news", "google_jobs"}

# Maximum ``num`` accepted by SerpApi per engine (others do not take ``num``).
_MAX_NUM = {"google_scholar": 20, "google": 100}


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
        Maximum number of results to return per query (default: 10). Google
        Scholar accepts at most 20 per request; use ``start`` to paginate.
    cache:
        A ``SerpApiCache`` instance. If the argument is omitted a default
        cache is created; pass ``cache=None`` explicitly to disable caching.
    start:
        Result offset for pagination (``start=10`` returns the second page of 10).
    language:
        Interface language (``hl``), e.g. ``"en"``.
    year_from, year_to:
        Google Scholar only: restrict results to a publication-year range
        (``as_ylo`` / ``as_yhi``).
    max_retries:
        Retries (with exponential backoff) for rate-limit and network errors.
        Authentication and bad-request errors are never retried.
    backoff_factor:
        Base delay in seconds; the n-th retry waits ``backoff_factor * 2**n``.
    extra_params:
        Additional parameters forwarded verbatim to the SerpApi SDK call
        (e.g. ``{"gl": "us"}``). They take precedence over the fields above.
    """

    api_key: str = Field(default="", description="SerpApi API key.")
    engine: str = Field(default="google_scholar", description="SerpApi engine name.")
    num_results: int = Field(default=10, ge=1, le=100, description="Max results per query.")
    cache: Optional[Any] = Field(default=None, description="SerpApiCache instance or None.", exclude=True)
    start: int = Field(default=0, ge=0, description="Pagination offset.")
    language: Optional[str] = Field(default=None, description="Interface language (hl).")
    year_from: Optional[int] = Field(default=None, description="Scholar: earliest year.")
    year_to: Optional[int] = Field(default=None, description="Scholar: latest year.")
    max_retries: int = Field(default=3, ge=0, le=10, description="Retries on transient errors.")
    backoff_factor: float = Field(default=1.0, ge=0, description="Backoff base in seconds.")
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

        # Only create a default cache when the caller did not pass ``cache`` at
        # all. An explicit ``cache=None`` means "no caching".
        if "cache" not in values:
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

    async def _aget_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[Document]:
        """Async variant. The SerpApi SDK is synchronous, so the blocking call
        (including retry sleeps) runs in a worker thread and never blocks the
        event loop."""
        return await asyncio.to_thread(self._get_relevant_documents, query)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request_params(self, query: str) -> Dict[str, Any]:
        """Every parameter sent to SerpApi except the API key."""
        params: Dict[str, Any] = {"q": query, "engine": self.engine}
        max_num = _MAX_NUM.get(self.engine)
        if max_num is not None:
            if self.num_results > max_num:
                logger.warning(
                    "%s accepts at most num=%d per request; capping num_results=%d. "
                    "Use `start` to paginate.",
                    self.engine, max_num, self.num_results,
                )
            params["num"] = min(self.num_results, max_num)
        if self.start:
            params["start"] = self.start
        if self.language:
            params["hl"] = self.language
        if self.engine == "google_scholar":
            if self.year_from is not None:
                params["as_ylo"] = self.year_from
            if self.year_to is not None:
                params["as_yhi"] = self.year_to
        params.update(self.extra_params)
        return params

    def _cache_params(self, query: str) -> Dict[str, Any]:
        """Request params that identify a cache entry (query/engine are keyed separately)."""
        return {k: v for k, v in self._request_params(query).items() if k not in ("q", "engine")}

    def _fetch(self, query: str) -> Dict[str, Any]:
        """Return raw SerpApi response dict, consulting cache first."""
        cache_params = self._cache_params(query)
        if self.cache is not None:
            cached = self.cache.get(query, self.engine, params=cache_params)
            if cached is not None:
                logger.info("Cache hit for query=%r engine=%s", query, self.engine)
                return cached

        raw = self._call_with_retry(query)

        # Only successful responses reach this point: errors raise above.
        if self.cache is not None:
            self.cache.set(query, self.engine, raw, params=cache_params)

        return raw

    def _call_with_retry(self, query: str) -> Dict[str, Any]:
        attempt = 0
        while True:
            try:
                return self._call_serpapi(query)
            except SerpApiError as exc:
                if not exc.retryable or attempt >= self.max_retries:
                    raise
                delay = self.backoff_factor * (2 ** attempt)
                logger.warning(
                    "SerpApi %s (%s); retry %d/%d in %.1fs",
                    type(exc).__name__, exc, attempt + 1, self.max_retries, delay,
                )
                time.sleep(delay)
                attempt += 1

    def _call_serpapi(self, query: str) -> Dict[str, Any]:
        """Make the live SerpApi network call and return the raw dict.

        Raises a subclass of :class:`SerpApiError` on failure. A response that
        merely says "no results" is returned as an empty dict, not an error.
        """
        try:
            from serpapi import GoogleSearch  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The `google-search-results` package is required. "
                "Install it with: pip install google-search-results"
            ) from exc

        params = {**self._request_params(query), "api_key": self.api_key}

        logger.debug("SerpApiRetriever calling engine=%s q=%r", self.engine, query)
        try:
            raw = dict(GoogleSearch(params).get_dict())
        except SerpApiError:
            raise
        except Exception as exc:  # SDK / transport failures
            if isinstance(exc, OSError):  # requests errors derive from OSError
                raise SerpApiNetworkError(f"Network error calling SerpApi: {exc}") from exc
            if type(exc).__module__.startswith("serpapi"):
                raise SerpApiRequestError(f"SerpApi SDK error: {exc}") from exc
            raise

        error = raw.get("error")
        if error:
            if is_empty_results_message(str(error)):
                return {k: v for k, v in raw.items() if k != "error"}
            exc_cls = classify_error_message(str(error))
            raise exc_cls(f"SerpApi error ({self.engine}): {error}")
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
