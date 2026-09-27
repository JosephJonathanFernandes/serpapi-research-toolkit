"""
Disk-based cache for SerpApi responses.

Uses ``diskcache`` as the backend. Each entry is keyed by a SHA-256 hash
of (query, engine) so repeated calls during development don't burn API credits.

Usage::

    from langchain_serpapi_provider.cache import SerpApiCache

    cache = SerpApiCache()               # default: ~/.serpapi_cache, TTL=24h
    hit = cache.get("machine learning", "google_scholar")
    if hit is None:
        raw = serpapi_call(...)
        cache.set("machine learning", "google_scholar", raw)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = str(Path.home() / ".serpapi_cache")
_DEFAULT_TTL = 60 * 60 * 24  # 24 hours in seconds


def _cache_key(query: str, engine: str) -> str:
    """Return a stable hex-digest key for (query, engine)."""
    payload = json.dumps({"query": query.strip().lower(), "engine": engine}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class SerpApiCache:
    """Thread-safe disk cache backed by *diskcache*.

    Parameters
    ----------
    cache_dir:
        Directory where the cache is stored. Defaults to ``~/.serpapi_cache``.
    ttl:
        Time-to-live in seconds. Defaults to 86400 (24 hours). Pass ``None``
        to keep entries forever.
    enabled:
        Set to ``False`` to disable the cache entirely (useful in tests or
        production runs where fresh data is critical).
    """

    def __init__(
        self,
        cache_dir: str = _DEFAULT_CACHE_DIR,
        ttl: Optional[int] = _DEFAULT_TTL,
        enabled: bool = True,
    ) -> None:
        self._ttl = ttl
        self._enabled = enabled
        self._cache = None

        if enabled:
            try:
                import diskcache  # type: ignore

                self._cache = diskcache.Cache(cache_dir)
                logger.debug("SerpApiCache: using diskcache at %s (TTL=%s)", cache_dir, ttl)
            except ImportError:
                logger.warning(
                    "diskcache is not installed; falling back to no-op cache. "
                    "Install it with: pip install diskcache"
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, query: str, engine: str) -> Optional[Dict[str, Any]]:
        """Return cached raw SerpApi response dict, or ``None`` on miss."""
        if not self._enabled or self._cache is None:
            return None
        key = _cache_key(query, engine)
        value = self._cache.get(key)
        if value is not None:
            logger.debug("SerpApiCache HIT  key=%s", key[:12])
        else:
            logger.debug("SerpApiCache MISS key=%s", key[:12])
        return value  # type: ignore[return-value]

    def set(self, query: str, engine: str, data: Dict[str, Any]) -> None:
        """Store ``data`` in the cache under (query, engine)."""
        if not self._enabled or self._cache is None:
            return
        key = _cache_key(query, engine)
        self._cache.set(key, data, expire=self._ttl)
        logger.debug("SerpApiCache SET  key=%s", key[:12])

    def invalidate(self, query: str, engine: str) -> None:
        """Remove a single entry from the cache."""
        if self._cache is None:
            return
        key = _cache_key(query, engine)
        self._cache.delete(key)

    def clear(self) -> None:
        """Wipe all cached entries."""
        if self._cache is not None:
            self._cache.clear()
            logger.debug("SerpApiCache cleared")

    def close(self) -> None:
        """Release the underlying cache handle."""
        if self._cache is not None:
            self._cache.close()

    def __enter__(self) -> "SerpApiCache":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def ttl(self) -> Optional[int]:
        return self._ttl
