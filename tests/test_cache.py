"""Tests for SerpApiCache."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from langchain_serpapi_provider.cache import SerpApiCache, _cache_key


# ---------------------------------------------------------------------------
# Cache key tests
# ---------------------------------------------------------------------------

class TestCacheKey:
    def test_same_query_same_key(self):
        assert _cache_key("attention", "google_scholar") == _cache_key("attention", "google_scholar")

    def test_different_query_different_key(self):
        assert _cache_key("attention", "google_scholar") != _cache_key("bert", "google_scholar")

    def test_different_engine_different_key(self):
        assert _cache_key("attention", "google_scholar") != _cache_key("attention", "google")

    def test_case_insensitive(self):
        assert _cache_key("Attention", "google_scholar") == _cache_key("attention", "google_scholar")

    def test_whitespace_normalized(self):
        assert _cache_key("  attention  ", "google_scholar") == _cache_key("attention", "google_scholar")

    def test_key_is_hex_string(self):
        key = _cache_key("query", "google")
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex digest


# ---------------------------------------------------------------------------
# Disabled cache tests
# ---------------------------------------------------------------------------

class TestDisabledCache:
    def test_disabled_cache_always_misses(self):
        cache = SerpApiCache(enabled=False)
        result = cache.get("query", "google")
        assert result is None

    def test_disabled_cache_set_is_noop(self):
        cache = SerpApiCache(enabled=False)
        cache.set("query", "google", {"data": 1})  # should not raise
        assert cache.get("query", "google") is None

    def test_disabled_cache_enabled_property(self):
        cache = SerpApiCache(enabled=False)
        assert cache.enabled is False


# ---------------------------------------------------------------------------
# Enabled cache tests (mocked diskcache)
# ---------------------------------------------------------------------------

class TestEnabledCache:
    @pytest.fixture
    def mock_diskcache(self, tmp_path, monkeypatch):
        """Inject a fake diskcache module so SerpApiCache uses an in-memory store."""
        import sys

        class FakeCache:
            def __init__(self, *args, **kwargs):
                self._store = {}

            def get(self, key, default=None):
                entry = self._store.get(key)
                if entry is None:
                    return default
                value, expire_at = entry
                if expire_at is not None and time.monotonic() > expire_at:
                    del self._store[key]
                    return default
                return value

            def set(self, key, value, expire=None):
                expire_at = (time.monotonic() + expire) if expire else None
                self._store[key] = (value, expire_at)

            def delete(self, key):
                self._store.pop(key, None)

            def clear(self):
                self._store.clear()

            def close(self):
                pass

        fake_module = MagicMock()
        fake_module.Cache = FakeCache
        monkeypatch.setitem(sys.modules, "diskcache", fake_module)
        return FakeCache

    def test_get_miss_returns_none(self, mock_diskcache, tmp_path):
        cache = SerpApiCache(cache_dir=str(tmp_path), enabled=True)
        assert cache.get("missing query", "google") is None

    def test_set_then_get_returns_data(self, mock_diskcache, tmp_path):
        cache = SerpApiCache(cache_dir=str(tmp_path), enabled=True)
        data = {"organic_results": [{"title": "Test"}]}
        cache.set("my query", "google_scholar", data)
        result = cache.get("my query", "google_scholar")
        assert result == data

    def test_different_engines_independent(self, mock_diskcache, tmp_path):
        cache = SerpApiCache(cache_dir=str(tmp_path), enabled=True)
        data_scholar = {"source": "scholar"}
        data_news = {"source": "news"}
        cache.set("query", "google_scholar", data_scholar)
        cache.set("query", "google_news", data_news)

        assert cache.get("query", "google_scholar") == data_scholar
        assert cache.get("query", "google_news") == data_news

    def test_invalidate_removes_entry(self, mock_diskcache, tmp_path):
        cache = SerpApiCache(cache_dir=str(tmp_path), enabled=True)
        cache.set("query", "google", {"data": 1})
        cache.invalidate("query", "google")
        assert cache.get("query", "google") is None

    def test_clear_removes_all(self, mock_diskcache, tmp_path):
        cache = SerpApiCache(cache_dir=str(tmp_path), enabled=True)
        cache.set("q1", "google", {"a": 1})
        cache.set("q2", "google_scholar", {"b": 2})
        cache.clear()
        assert cache.get("q1", "google") is None
        assert cache.get("q2", "google_scholar") is None

    def test_ttl_property(self, mock_diskcache, tmp_path):
        cache = SerpApiCache(cache_dir=str(tmp_path), ttl=3600, enabled=True)
        assert cache.ttl == 3600

    def test_none_ttl_keeps_forever(self, mock_diskcache, tmp_path):
        cache = SerpApiCache(cache_dir=str(tmp_path), ttl=None, enabled=True)
        assert cache.ttl is None

    def test_context_manager(self, mock_diskcache, tmp_path):
        with SerpApiCache(cache_dir=str(tmp_path), enabled=True) as cache:
            cache.set("q", "google", {"x": 1})
            assert cache.get("q", "google") == {"x": 1}
        # close() called — no exception

    def test_missing_diskcache_falls_back_gracefully(self, monkeypatch, tmp_path):
        """If diskcache isn't installed, cache should degrade to no-op."""
        import langchain_serpapi_provider.cache as cache_module

        original = None
        try:
            import diskcache as _dc
            original = _dc
        except ImportError:
            pass

        # Simulate ImportError by making the import fail
        import builtins
        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "diskcache":
                raise ImportError("No module named 'diskcache'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        cache = SerpApiCache(cache_dir=str(tmp_path), enabled=True)
        # Should degrade to no-op
        assert cache.get("q", "google") is None
        cache.set("q", "google", {"data": 1})  # should not raise
