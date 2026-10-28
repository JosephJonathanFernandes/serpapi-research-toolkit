"""Tests for the hardening work: cache keys, errors/retry, async, params, LCEL."""

from __future__ import annotations

import asyncio
import logging
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from langchain_serpapi_provider import (
    SerpApiAuthError,
    SerpApiCache,
    SerpApiNetworkError,
    SerpApiQuotaError,
    SerpApiRateLimitError,
    SerpApiRequestError,
    SerpApiRetriever,
    build_auto_engine_tool,
    build_serpapi_tool,
)
from langchain_serpapi_provider.cache import _cache_key

SCHOLAR_RAW = {
    "organic_results": [
        {
            "title": "Paper A",
            "link": "https://example.org/a",
            "snippet": "Snippet A",
            "publication_info": {"summary": "X Y - 2020 - Journal"},
            "inline_links": {"cited_by": {"total": 7}},
        }
    ]
}


def _install_fake_serpapi(monkeypatch, responses):
    """Install a fake ``serpapi.GoogleSearch`` returning/raising ``responses`` in order."""
    calls = []
    queue = list(responses)

    class FakeGoogleSearch:
        def __init__(self, params):
            calls.append(dict(params))

        def get_dict(self):
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    module = types.ModuleType("serpapi")
    module.GoogleSearch = FakeGoogleSearch
    monkeypatch.setitem(sys.modules, "serpapi", module)
    return calls


def _retriever(**kwargs):
    kwargs.setdefault("cache", None)
    kwargs.setdefault("backoff_factor", 0)
    return SerpApiRetriever(api_key="k", **kwargs)


# --------------------------------------------------------------------------
# Cache keys
# --------------------------------------------------------------------------

class TestCacheKeyParams:
    def test_params_change_key(self):
        a = _cache_key("q", "google_scholar", {"num": 5})
        b = _cache_key("q", "google_scholar", {"num": 10})
        assert a != b

    def test_param_order_irrelevant(self):
        a = _cache_key("q", "google_scholar", {"num": 5, "hl": "en"})
        b = _cache_key("q", "google_scholar", {"hl": "en", "num": 5})
        assert a == b

    def test_no_params_equals_empty_params(self):
        assert _cache_key("q", "google") == _cache_key("q", "google", {})

    def test_retriever_params_do_not_collide(self):
        ten = _retriever(num_results=10)
        five = _retriever(num_results=5)
        paged = _retriever(num_results=5, start=5)
        years = _retriever(num_results=5, year_from=2015, year_to=2020)
        keys = {
            _cache_key("q", "google_scholar", r._cache_params("q"))
            for r in (ten, five, paged, years)
        }
        assert len(keys) == 4

    def test_api_key_not_in_cache_params(self):
        assert "api_key" not in _retriever()._cache_params("q")


class TestCacheEnv:
    def test_ttl_zero_disables(self):
        assert SerpApiCache(ttl=0).enabled is False

    def test_env_ttl_zero_disables(self, monkeypatch):
        monkeypatch.setenv("SERPAPI_CACHE_TTL", "0")
        assert SerpApiCache().enabled is False

    def test_env_ttl_value(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SERPAPI_CACHE_TTL", "123")
        monkeypatch.setenv("SERPAPI_CACHE_DIR", str(tmp_path))
        cache = SerpApiCache()
        try:
            assert cache.ttl == 123
        finally:
            cache.close()

    def test_invalid_env_ttl_falls_back(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SERPAPI_CACHE_TTL", "abc")
        monkeypatch.setenv("SERPAPI_CACHE_DIR", str(tmp_path))
        cache = SerpApiCache()
        try:
            assert cache.ttl == 86400
        finally:
            cache.close()

    def test_real_diskcache_roundtrip_with_params(self, tmp_path):
        pytest.importorskip("diskcache")
        with SerpApiCache(cache_dir=str(tmp_path)) as cache:
            cache.set("q", "google", {"a": 1}, params={"num": 5})
            assert cache.get("q", "google", params={"num": 5}) == {"a": 1}
            assert cache.get("q", "google", params={"num": 10}) is None


class TestExplicitNoCache:
    def test_cache_none_means_no_cache(self):
        assert SerpApiRetriever(api_key="k", cache=None).cache is None

    def test_omitted_cache_gets_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SERPAPI_CACHE_DIR", str(tmp_path))
        r = SerpApiRetriever(api_key="k")
        try:
            assert isinstance(r.cache, SerpApiCache)
        finally:
            r.cache.close()


# --------------------------------------------------------------------------
# Request params
# --------------------------------------------------------------------------

class TestRequestParams:
    def test_scholar_year_range_language_start(self, monkeypatch):
        calls = _install_fake_serpapi(monkeypatch, [SCHOLAR_RAW])
        r = _retriever(num_results=5, start=10, language="en", year_from=2015, year_to=2020)
        r.invoke("q")
        sent = calls[0]
        assert sent["num"] == 5 and sent["start"] == 10 and sent["hl"] == "en"
        assert sent["as_ylo"] == 2015 and sent["as_yhi"] == 2020
        assert sent["api_key"] == "k"

    def test_year_range_ignored_for_other_engines(self):
        r = _retriever(engine="google", year_from=2015)
        assert "as_ylo" not in r._request_params("q")

    def test_jobs_and_news_send_no_num(self):
        assert "num" not in _retriever(engine="google_jobs")._request_params("q")
        assert "num" not in _retriever(engine="google_news")._request_params("q")

    def test_scholar_num_capped_at_20(self):
        assert _retriever(num_results=50)._request_params("q")["num"] == 20

    def test_extra_params_override(self):
        r = _retriever(language="en", extra_params={"hl": "fr", "gl": "us"})
        p = r._request_params("q")
        assert p["hl"] == "fr" and p["gl"] == "us"


# --------------------------------------------------------------------------
# Errors and retry
# --------------------------------------------------------------------------

class TestErrors:
    def test_invalid_key_raises_auth_error_no_retry(self, monkeypatch):
        calls = _install_fake_serpapi(monkeypatch, [{"error": "Invalid API key. Your API key should be here: ..."}])
        with pytest.raises(SerpApiAuthError):
            _retriever().invoke("q")
        assert len(calls) == 1

    def test_quota_error_not_retried(self, monkeypatch):
        calls = _install_fake_serpapi(monkeypatch, [{"error": "Your account has run out of searches."}])
        with pytest.raises(SerpApiQuotaError):
            _retriever().invoke("q")
        assert len(calls) == 1

    def test_rate_limit_retried_then_succeeds(self, monkeypatch):
        calls = _install_fake_serpapi(
            monkeypatch,
            [{"error": "Too many requests, rate limit"}, {"error": "rate limit"}, SCHOLAR_RAW],
        )
        docs = _retriever().invoke("q")
        assert len(docs) == 1
        assert len(calls) == 3

    def test_rate_limit_gives_up_after_max_retries(self, monkeypatch):
        calls = _install_fake_serpapi(monkeypatch, [{"error": "rate limit"}] * 3)
        with pytest.raises(SerpApiRateLimitError):
            _retriever(max_retries=2).invoke("q")
        assert len(calls) == 3

    def test_network_error_retried(self, monkeypatch):
        calls = _install_fake_serpapi(monkeypatch, [ConnectionError("boom"), SCHOLAR_RAW])
        assert len(_retriever().invoke("q")) == 1
        assert len(calls) == 2

    def test_network_error_raised_after_retries(self, monkeypatch):
        _install_fake_serpapi(monkeypatch, [TimeoutError("t")] * 2)
        with pytest.raises(SerpApiNetworkError):
            _retriever(max_retries=1).invoke("q")

    def test_other_error_is_request_error(self, monkeypatch):
        calls = _install_fake_serpapi(monkeypatch, [{"error": "Unsupported `foo` parameter."}])
        with pytest.raises(SerpApiRequestError):
            _retriever().invoke("q")
        assert len(calls) == 1

    def test_empty_results_message_is_not_an_error(self, monkeypatch):
        _install_fake_serpapi(monkeypatch, [{"error": "Google hasn't returned any results for this query."}])
        assert _retriever().invoke("q") == []

    def test_errors_are_runtime_errors(self):
        assert issubclass(SerpApiAuthError, RuntimeError)

    def test_backoff_delays_are_exponential(self, monkeypatch):
        _install_fake_serpapi(monkeypatch, [{"error": "rate limit"}] * 3 + [SCHOLAR_RAW])
        with patch("langchain_serpapi_provider.retriever.time.sleep") as sleep:
            _retriever(backoff_factor=0.5).invoke("q")
        assert [c.args[0] for c in sleep.call_args_list] == [0.5, 1.0, 2.0]

    def test_errors_are_never_cached(self, monkeypatch):
        _install_fake_serpapi(monkeypatch, [{"error": "Invalid API key."}])
        cache = MagicMock()
        cache.get.return_value = None
        with pytest.raises(SerpApiAuthError):
            SerpApiRetriever(api_key="k", cache=cache).invoke("q")
        cache.set.assert_not_called()

    def test_success_is_cached_with_params(self, monkeypatch):
        _install_fake_serpapi(monkeypatch, [SCHOLAR_RAW])
        cache = MagicMock()
        cache.get.return_value = None
        SerpApiRetriever(api_key="k", cache=cache, num_results=3, year_from=2010).invoke("q")
        cache.set.assert_called_once_with(
            "q", "google_scholar", SCHOLAR_RAW, params={"num": 3, "as_ylo": 2010}
        )


# --------------------------------------------------------------------------
# Async
# --------------------------------------------------------------------------

class TestAsync:
    def test_ainvoke_returns_documents(self, monkeypatch):
        _install_fake_serpapi(monkeypatch, [SCHOLAR_RAW])
        docs = asyncio.run(_retriever().ainvoke("q"))
        assert docs[0].metadata["title"] == "Paper A"

    def test_ainvoke_runs_concurrently(self, monkeypatch):
        _install_fake_serpapi(monkeypatch, [SCHOLAR_RAW, SCHOLAR_RAW])

        async def go():
            r = _retriever()
            return await asyncio.gather(r.ainvoke("a"), r.ainvoke("b"))

        a, b = asyncio.run(go())
        assert len(a) == len(b) == 1

    def test_async_tool(self, monkeypatch):
        _install_fake_serpapi(monkeypatch, [SCHOLAR_RAW])
        tool = build_serpapi_tool(api_key="k", cache=None)
        out = asyncio.run(tool.ainvoke("q"))
        assert "Paper A" in out


# --------------------------------------------------------------------------
# Engine selection
# --------------------------------------------------------------------------

class TestAutoTool:
    def test_custom_selector_is_used_and_logged(self, monkeypatch, caplog):
        calls = _install_fake_serpapi(monkeypatch, [{"news_results": []}])
        tool = build_auto_engine_tool(
            api_key="k", cache=None, engine_selector=lambda q: "google_news"
        )
        with caplog.at_level(logging.INFO, logger="langchain_serpapi_provider.tool"):
            tool.invoke("anything")
        assert calls[0]["engine"] == "google_news"
        assert "google_news" in caplog.text and "selector" in caplog.text

    def test_engine_override_skips_selector(self, monkeypatch):
        calls = _install_fake_serpapi(monkeypatch, [{"organic_results": []}])
        selector = MagicMock(return_value="google_jobs")
        tool = build_auto_engine_tool(api_key="k", cache=None, engine="google", engine_selector=selector)
        tool.invoke("ML engineer jobs")
        assert calls[0]["engine"] == "google"
        selector.assert_not_called()

    def test_bad_selector_output_falls_back_to_google(self, monkeypatch):
        calls = _install_fake_serpapi(monkeypatch, [{"organic_results": []}])
        tool = build_auto_engine_tool(api_key="k", cache=None, engine_selector=lambda q: "bing")
        tool.invoke("q")
        assert calls[0]["engine"] == "google"

    def test_invalid_override_rejected(self):
        with pytest.raises(ValueError):
            build_auto_engine_tool(api_key="k", engine="bing")


# --------------------------------------------------------------------------
# LCEL integration (real langchain-core runnables, only SerpApi is faked)
# --------------------------------------------------------------------------

class TestLCEL:
    def test_retriever_in_lcel_chain(self, monkeypatch):
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnablePassthrough

        _install_fake_serpapi(monkeypatch, [SCHOLAR_RAW])
        retriever = _retriever(num_results=1)

        def format_docs(docs):
            return "\n".join(f"{d.metadata['title']} ({d.metadata['citation_count']} cites): {d.page_content}" for d in docs)

        prompt = ChatPromptTemplate.from_template("Context:\n{context}\n\nQ: {question}")
        seen = {}

        def capture(prompt_value):
            seen["prompt"] = prompt_value.to_string()
            return prompt_value

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | capture
            | FakeListChatModel(responses=["answer"])
            | StrOutputParser()
        )
        assert chain.invoke("what is paper A?") == "answer"
        assert "Paper A (7 cites): Snippet A" in seen["prompt"]

    def test_tool_is_bindable_and_invokable(self, monkeypatch):
        _install_fake_serpapi(monkeypatch, [SCHOLAR_RAW])
        tool = build_serpapi_tool(api_key="k", cache=None, name="scholar")
        out = tool.invoke("x")
        assert "Paper A" in out and "Citations: 7" in out
