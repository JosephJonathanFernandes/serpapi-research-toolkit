"""Tests for SerpApiRetriever."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from langchain_serpapi_provider.retriever import SerpApiRetriever, SUPPORTED_ENGINES
from langchain_serpapi_provider.cache import SerpApiCache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SCHOLAR_RAW = {
    "organic_results": [
        {
            "title": "Attention Is All You Need",
            "link": "https://arxiv.org/abs/1706.03762",
            "snippet": "We propose a new simple network architecture, the Transformer...",
            "publication_info": {
                "summary": "A Vaswani, N Shazeer - 2017 - NeurIPS",
                "authors": [{"name": "Ashish Vaswani"}],
            },
            "inline_links": {
                "cited_by": {"total": 95000},
            },
        },
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "link": "https://arxiv.org/abs/1810.04805",
            "snippet": "We introduce a new language representation model called BERT...",
            "publication_info": {
                "summary": "J Devlin, MW Chang - 2019 - ACL",
                "authors": [{"name": "Jacob Devlin"}],
            },
            "inline_links": {
                "cited_by": {"total": 70000},
            },
        },
    ],
    "search_information": {"total_results": 2},
}

GOOGLE_RAW = {
    "organic_results": [
        {
            "title": "LangChain Docs",
            "link": "https://python.langchain.com",
            "snippet": "LangChain is a framework for building LLM-powered applications.",
        }
    ]
}

NEWS_RAW = {
    "news_results": [
        {
            "title": "OpenAI announces GPT-5",
            "link": "https://openai.com/blog/gpt-5",
            "snippet": "OpenAI today announced GPT-5...",
            "source": {"name": "OpenAI Blog"},
            "date": "2026-01-01",
        }
    ]
}

JOBS_RAW = {
    "jobs_results": [
        {
            "title": "ML Engineer",
            "company_name": "Acme Corp",
            "location": "Remote",
            "description": "We are looking for an ML Engineer to join our team.",
            "share_link": "https://jobs.example.com/1",
        }
    ]
}


def _make_retriever(engine="google_scholar", num_results=10, cache=None):
    """Create a retriever with a mocked-out SerpApi call."""
    if cache is None:
        cache = SerpApiCache(enabled=False)
    return SerpApiRetriever(
        api_key="test_key_123",
        engine=engine,
        num_results=num_results,
        cache=cache,
    )


# ---------------------------------------------------------------------------
# Construction / validation tests
# ---------------------------------------------------------------------------

class TestRetrieverConstruction:
    def test_requires_api_key(self):
        import os
        # Ensure env var not set
        old = os.environ.pop("SERPAPI_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="API key"):
                SerpApiRetriever(api_key="", engine="google")
        finally:
            if old:
                os.environ["SERPAPI_API_KEY"] = old

    def test_reads_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("SERPAPI_API_KEY", "env_key_abc")
        r = SerpApiRetriever(engine="google", cache=SerpApiCache(enabled=False))
        assert r.api_key == "env_key_abc"

    def test_rejects_unsupported_engine(self):
        with pytest.raises(ValueError, match="Unsupported engine"):
            SerpApiRetriever(api_key="k", engine="bing")

    def test_all_supported_engines_accepted(self):
        for engine in SUPPORTED_ENGINES:
            r = SerpApiRetriever(api_key="k", engine=engine, cache=SerpApiCache(enabled=False))
            assert r.engine == engine


# ---------------------------------------------------------------------------
# Retrieval tests (mocked SerpApi SDK)
# ---------------------------------------------------------------------------

class TestRetrieverDocuments:
    @patch("langchain_serpapi_provider.retriever.SerpApiRetriever._call_serpapi")
    def test_scholar_returns_documents(self, mock_call):
        mock_call.return_value = SCHOLAR_RAW
        retriever = _make_retriever(engine="google_scholar")
        docs = retriever.invoke("transformer architecture")

        assert len(docs) == 2
        assert docs[0].metadata["title"] == "Attention Is All You Need"
        assert docs[0].metadata["citation_count"] == 95000
        assert docs[0].metadata["publication_year"] == 2017
        assert docs[0].metadata["source_engine"] == "google_scholar"
        assert "Transformer" in docs[0].page_content

    @patch("langchain_serpapi_provider.retriever.SerpApiRetriever._call_serpapi")
    def test_google_returns_documents(self, mock_call):
        mock_call.return_value = GOOGLE_RAW
        retriever = _make_retriever(engine="google")
        docs = retriever.invoke("langchain documentation")

        assert len(docs) == 1
        assert docs[0].metadata["source_engine"] == "google"
        assert docs[0].metadata["url"] == "https://python.langchain.com"

    @patch("langchain_serpapi_provider.retriever.SerpApiRetriever._call_serpapi")
    def test_news_returns_documents(self, mock_call):
        mock_call.return_value = NEWS_RAW
        retriever = _make_retriever(engine="google_news")
        docs = retriever.invoke("latest AI news")

        assert len(docs) == 1
        assert docs[0].metadata["source_name"] == "OpenAI Blog"
        assert docs[0].metadata["published_date"] == "2026-01-01"

    @patch("langchain_serpapi_provider.retriever.SerpApiRetriever._call_serpapi")
    def test_jobs_returns_documents(self, mock_call):
        mock_call.return_value = JOBS_RAW
        retriever = _make_retriever(engine="google_jobs")
        docs = retriever.invoke("ML engineer jobs")

        assert len(docs) == 1
        assert docs[0].metadata["company_name"] == "Acme Corp"
        assert docs[0].metadata["location"] == "Remote"

    @patch("langchain_serpapi_provider.retriever.SerpApiRetriever._call_serpapi")
    def test_num_results_limits_output(self, mock_call):
        # Give back 2 scholar results but ask for only 1
        mock_call.return_value = SCHOLAR_RAW
        retriever = _make_retriever(engine="google_scholar", num_results=1)
        docs = retriever.invoke("transformers")
        assert len(docs) == 1

    @patch("langchain_serpapi_provider.retriever.SerpApiRetriever._call_serpapi")
    def test_empty_results(self, mock_call):
        mock_call.return_value = {"organic_results": []}
        retriever = _make_retriever(engine="google_scholar")
        docs = retriever.invoke("xyzzy impossible query")
        assert docs == []


# ---------------------------------------------------------------------------
# Caching tests
# ---------------------------------------------------------------------------

class TestRetrieverCaching:
    @patch("langchain_serpapi_provider.retriever.SerpApiRetriever._call_serpapi")
    def test_cache_prevents_second_api_call(self, mock_call):
        mock_call.return_value = SCHOLAR_RAW
        cache = SerpApiCache(enabled=False)  # memory-only stub

        # Inject a real cache hit manually
        real_cache = MagicMock()
        real_cache.get.return_value = SCHOLAR_RAW  # always returns cached data

        retriever = _make_retriever(engine="google_scholar", cache=real_cache)
        docs = retriever.invoke("transformers")

        # API should NOT have been called since cache returned a hit
        mock_call.assert_not_called()
        assert len(docs) == 2

    @patch("langchain_serpapi_provider.retriever.SerpApiRetriever._call_serpapi")
    def test_cache_is_populated_on_miss(self, mock_call):
        mock_call.return_value = SCHOLAR_RAW

        real_cache = MagicMock()
        real_cache.get.return_value = None  # cache miss

        retriever = _make_retriever(engine="google_scholar", cache=real_cache)
        retriever.invoke("transformers")

        mock_call.assert_called_once()
        real_cache.set.assert_called_once_with("transformers", "google_scholar", SCHOLAR_RAW)

    @patch("langchain_serpapi_provider.retriever.SerpApiRetriever._call_serpapi")
    def test_no_cache_always_calls_api(self, mock_call):
        mock_call.return_value = SCHOLAR_RAW
        retriever = _make_retriever(engine="google_scholar", cache=None)
        retriever.invoke("transformers")
        retriever.invoke("transformers")
        assert mock_call.call_count == 2
