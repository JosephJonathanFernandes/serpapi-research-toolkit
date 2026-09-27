"""Tests for schema normalization across engines."""

from __future__ import annotations

import pytest
from langchain_serpapi_provider.schema import normalize_response, SerpApiResult


SCHOLAR_RAW = {
    "organic_results": [
        {
            "title": "Attention Is All You Need",
            "link": "https://arxiv.org/abs/1706.03762",
            "snippet": "Transformer architecture...",
            "publication_info": {
                "summary": "A Vaswani - 2017 - NeurIPS",
                "authors": [{"name": "Ashish Vaswani"}],
            },
            "inline_links": {"cited_by": {"total": 95000}},
        }
    ]
}

NEWS_RAW = {
    "news_results": [
        {
            "title": "GPT-5 Released",
            "link": "https://news.example.com",
            "snippet": "OpenAI releases GPT-5",
            "source": {"name": "Tech News"},
            "date": "2026-01-01",
        }
    ]
}

JOBS_RAW = {
    "jobs_results": [
        {
            "title": "Data Scientist",
            "company_name": "TechCorp",
            "location": "New York",
            "description": "A" * 600,  # over 500 chars — should truncate
            "share_link": "https://jobs.example.com/123",
        }
    ]
}

GOOGLE_RAW = {
    "organic_results": [
        {"title": "Python Docs", "link": "https://python.org", "snippet": "Official Python docs"}
    ]
}


class TestScholarNormalization:
    def test_fields_populated(self):
        resp = normalize_response(SCHOLAR_RAW, "google_scholar", "transformers")
        r = resp.results[0]
        assert r.title == "Attention Is All You Need"
        assert r.url == "https://arxiv.org/abs/1706.03762"
        assert r.citation_count == 95000
        assert r.publication_year == 2017
        assert r.source_engine == "google_scholar"
        assert r.authors == "Ashish Vaswani"

    def test_query_and_engine_on_response(self):
        resp = normalize_response(SCHOLAR_RAW, "google_scholar", "my query")
        assert resp.query == "my query"
        assert resp.engine == "google_scholar"


class TestNewsNormalization:
    def test_news_fields(self):
        resp = normalize_response(NEWS_RAW, "google_news", "AI news")
        r = resp.results[0]
        assert r.title == "GPT-5 Released"
        assert r.source_name == "Tech News"
        assert r.published_date == "2026-01-01"
        assert r.source_engine == "google_news"


class TestJobsNormalization:
    def test_jobs_fields(self):
        resp = normalize_response(JOBS_RAW, "google_jobs", "data scientist jobs")
        r = resp.results[0]
        assert r.company_name == "TechCorp"
        assert r.location == "New York"
        assert len(r.snippet) <= 500  # truncated
        assert r.url == "https://jobs.example.com/123"

    def test_description_truncated_at_500(self):
        resp = normalize_response(JOBS_RAW, "google_jobs", "jobs")
        assert len(resp.results[0].snippet) == 500


class TestGoogleNormalization:
    def test_google_fields(self):
        resp = normalize_response(GOOGLE_RAW, "google", "python")
        r = resp.results[0]
        assert r.title == "Python Docs"
        assert r.source_engine == "google"


class TestUnknownEngine:
    def test_falls_back_to_google_normalizer(self):
        raw = {"organic_results": [{"title": "Fallback", "link": "http://x.com", "snippet": "x"}]}
        resp = normalize_response(raw, "unknown_engine", "query")
        assert resp.results[0].title == "Fallback"
