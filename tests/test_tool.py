"""Tests for the engine selector and tool builder."""

from __future__ import annotations

import pytest
from langchain_serpapi_provider.tool import select_engine


class TestSelectEngine:
    def test_scholar_keywords(self):
        assert select_engine("latest research papers on transformers") == "google_scholar"
        assert select_engine("study on attention mechanisms") == "google_scholar"
        assert select_engine("peer-reviewed paper citation count") == "google_scholar"

    def test_news_keywords(self):
        assert select_engine("latest AI news today") == "google_news"
        assert select_engine("breaking announcement from OpenAI") == "google_news"

    def test_jobs_keywords(self):
        assert select_engine("ML engineer jobs remote hiring") == "google_jobs"
        assert select_engine("data scientist career salary") == "google_jobs"

    def test_generic_query_returns_google(self):
        assert select_engine("how to cook pasta") == "google"
        assert select_engine("best pizza in town") == "google"

    def test_scholar_beats_generic(self):
        # "paper" should outweigh no news/jobs signals
        result = select_engine("BERT paper performance benchmark findings")
        assert result == "google_scholar"

    def test_empty_query_returns_google(self):
        assert select_engine("") == "google"

    def test_multi_word_phrases(self):
        assert select_engine("literature review on neural networks") == "google_scholar"
        assert select_engine("peer reviewed study of sleep deprivation") == "google_scholar"
        assert select_engine("official press release from company") == "google_news"
        assert select_engine("current events in robotics this week") == "google_news"
        assert select_engine("looking for remote work as a developer") == "google_jobs"
        assert select_engine("work from home python engineer") == "google_jobs"

    def test_case_insensitive(self):
        assert select_engine("RESEARCH Papers on BERT") == "google_scholar"
        assert select_engine("LITERATURE REVIEW on LLMs") == "google_scholar"
