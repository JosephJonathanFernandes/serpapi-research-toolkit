"""Offline demo mode: replay recorded SerpApi responses so the app works without a key."""

from __future__ import annotations

import json
from typing import List

from langchain_core.documents import Document

from langchain_serpapi_provider import SerpApiRetriever
from langchain_serpapi_provider.schema import normalize_response

from sample_queries import SAMPLE_QUERIES, fixture_path, normalize_query


def demo_queries_available() -> List[str]:
    """Sample queries that have a recorded fixture on disk."""
    return [q for _, q in SAMPLE_QUERIES if fixture_path(q).exists()]


def load_fixture_docs(query: str, num_results: int = 20) -> List[Document]:
    """Return Documents for *query* from its recorded fixture.

    Goes through the same ``normalize_response`` + ``SerpApiRetriever._to_document``
    path as live results, so the demo exercises the real schema code.

    Raises ``FileNotFoundError`` if the query was never recorded.
    """
    path = fixture_path(query)
    if not path.exists():
        raise FileNotFoundError(
            f"No recorded fixture for {normalize_query(query)!r}. Demo mode only "
            "supports the sample questions; add an API key for free-form queries."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    response = normalize_response(payload["response"], engine="google_scholar", query=query)
    return [SerpApiRetriever._to_document(r) for r in response.results[:num_results]]
