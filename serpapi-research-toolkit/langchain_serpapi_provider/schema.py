"""
Normalized Pydantic schemas for SerpApi results across different engines.

Each SerpApi engine returns a different JSON shape. This module normalizes
them all into a single `SerpApiResult` model that callers can rely on
regardless of which engine was queried.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl, model_validator


class SerpApiResult(BaseModel):
    """A single, engine-agnostic search result from SerpApi."""

    title: str = Field(..., description="Title of the result.")
    url: Optional[str] = Field(None, description="Link to the result.")
    snippet: str = Field(
        default="",
        description="Short summary / abstract excerpt.",
    )
    source_engine: str = Field(..., description="Which SerpApi engine produced this result.")

    # Scholar-only fields
    citation_count: Optional[int] = Field(None, description="Number of citations (Google Scholar only).")
    publication_year: Optional[int] = Field(None, description="Year of publication (Google Scholar only).")
    authors: Optional[str] = Field(None, description="Author string (Google Scholar only).")
    publication_info: Optional[str] = Field(None, description="Journal / conference info (Scholar only).")

    # News-only fields
    source_name: Optional[str] = Field(None, description="Name of the news outlet.")
    published_date: Optional[str] = Field(None, description="Publication date string (news only).")

    # Jobs-only fields
    company_name: Optional[str] = Field(None, description="Company posting the job.")
    location: Optional[str] = Field(None, description="Job location.")

    model_config = {"extra": "allow"}


class NormalizedSearchResponse(BaseModel):
    """Container wrapping all results for a single SerpApi query."""

    query: str
    engine: str
    results: List[SerpApiResult] = Field(default_factory=list)
    total_results: Optional[int] = Field(None, description="Reported total results count.")


# ---------------------------------------------------------------------------
# Engine-specific raw-to-normalized helpers
# ---------------------------------------------------------------------------

def _normalize_scholar(raw: dict) -> List[SerpApiResult]:
    items = raw.get("organic_results", [])
    out: List[SerpApiResult] = []
    for item in items:
        pub_info = item.get("publication_info", {})
        inline_links = item.get("inline_links", {})
        cited = inline_links.get("cited_by", {}).get("total")
        year = None
        summary = pub_info.get("summary", "")
        # Try to extract year from summary string like "L Smith, J Doe - 2023 - Journal"
        import re
        m = re.search(r"\b(19|20)\d{2}\b", summary)
        if m:
            year = int(m.group())

        # Extract all authors from structured list or fallback to summary prefix
        author_names = []
        if isinstance(pub_info.get("authors"), list):
            for a in pub_info["authors"]:
                if isinstance(a, dict) and a.get("name"):
                    author_names.append(a["name"])
                elif isinstance(a, str):
                    author_names.append(a)

        authors_str = ", ".join(author_names) if author_names else None
        if not authors_str and summary and " - " in summary:
            candidate = summary.split(" - ")[0].strip()
            if candidate:
                authors_str = candidate

        # Scholar link fallback: check top-level link, then resources list (e.g. PDF link)
        link = item.get("link")
        if not link and item.get("resources"):
            res_list = item.get("resources")
            if isinstance(res_list, list) and res_list and isinstance(res_list[0], dict):
                link = res_list[0].get("link")

        out.append(
            SerpApiResult(
                title=item.get("title", ""),
                url=link,
                snippet=item.get("snippet", ""),
                source_engine="google_scholar",
                citation_count=int(cited) if cited is not None else None,
                publication_year=year,
                authors=authors_str,
                publication_info=summary or None,
            )
        )
    return out


def _normalize_google(raw: dict) -> List[SerpApiResult]:
    items = raw.get("organic_results", [])
    return [
        SerpApiResult(
            title=item.get("title", ""),
            url=item.get("link"),
            snippet=item.get("snippet", ""),
            source_engine="google",
        )
        for item in items
    ]


def _normalize_google_news(raw: dict) -> List[SerpApiResult]:
    items = raw.get("news_results", [])
    return [
        SerpApiResult(
            title=item.get("title", ""),
            url=item.get("link"),
            snippet=item.get("snippet", ""),
            source_engine="google_news",
            source_name=item.get("source", {}).get("name") if isinstance(item.get("source"), dict) else item.get("source"),
            published_date=item.get("date"),
        )
        for item in items
    ]


def _normalize_google_jobs(raw: dict) -> List[SerpApiResult]:
    items = raw.get("jobs_results", [])
    results: List[SerpApiResult] = []
    for item in items:
        # Prioritize DIRECT job application link (e.g. company career site, LinkedIn)
        # over Google's internal search-wrapper share_link
        link = None
        apply_options = item.get("apply_options")
        if isinstance(apply_options, list) and apply_options:
            first_opt = apply_options[0]
            if isinstance(first_opt, dict) and first_opt.get("link"):
                link = first_opt["link"]

        # Fallback to Google's internal share_link if direct apply options are missing
        if not link:
            link = item.get("share_link")

        # Fallback for description from job_highlights if description is empty
        snippet = item.get("description", "")
        if not snippet and item.get("job_highlights"):
            highlights = item.get("job_highlights")
            if isinstance(highlights, list):
                snippet = " ".join(
                    str(h.get("items", [])) if isinstance(h, dict) else str(h)
                    for h in highlights
                )

        results.append(
            SerpApiResult(
                title=item.get("title", ""),
                url=link,
                snippet=snippet[:500] if snippet else "",
                source_engine="google_jobs",
                company_name=item.get("company_name"),
                location=item.get("location"),
            )
        )
    return results


_NORMALIZERS = {
    "google_scholar": _normalize_scholar,
    "google": _normalize_google,
    "google_news": _normalize_google_news,
    "google_jobs": _normalize_google_jobs,
}


def normalize_response(raw: dict, engine: str, query: str) -> NormalizedSearchResponse:
    """Convert a raw SerpApi response dict into a ``NormalizedSearchResponse``."""
    normalizer = _NORMALIZERS.get(engine, _normalize_google)
    results = normalizer(raw)
    return NormalizedSearchResponse(
        query=query,
        engine=engine,
        results=results,
        total_results=raw.get("search_information", {}).get("total_results"),
    )
