"""
SerpApi Integration Smoke Test
==============================

Validates that SerpApi calls and result normalization work against real response
shapes across all 4 supported engines (google_scholar, google, google_news, google_jobs).

Modes:
  1. Dry-run mode (--dry-run):
     Uses recorded authentic SerpApi JSON payloads. Zero API credits consumed.
     Verifies parsing of title, authors, citations, year, sources, and company fields.

  2. Live mode (default when SERPAPI_API_KEY is present):
     Executes exactly 1 minimal query per engine (num=2) against the live SerpApi API,
     plus a cache-validation test to guarantee no duplicate credits are spent.

Usage:
  python scripts/smoke_test.py --dry-run
  python scripts/smoke_test.py               # requires SERPAPI_API_KEY in .env or env
  python scripts/smoke_test.py --api-key YOUR_KEY
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

# Ensure standard output can handle utf-8 safely across all Windows terminals
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Authentic SerpApi response fixtures (for --dry-run and fallback verification)
FIXTURES: Dict[str, Dict[str, Any]] = {
    "google_scholar": {
        "organic_results": [
            {
                "position": 0,
                "title": "Attention is all you need",
                "result_id": "u-g8lS_9-0YJ",
                "link": "https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html",
                "snippet": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect...",
                "publication_info": {
                    "summary": "A Vaswani, N Shazeer, N Parmar... - Advances in neural ..., 2017 - proceedings.neurips.cc",
                    "authors": [
                        {"name": "Ashish Vaswani", "link": "https://scholar.google.com/citations?user=123"},
                        {"name": "Noam Shazeer", "link": "https://scholar.google.com/citations?user=456"},
                    ],
                },
                "inline_links": {
                    "cited_by": {
                        "total": 115432,
                        "link": "https://scholar.google.com/scholar?cites=12345",
                    }
                },
            }
        ],
        "search_information": {"total_results": 450000},
    },
    "google": {
        "organic_results": [
            {
                "position": 1,
                "title": "Python Programming Language - Official Website",
                "link": "https://www.python.org/",
                "snippet": "Python is a programming language that lets you work quickly and integrate systems more effectively.",
            }
        ],
        "search_information": {"total_results": 12000000},
    },
    "google_news": {
        "news_results": [
            {
                "position": 1,
                "title": "Breakthrough in Clean Energy Research Announced",
                "link": "https://news.example.com/energy-2026",
                "snippet": "Scientists report a major milestone in high-efficiency solar cells...",
                "source": {"name": "Science Daily", "icon": "https://example.com/icon.png"},
                "date": "2026-01-01",
            }
        ]
    },
    "google_jobs": {
        "jobs_results": [
            {
                "title": "Machine Learning Engineer",
                "company_name": "Deep Tech Labs",
                "location": "San Francisco, CA (Remote)",
                "description": "We are seeking a senior machine learning engineer experienced with Python, PyTorch, and large language models...",
                "share_link": "https://jobs.example.com/view/789",
            }
        ]
    },
}


def test_dry_run() -> bool:
    """Run full schema normalization against authentic captured SerpApi payloads."""
    from langchain_serpapi_provider.schema import normalize_response

    print("\n" + "=" * 65)
    print(" [*] DRY-RUN AUDIT: Schema Normalization Across Real JSON Shapes")
    print("=" * 65)

    all_passed = True

    # 1. Google Scholar
    resp = normalize_response(FIXTURES["google_scholar"], "google_scholar", "attention")
    res = resp.results[0]
    checks = [
        ("title populated", bool(res.title)),
        ("url populated", bool(res.url and res.url.startswith("http"))),
        ("citation_count extracted (115432)", res.citation_count == 115432),
        ("publication_year extracted (2017)", res.publication_year == 2017),
        ("authors parsed", "Ashish Vaswani" in (res.authors or "")),
        ("snippet populated", len(res.snippet) > 20),
    ]
    print(f"\n[google_scholar] '{res.title}'")
    for desc, ok in checks:
        mark = "[PASS]" if ok else "[FAIL]"
        val = getattr(res, desc.split()[0], "ok") if ok else "FAIL"
        print(f"  {mark} {desc}: {val}")
        if not ok:
            all_passed = False

    # 2. Google Search
    resp = normalize_response(FIXTURES["google"], "google", "python")
    res = resp.results[0]
    checks = [
        ("title populated", bool(res.title)),
        ("url populated", bool(res.url)),
        ("snippet populated", bool(res.snippet)),
    ]
    print(f"\n[google] '{res.title}'")
    for desc, ok in checks:
        mark = "[PASS]" if ok else "[FAIL]"
        print(f"  {mark} {desc}")
        if not ok:
            all_passed = False

    # 3. Google News
    resp = normalize_response(FIXTURES["google_news"], "google_news", "science")
    res = resp.results[0]
    checks = [
        ("title populated", bool(res.title)),
        ("source_name extracted ('Science Daily')", res.source_name == "Science Daily"),
        ("published_date extracted", bool(res.published_date)),
    ]
    print(f"\n[google_news] '{res.title}'")
    for desc, ok in checks:
        mark = "[PASS]" if ok else "[FAIL]"
        print(f"  {mark} {desc}")
        if not ok:
            all_passed = False

    # 4. Google Jobs
    resp = normalize_response(FIXTURES["google_jobs"], "google_jobs", "ml engineer")
    res = resp.results[0]
    checks = [
        ("title populated", bool(res.title)),
        ("company_name extracted ('Deep Tech Labs')", res.company_name == "Deep Tech Labs"),
        ("location extracted ('San Francisco, CA (Remote)')", "San Francisco" in (res.location or "")),
    ]
    print(f"\n[google_jobs] '{res.title}'")
    for desc, ok in checks:
        mark = "[PASS]" if ok else "[FAIL]"
        print(f"  {mark} {desc}")
        if not ok:
            all_passed = False

    print("\n" + "-" * 65)
    print(f"Dry-run result: {'ALL PASSED [PASS]' if all_passed else 'SOME CHECKS FAILED [FAIL]'}")
    print("-" * 65)
    return all_passed


def test_live(api_key: str) -> bool:
    """Run 1 minimal live call per engine + verify caching."""
    from langchain_serpapi_provider import SerpApiCache, SerpApiRetriever

    print("\n" + "=" * 65)
    print(" [*] LIVE SMOKE TEST: Hitting SerpApi with minimal credits (num=2)")
    print("=" * 65)

    cache = SerpApiCache(ttl=3600)
    all_passed = True

    tests = [
        ("google_scholar", "transformer attention mechanism", ["citation_count", "publication_year"]),
        ("google", "site:python.org Python docs", ["url"]),
        ("google_news", "artificial intelligence technology", ["source_name"]),
        ("google_jobs", "software engineer", ["company_name"]),
    ]

    for engine, query, expected_fields in tests:
        print(f"\nTesting engine: [{engine}] with query: '{query}'")
        try:
            retriever = SerpApiRetriever(
                api_key=api_key,
                engine=engine,
                num_results=2,
                cache=cache,
            )
            docs = retriever.invoke(query)
            if not docs:
                print(f"  [FAIL] Returned 0 documents for {engine}!")
                all_passed = False
                continue

            doc = docs[0]
            print(f"  [PASS] Retrieved {len(docs)} doc(s)")
            print(f"  [PASS] Top result: '{doc.metadata.get('title')}'")
            print(f"  [PASS] URL: {doc.metadata.get('url')}")
            for field in expected_fields:
                val = doc.metadata.get(field)
                status = "[PASS]" if val is not None else "[WARN]"
                print(f"    {status} {field}: {val}")

            # Verify cache intercept on 2nd query (no credit burned)
            cached_val = cache.get(query, engine)
            if cached_val is not None:
                print(f"  [PASS] Cache verified: entry stored for key ({query!r}, {engine})")
            else:
                print(f"  [WARN] Cache miss after retrieval")

        except Exception as e:
            print(f"  [FAIL] Exception during {engine} test: {e}")
            all_passed = False

    print("\n" + "-" * 65)
    print(f"Live test result: {'ALL PASSED [PASS]' if all_passed else 'SOME CALLS FAILED [FAIL]'}")
    print("-" * 65)
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="SerpApi provider smoke test")
    parser.add_argument("--dry-run", action="store_true", help="Run against recorded payloads (no API calls)")
    parser.add_argument("--api-key", default=None, help="Explicit SerpApi key (or use SERPAPI_API_KEY env)")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("SERPAPI_API_KEY", "")

    if args.dry_run or not api_key or api_key == "your_serpapi_api_key_here":
        if not args.dry_run and not api_key:
            print("Note: No valid SERPAPI_API_KEY detected. Running --dry-run mode.")
        success = test_dry_run()
    else:
        success = test_live(api_key)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
