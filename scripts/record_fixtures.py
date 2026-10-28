"""
Record real SerpApi responses as demo fixtures (costs 1 credit per query).

The Streamlit demo's offline "demo mode" replays these files, so anyone can
run the app without an API key. Re-run this script to refresh them:

    python scripts/record_fixtures.py

    python scripts/record_fixtures.py          # demo-mode fixtures (3 queries)
    python scripts/record_fixtures.py --eval   # extra contested topics for eval/ (7 queries)

Only fields the demo needs are kept; the API key and SerpApi's per-search
metadata/links are stripped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "research_agent"))

from sample_queries import (  # noqa: E402
    EVAL_EXTRA_QUERIES,
    EVAL_FIXTURE_DIR,
    FIXTURE_DIR,
    SAMPLE_QUERIES,
    fixture_path,
)

from langchain_serpapi_provider import SerpApiRetriever  # noqa: E402

NUM_RESULTS = 10


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Record the extra eval topics into eval/fixtures.")
    args = parser.parse_args()

    api_key = os.getenv("SERPAPI_API_KEY", "")
    if not api_key:
        print("SERPAPI_API_KEY is required to record fixtures.")
        return 1

    if args.eval:
        targets = [(q, q) for q in EVAL_EXTRA_QUERIES]
        directory = EVAL_FIXTURE_DIR
    else:
        targets = list(SAMPLE_QUERIES)
        directory = FIXTURE_DIR

    for label, query in targets:
        retriever = SerpApiRetriever(
            api_key=api_key, engine="google_scholar", num_results=NUM_RESULTS, cache=None
        )
        raw = retriever._call_with_retry(query)
        params = {k: v for k, v in raw.get("search_parameters", {}).items() if k != "api_key"}
        payload = {
            "query": query,
            "engine": "google_scholar",
            "num_results": NUM_RESULTS,
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "response": {
                "organic_results": raw.get("organic_results", []),
                "search_information": raw.get("search_information", {}),
                "search_parameters": params,
            },
        }
        out = fixture_path(query, directory)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        text = out.read_text(encoding="utf-8")
        assert api_key not in text, "API key leaked into fixture!"
        print(f"[ok] {label}: {len(payload['response']['organic_results'])} results -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
