"""
Measure the disk-cache speedup quoted in the README.

Issues ONE live Google Scholar request (1 credit) into a throwaway cache
directory, then repeats the identical request from the cache.

    python scripts/bench_cache.py
"""

from __future__ import annotations

import os
import statistics
import sys
import tempfile
import time

from dotenv import load_dotenv

load_dotenv()

from langchain_serpapi_provider import SerpApiCache, SerpApiRetriever  # noqa: E402

QUERY = "bilingual advantage executive function cognitive control"
WARM_RUNS = 50


def main() -> int:
    api_key = os.getenv("SERPAPI_API_KEY", "")
    if not api_key:
        print("SERPAPI_API_KEY is required (uses 1 credit).")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        with SerpApiCache(cache_dir=tmp, ttl=3600) as cache:
            retriever = SerpApiRetriever(api_key=api_key, num_results=10, cache=cache)

            t0 = time.perf_counter()
            docs = retriever.invoke(QUERY)
            cold = time.perf_counter() - t0

            warm = []
            for _ in range(WARM_RUNS):
                t0 = time.perf_counter()
                again = retriever.invoke(QUERY)
                warm.append(time.perf_counter() - t0)

    assert [d.metadata["title"] for d in docs] == [d.metadata["title"] for d in again]
    print(f"results            : {len(docs)}")
    print(f"cold (live fetch)  : {cold * 1000:9.1f} ms")
    print(f"warm median ({WARM_RUNS:>2})  : {statistics.median(warm) * 1000:9.2f} ms")
    print(f"warm max           : {max(warm) * 1000:9.2f} ms")
    print(f"speedup (median)   : {cold / statistics.median(warm):9.0f}x")
    print("Note: includes JSON normalization; excludes Streamlit rendering.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
