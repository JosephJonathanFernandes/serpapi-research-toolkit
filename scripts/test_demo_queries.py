"""
Run the 3 sample demo queries end to end through the research pipeline.

    python scripts/test_demo_queries.py --offline   # recorded fixtures, no API key
    python scripts/test_demo_queries.py             # live SerpApi (uses credits)

Steps: retrieval -> embeddings -> silhouette-chosen k -> KMeans -> topic labels
-> possible-disagreement detection. Exits non-zero if any step yields nothing.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "research_agent"))

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from analysis import choose_k, cluster_papers, embed_texts_tfidf, extract_cluster_topics  # noqa: E402
from demo_data import load_fixture_docs  # noqa: E402
from disagreement import detect_possible_disagreement  # noqa: E402
from sample_queries import SAMPLE_QUERIES  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Replay recorded fixtures (no API key).")
    args = parser.parse_args()

    api_key = os.getenv("SERPAPI_API_KEY", "")
    offline = args.offline or not api_key
    if offline and not args.offline:
        print("No SERPAPI_API_KEY found; running offline against recorded fixtures.")

    cache = None
    if not offline:
        from dotenv import load_dotenv

        load_dotenv()
        from langchain_serpapi_provider import SerpApiCache, SerpApiRetriever

        cache = SerpApiCache()

    failures = 0
    print("=" * 70)
    print(f" END-TO-END PIPELINE CHECK ({'offline fixtures' if offline else 'live SerpApi'})")
    print("=" * 70)
    for label, query in SAMPLE_QUERIES:
        print(f"\n--- {label}: '{query}' ---")
        t0 = time.perf_counter()
        if offline:
            docs = load_fixture_docs(query)
        else:
            retriever = SerpApiRetriever(engine="google_scholar", num_results=10, cache=cache)
            docs = retriever.invoke(query)
        print(f"Retrieved {len(docs)} papers in {(time.perf_counter() - t0) * 1000:.0f} ms")
        if not docs:
            failures += 1
            continue

        texts = [f"{d.metadata.get('title', '')} {d.page_content}" for d in docs]
        embeddings = embed_texts_tfidf(texts)
        k, scores = choose_k(embeddings)
        labels = cluster_papers(embeddings, k)
        print(f"Chosen k={k} (silhouette: {', '.join(f'{kk}:{s:.2f}' for kk, s in scores.items())})")

        groups = defaultdict(list)
        for idx, label_ in enumerate(labels):
            groups[int(label_)].append(idx)
        for cl, idxs in sorted(groups.items()):
            topic = extract_cluster_topics(texts, idxs)
            result = detect_possible_disagreement([docs[i].page_content for i in idxs])
            flag = "POSSIBLE DISAGREEMENT" if result.flagged else "no flag"
            print(f"  Cluster {cl + 1}: '{topic}' ({len(idxs)} papers) -> {flag}")

    print("\n" + "=" * 70)
    print(" FAILED" if failures else " OK")
    print("=" * 70)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
