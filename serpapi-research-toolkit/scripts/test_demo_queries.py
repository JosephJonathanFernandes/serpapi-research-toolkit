"""
Test all 3 sample demo queries end-to-end through the full research agent pipeline:
- Live retrieval via SerpApiRetriever (Scholar)
- Exact cache response time measurement
- Embeddings computation
- KMeans clustering (k=3)
- Topic extraction via TF-IDF n-grams
- Divergence detection and cue inspection
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.abspath("."))

# Windows console encoding safeguard
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from langchain_serpapi_provider import SerpApiCache, SerpApiRetriever
from examples.research_agent.app import (
    cluster_papers,
    detect_divergence,
    extract_cluster_topics,
    get_embeddings,
)

cache = SerpApiCache()
queries = [
    "large language models theory of mind reasoning",
    "intermittent fasting insulin sensitivity metabolic health",
    "microplastics human gut microbiome health impacts",
]

print("\n" + "=" * 70)
print(" 🔬 END-TO-END PIPELINE AUDIT ON ALL 3 DEMO QUERIES")
print("=" * 70)

for q_idx, q in enumerate(queries, 1):
    print(f"\n--- Query {q_idx}: '{q}' ---")
    t0 = time.perf_counter()
    retriever = SerpApiRetriever(engine="google_scholar", num_results=10, cache=cache)
    docs = retriever.invoke(q)
    fetch_time = time.perf_counter() - t0

    # Test repeat cache retrieval speed
    t_cache_0 = time.perf_counter()
    docs_cached = retriever.invoke(q)
    cache_time = time.perf_counter() - t_cache_0

    print(f"Retrieval: {len(docs)} papers | Initial: {fetch_time:.2f}s | Cached: {cache_time*1000:.1f}ms")

    texts = [f"{d.metadata.get('title', '')} {d.page_content}" for d in docs]
    embeddings = get_embeddings(texts)
    labels = cluster_papers(embeddings, k=3)

    clusters = defaultdict(list)
    cluster_indices = defaultdict(list)
    for idx, (doc, label) in enumerate(zip(docs, labels)):
        clusters[label].append(doc)
        cluster_indices[label].append(idx)

    for label, papers in sorted(clusters.items()):
        snippets = [p.page_content for p in papers]
        divergent, cues = detect_divergence(snippets)
        topic = extract_cluster_topics(texts, cluster_indices[label])
        badge = "[DIVERGENCE FLAGGED ⚠️]" if divergent else "[No divergence]"
        print(f"  Cluster {label+1}: '{topic}' ({len(papers)} papers) -> {badge}")

        if divergent:
            for p, (pos, neg) in zip(papers, cues):
                if pos or neg:
                    title_short = p.metadata.get("title", "")[:40]
                    cues_str = []
                    if pos:
                        cues_str.append(f"Affirmative: {pos[:2]}")
                    if neg:
                        cues_str.append(f"Cautionary: {neg[:2]}")
                    print(f"    • '{title_short}...' -> {' | '.join(cues_str)}")

print("\n" + "=" * 70)
print(" AUDIT COMPLETE")
print("=" * 70)
