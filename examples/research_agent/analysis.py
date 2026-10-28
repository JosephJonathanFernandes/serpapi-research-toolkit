"""
Embedding, clustering and labeling helpers for the research demo.

Kept free of Streamlit imports so they can be unit tested and reused.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def embed_texts_spacy(nlp: Any, texts: Sequence[str]) -> np.ndarray:
    """Embed texts using spaCy's word vectors (300-dim for en_core_web_md)."""
    vecs = [nlp(t[:1000]).vector for t in texts]  # truncate to avoid OOM
    return np.array(vecs, dtype=np.float32)


def embed_texts_tfidf(texts: Sequence[str]) -> np.ndarray:
    """Fallback: TF-IDF + truncated SVD to produce dense vectors (up to 100-dim)."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X = vectorizer.fit_transform(list(texts))
    n_components = min(100, X.shape[1] - 1, X.shape[0] - 1)
    if n_components < 2:
        return X.toarray().astype(np.float32)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    return svd.fit_transform(X).astype(np.float32)


def embed_texts_sentence_transformer(
    texts: Sequence[str], model_name: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """Embed texts using SentenceTransformers or HuggingFace pipeline."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        model = SentenceTransformer(model_name)
        return np.asarray(model.encode(list(texts), show_progress_bar=False), dtype=np.float32)
    except ImportError:
        pass

    try:
        from transformers import pipeline  # type: ignore
        pipe = pipeline("feature-extraction", model=f"sentence-transformers/{model_name}")
        features = pipe(list(texts), truncation=True, max_length=512)
        # Mean pooling over token sequence
        pooled = [np.mean(np.array(f[0]), axis=0) for f in features]
        return np.array(pooled, dtype=np.float32)
    except Exception as exc:
        raise ImportError(
            "SentenceTransformer embedding requires `sentence-transformers` or `transformers` + `torch`: "
            "pip install sentence-transformers"
        ) from exc


def compare_embeddings(
    texts: Sequence[str],
    nlp: Optional[Any] = None,
    k_min: int = 2,
    k_max: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """Compare TF-IDF+SVD, spaCy, and Sentence-Transformer on cluster quality.

    Computes silhouette scores across candidate k values for each embedding method.
    Returns a dictionary mapping method name to its best_k, best_silhouette,
    all_scores, and dimensionality.
    """
    results: Dict[str, Dict[str, Any]] = {}

    # 1. TF-IDF + SVD (standard baseline)
    try:
        emb_tfidf = embed_texts_tfidf(texts)
        best_k, scores = choose_k(emb_tfidf, k_min=k_min, k_max=k_max)
        results["tfidf_svd"] = {
            "dim": int(emb_tfidf.shape[1]) if emb_tfidf.ndim > 1 else 1,
            "best_k": best_k,
            "best_silhouette": float(scores.get(best_k, 0.0)),
            "scores": scores,
            "available": True,
        }
    except Exception as exc:
        results["tfidf_svd"] = {"available": False, "error": str(exc)}

    # 2. spaCy (if model passed)
    if nlp is not None:
        try:
            emb_spacy = embed_texts_spacy(nlp, texts)
            best_k, scores = choose_k(emb_spacy, k_min=k_min, k_max=k_max)
            results["spacy"] = {
                "dim": int(emb_spacy.shape[1]),
                "best_k": best_k,
                "best_silhouette": float(scores.get(best_k, 0.0)),
                "scores": scores,
                "available": True,
            }
        except Exception as exc:
            results["spacy"] = {"available": False, "error": str(exc)}

    # 3. Sentence Transformer
    try:
        emb_st = embed_texts_sentence_transformer(texts)
        best_k, scores = choose_k(emb_st, k_min=k_min, k_max=k_max)
        results["sentence_transformer"] = {
            "dim": int(emb_st.shape[1]),
            "best_k": best_k,
            "best_silhouette": float(scores.get(best_k, 0.0)),
            "scores": scores,
            "available": True,
        }
    except Exception as exc:
        results["sentence_transformer"] = {"available": False, "error": str(exc)}

    return results


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """Scale rows to unit length so Euclidean KMeans approximates cosine clustering."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def cluster_papers(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Return a cluster label (int) for each embedding."""
    from sklearn.cluster import KMeans

    k = max(1, min(k, len(embeddings)))
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    return km.fit_predict(l2_normalize(embeddings))


def choose_k(
    embeddings: np.ndarray, k_min: int = 2, k_max: int = 6, min_cluster_size: int = 2
) -> Tuple[int, Dict[int, float]]:
    """Pick the cluster count with the best silhouette score.

    Returns ``(best_k, {k: silhouette})``. Silhouette on 10-20 points is a noisy
    guide and drifts toward many tiny clusters, so any k whose smallest cluster
    has fewer than ``min_cluster_size`` papers is skipped. With fewer than 4
    papers there is nothing to choose: returns ``min(2, n)`` and no scores.
    The UI still lets the user override the choice.
    """
    from sklearn.metrics import silhouette_score

    n = len(embeddings)
    if n < 4:
        return max(1, min(2, n)), {}

    X = l2_normalize(embeddings)
    scores: Dict[int, float] = {}
    for k in range(max(2, k_min), min(k_max, n - 1) + 1):
        labels = cluster_papers(embeddings, k)
        counts = np.bincount(labels)
        if len(counts) < 2 or counts.min() < min_cluster_size:
            continue
        scores[k] = float(silhouette_score(X, labels))
    if not scores:
        return 2, {}
    best_k = max(scores, key=lambda k: (scores[k], -k))  # ties -> fewer clusters
    return best_k, scores


# ---------------------------------------------------------------------------
# Cluster labels and representative papers
# ---------------------------------------------------------------------------


def extract_cluster_topics(all_texts: Sequence[str], cluster_indices: Sequence[int]) -> str:
    """Return up to two distinctive TF-IDF n-grams describing the cluster.

    These labels are auto-generated and can read awkwardly; the UI always shows
    the cluster's top papers next to the label so users can judge it.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2000)
        X = vec.fit_transform(list(all_texts))
        features = np.array(vec.get_feature_names_out())
        mean_tfidf = np.asarray(X[list(cluster_indices)].mean(axis=0)).flatten()
        top_idx = mean_tfidf.argsort()[::-1][:2]
        keywords = [features[i].title() for i in top_idx if mean_tfidf[i] > 0]
        return " & ".join(keywords) if keywords else "General Topic"
    except Exception:
        return "General Topic"


def sort_papers(docs: Sequence[Any], sort_by: str = "relevance") -> List[Any]:
    """Sort documents by ``relevance`` (original order), ``citations`` or ``year``."""
    docs = list(docs)
    if sort_by == "citations":
        return sorted(docs, key=lambda d: d.metadata.get("citation_count") or 0, reverse=True)
    if sort_by == "year":
        return sorted(docs, key=lambda d: d.metadata.get("publication_year") or 0, reverse=True)
    return docs


def top_papers(docs: Sequence[Any], n: int = 3) -> List[Any]:
    """The *n* most-cited papers, a quick way to sanity-check a cluster label."""
    return sort_papers(docs, "citations")[:n]
