"""
Research Agent — Streamlit Demo
================================
Demonstrates the langchain_serpapi_provider plugin.

Features
--------
- Enter a research question → fetch top papers via SerpApi (Scholar engine)
- Embed each paper's title + snippet using spaCy or TF-IDF vectors
- Cluster papers by topic similarity (KMeans)
- Within each cluster, flag papers with possible divergent findings
  (opposing conclusion keywords) — labeled conservatively for human review
- Display results as clustered cards with citations

Run
---
    cd examples/research_agent
    streamlit run app.py
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Research Agent | SerpApi + LangChain",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Inline CSS — premium dark-mode design
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #0d1117;
    color: #e6edf3;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #30363d;
}

/* Main area */
.main .block-container { padding-top: 2rem; }

/* Hero header */
.hero {
    background: linear-gradient(135deg, #1f6feb22, #58a6ff11);
    border: 1px solid #1f6feb44;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
}
.hero h1 { font-size: 2rem; font-weight: 700; color: #58a6ff; margin: 0 0 0.4rem; }
.hero p  { color: #8b949e; margin: 0; font-size: 0.95rem; }

/* Cluster card */
.cluster-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.5rem;
    transition: border-color 0.2s;
}
.cluster-card:hover { border-color: #58a6ff55; }
.cluster-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #58a6ff;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Paper card */
.paper-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s;
}
.paper-card:hover { border-color: #388bfd; }
.paper-title { font-weight: 600; color: #e6edf3; margin-bottom: 0.3rem; font-size: 0.95rem; }
.paper-snippet { color: #8b949e; font-size: 0.85rem; line-height: 1.5; }
.paper-meta { margin-top: 0.5rem; font-size: 0.78rem; color: #6e7681; }
.paper-meta span { margin-right: 1rem; }
.paper-url a { color: #58a6ff; text-decoration: none; }
.paper-url a:hover { text-decoration: underline; }

/* Divergence badge */
.divergence-badge {
    background: linear-gradient(90deg, #3d1a00, #5a2800);
    border: 1px solid #f0883e55;
    border-radius: 6px;
    padding: 0.5rem 0.75rem;
    margin-top: 0.5rem;
    font-size: 0.8rem;
    color: #f0883e;
    display: flex;
    gap: 0.4rem;
    align-items: flex-start;
}

/* Stats row */
.stat-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.stat-value { font-size: 1.8rem; font-weight: 700; color: #58a6ff; }
.stat-label { font-size: 0.78rem; color: #8b949e; margin-top: 0.2rem; }

/* Input */
.stTextArea textarea { background: #161b22 !important; border-color: #30363d !important; color: #e6edf3 !important; border-radius: 8px !important; }
.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.6rem 2rem;
    transition: opacity 0.2s;
    width: 100%;
}
.stButton > button:hover { opacity: 0.85; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <h1>🔬 Research Agent</h1>
  <p>Enter a research question — the agent retrieves academic papers via SerpApi,
  clusters them by topic similarity, and flags possible divergent findings for
  human review.</p>
</div>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar config
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    api_key = st.text_input(
        "SerpApi API Key",
        value=os.getenv("SERPAPI_API_KEY", ""),
        type="password",
        help="Get yours at serpapi.com",
    )

    num_results = st.slider("Papers to retrieve", min_value=5, max_value=20, value=10)
    num_clusters = st.slider("Topic clusters", min_value=2, max_value=6, value=3)
    use_cache = st.checkbox("Enable disk cache (saves API credits)", value=True)

    st.divider()
    st.markdown("### 📖 About")
    st.caption(
        "Powered by **langchain-serpapi-provider** + **spaCy / TF-IDF** + **KMeans**.\n\n"
        "• **Indexed Year:** Reflects the canonical edition/year indexed by Google Scholar (which may aggregate preprints published earlier).\n\n"
        "• **Divergence:** Flags clusters where abstracts exhibit both affirmative and cautionary conclusion keywords on the same sub-topic."
    )


# ---------------------------------------------------------------------------
# Helper: lazy imports (don't slow down page load)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_spacy():
    try:
        import spacy  # type: ignore
        try:
            return spacy.load("en_core_web_md")
        except OSError:
            return None
    except ImportError:
        return None


@st.cache_resource
def load_faiss():
    try:
        import faiss  # type: ignore
        return faiss
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Embedding pipeline
# ---------------------------------------------------------------------------

def embed_texts_spacy(nlp, texts: List[str]) -> np.ndarray:
    """Embed texts using spaCy's word vectors (300-dim)."""
    vecs = []
    for t in texts:
        doc = nlp(t[:1000])  # truncate to avoid OOM
        vecs.append(doc.vector)
    return np.array(vecs, dtype=np.float32)


def embed_texts_tfidf(texts: List[str]) -> np.ndarray:
    """Fallback: TF-IDF + truncated SVD to produce dense 100-dim vectors."""
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.decomposition import TruncatedSVD  # type: ignore

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X = vectorizer.fit_transform(texts)
    n_components = min(100, X.shape[1] - 1, X.shape[0] - 1)
    if n_components < 2:
        # Not enough texts — just use raw TF-IDF
        return X.toarray().astype(np.float32)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    return svd.fit_transform(X).astype(np.float32)


def get_embeddings(texts: List[str]) -> np.ndarray:
    nlp = load_spacy()
    if nlp is not None:
        return embed_texts_spacy(nlp, texts)
    return embed_texts_tfidf(texts)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_papers(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Return cluster label array (int) for each embedding."""
    from sklearn.cluster import KMeans  # type: ignore

    k = min(k, len(embeddings))
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(embeddings)
    return labels


# ---------------------------------------------------------------------------
# Divergence / mixed outcome signal detection (heuristic)
# ---------------------------------------------------------------------------

# Directional academic outcome patterns — precise multi-word phrases and markers
# to prevent shallow single-word false positives (e.g. matching "limiting" in "limiting calories")
_AFFIRMATIVE_PATTERNS = [
    r"\boutperform(?:s|ed|ing)?\b",
    r"\benhance(?:s|d|ment)?\s+(?:cognitive|executive|memory|performance|health|function)\b",
    r"\b(?:novel\s+)?evidence\s+in\s+support\s+of\b",
    r"\bsignificant(?:ly)?\s+improv(?:e|ed|ement|ing)\b",
    r"\bdemonstrated\s+advantage\b",
    r"\bpositive\s+association\b",
    r"\btherapeutic\s+efficacy\b",
    r"\bclinically\s+(?:effective|meaningful)\b",
    r"\bconfirms?\s+(?:an?\s+)?advantage\b",
]

_NULL_OR_CONTRARY_PATTERNS = [
    r"\bno\s+(?:general\s+)?(?:cognitive|executive)?\s*advantages?\b",
    r"\baffords\s+no\b",
    r"\bfailed\s+to\s+replicate\b",
    r"\bfailure\s+to\s+replicate\b",
    r"\bdifficult\s+to\s+pin\s+down\b",
    r"\bno\s+significant\s+(?:difference|effect|advantage|improvement)\b",
    r"\black\s+of\s+evidence\b",
    r"\bnull\s+(?:effect|findings?)\b",
    r"\bconflicting\s+evidence\b",
    r"\bdoes\s+not\s+support\b",
    r"\bdid\s+not\s+support\b",
    r"\bfails?\s+to\s+find\b",
    r"\bineffective\b",
    r"\bno\s+effect\b",
]


def _score_text(text: str) -> Tuple[List[str], List[str]]:
    """Return matching (positive_cues, null_or_contrary_cues) keyword lists."""
    if not text:
        return [], []
    pos_matches = []
    for pat in _AFFIRMATIVE_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            match_str = m.group(0).lower().strip()
            if match_str not in pos_matches:
                pos_matches.append(match_str)

    neg_matches = []
    for pat in _NULL_OR_CONTRARY_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            match_str = m.group(0).lower().strip()
            if match_str not in neg_matches:
                neg_matches.append(match_str)

    return pos_matches, neg_matches


def detect_divergence(texts: List[str]) -> Tuple[bool, List[Tuple[List[str], List[str]]]]:
    """
    Return (has_divergence, per_paper_cues).
    A cluster is flagged for human review if it contains papers with both
    affirmative and null/contrary outcome signals on the topic.
    """
    per_paper = [_score_text(t) for t in texts]
    has_pos = any(len(pos) > 0 for pos, _ in per_paper)
    has_neg = any(len(neg) > 0 for _, neg in per_paper)
    return (has_pos and has_neg), per_paper


def extract_cluster_topics(all_texts: List[str], cluster_indices: List[int]) -> str:
    """Extract top 2 distinctive key phrases representing the cluster."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2000)
        X = vec.fit_transform(all_texts)
        features = np.array(vec.get_feature_names_out())
        mean_tfidf = np.asarray(X[cluster_indices].mean(axis=0)).flatten()
        top_idx = mean_tfidf.argsort()[::-1][:2]
        keywords = [features[i].title() for i in top_idx if mean_tfidf[i] > 0]
        return " & ".join(keywords) if keywords else "General Topic"
    except Exception:
        return "General Topic"


# ---------------------------------------------------------------------------
# Main query flow
# ---------------------------------------------------------------------------

def run_research(query: str, api_key: str, num_results: int, num_clusters: int, use_cache: bool):
    from langchain_serpapi_provider import SerpApiRetriever
    from langchain_serpapi_provider.cache import SerpApiCache

    # Strip trailing punctuation/question marks which restrict Google Scholar exact matching
    search_query = query.strip().rstrip("?").strip()

    cache = SerpApiCache(enabled=use_cache)
    retriever = SerpApiRetriever(
        api_key=api_key,
        engine="google_scholar",
        num_results=num_results,
        cache=cache,
    )

    with st.spinner("🔍 Fetching papers from Google Scholar…"):
        docs = retriever.invoke(search_query)

    if not docs:
        st.error("No results returned. Try a different query or check your API key.")
        return

    # Stats row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<div class="stat-box"><div class="stat-value">{len(docs)}</div>'
            f'<div class="stat-label">Papers Retrieved</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        total_citations = sum(d.metadata.get("citation_count", 0) or 0 for d in docs)
        st.markdown(
            f'<div class="stat-box"><div class="stat-value">{total_citations:,}</div>'
            f'<div class="stat-label">Total Citations</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="stat-box"><div class="stat-value">{num_clusters}</div>'
            f'<div class="stat-label">Topic Clusters</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Embed
    with st.spinner("🧠 Computing embeddings…"):
        texts = [
            f"{d.metadata.get('title', '')} {d.page_content}"
            for d in docs
        ]
        embeddings = get_embeddings(texts)

    # Cluster
    with st.spinner("📊 Clustering by topic similarity…"):
        labels = cluster_papers(embeddings, k=num_clusters)

    # Group docs and text indices by cluster
    clusters: Dict[int, List] = defaultdict(list)
    cluster_doc_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, (doc, label) in enumerate(zip(docs, labels)):
        clusters[int(label)].append(doc)
        cluster_doc_indices[int(label)].append(idx)

    # Render clusters
    CLUSTER_ICONS = ["🔵", "🟣", "🟢", "🟡", "🟠", "🔴"]
    for cluster_idx, (label, papers) in enumerate(sorted(clusters.items())):
        icon = CLUSTER_ICONS[cluster_idx % len(CLUSTER_ICONS)]
        paper_texts = [f"{p.metadata.get('title', '')} {p.page_content}" for p in papers]
        divergent, paper_cues = detect_divergence(paper_texts)
        cluster_topic = extract_cluster_topics(texts, cluster_doc_indices[label])

        # Build cluster HTML
        divergence_html = ""
        if divergent:
            divergence_html = (
                '<div class="divergence-badge">'
                '⚠️ <span><strong>Mixed findings signal — human review recommended.</strong> '
                "Papers in this cluster contain contrasting outcome cues (e.g. studies reporting advantages/enhancements alongside population or replication studies finding null effects or limitations). "
                "This heuristic flags active scientific disputes for primary paper inspection."
                "</span></div>"
            )

        papers_html = ""
        for paper, (pos_cues, neg_cues) in zip(papers, paper_cues):
            meta = paper.metadata
            title = meta.get("title", "(untitled)")
            url = meta.get("url", "")
            snippet = paper.page_content or ""
            year = meta.get("publication_year")
            cites = meta.get("citation_count")
            authors = meta.get("authors")
            pub_info = meta.get("publication_info", "")

            url_html = f'<div class="paper-url"><a href="{url}" target="_blank">↗ View paper</a></div>' if url else ""
            meta_parts = []
            if authors:
                meta_parts.append(f"👤 {authors}")
            if year:
                meta_parts.append(f"📅 Indexed: {year}")
            if cites is not None:
                meta_parts.append(f"📚 {cites:,} citations")
            if pub_info:
                meta_parts.append(f"📖 {pub_info}")
            meta_html = "".join(f"<span>{p}</span>" for p in meta_parts)

            # Defensive cue tags
            cues_html = ""
            if divergent:
                cue_items = []
                if pos_cues:
                    cue_items.append(f'<span style="background: #1b4332; color: #74c69d; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;">Affirmative cue: {", ".join(pos_cues[:2])}</span>')
                if neg_cues:
                    cue_items.append(f'<span style="background: #4a154b; color: #e0aaff; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;">Cautionary / Null cue: {", ".join(neg_cues[:2])}</span>')
                if cue_items:
                    cues_html = f'<div style="margin-top: 6px; display: flex; gap: 8px;">{"".join(cue_items)}</div>'

            papers_html += (
                f'<div class="paper-card">'
                f'  <div class="paper-title">{title}</div>'
                f'  <div class="paper-snippet">{snippet[:300]}{"…" if len(snippet) > 300 else ""}</div>'
                f'  <div class="paper-meta">{meta_html}</div>'
                f"  {cues_html}"
                f"  {url_html}"
                f"</div>"
            )

        st.markdown(
            f'<div class="cluster-card">'
            f'  <div class="cluster-title">{icon} Cluster {cluster_idx + 1}: {cluster_topic} &mdash; {len(papers)} paper{"s" if len(papers) != 1 else ""}</div>'
            f"  {divergence_html}"
            f"  {papers_html}"
            f"</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Query input & quick examples
# ---------------------------------------------------------------------------

st.markdown("<p style='color: #8b949e; font-size: 0.85rem; margin-bottom: 6px;'><strong>Try a sample research question:</strong></p>", unsafe_allow_html=True)
sample_cols = st.columns(3)
sample_queries = [
    ("Bilingual Advantage", "bilingual advantage executive function cognitive control"),
    ("Intermittent Fasting", "intermittent fasting insulin sensitivity metabolic health"),
    ("LLM Theory of Mind", "large language models theory of mind reasoning"),
]

if "current_query" not in st.session_state:
    st.session_state.current_query = sample_queries[0][1]

for col, (label, sq) in zip(sample_cols, sample_queries):
    if col.button(f"💡 {label}", help=sq, use_container_width=True):
        st.session_state.current_query = sq
        st.rerun()

query = st.text_area(
    "Research question",
    value=st.session_state.current_query,
    height=80,
    label_visibility="collapsed",
)

if st.button("🔬 Run Research Agent", use_container_width=True):
    if not api_key:
        st.error("Please enter your SerpApi API key in the sidebar.")
    elif not query.strip():
        st.warning("Please enter a research question.")
    else:
        run_research(
            query=query.strip(),
            api_key=api_key,
            num_results=num_results,
            num_clusters=num_clusters,
            use_cache=use_cache,
        )
