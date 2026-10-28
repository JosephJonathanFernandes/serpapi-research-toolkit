"""
Research Agent — Streamlit Demo
================================
Demonstrates the langchain_serpapi_provider plugin.

Features
--------
- Enter a research question → fetch top papers via SerpApi (Scholar engine),
  or replay recorded results in **demo mode** (no API key needed)
- Embed each paper's title + snippet using spaCy or TF-IDF vectors
- Cluster papers by topic similarity (KMeans; k chosen by silhouette score)
- Within each cluster, flag a *possible disagreement* between snippets —
  a weak, snippet-based signal labeled conservatively for human review
- Show each cluster's top papers next to its auto-generated label
- Sort by citations / year, export CSV and BibTeX

Run
---
    cd examples/research_agent
    streamlit run app.py
"""

from __future__ import annotations

import html
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
try:  # allow running straight from a clone / Streamlit Cloud without `pip install -e .`
    import langchain_serpapi_provider  # noqa: F401
except ImportError:
    sys.path.insert(0, str(_HERE.parents[1]))

from analysis import (  # noqa: E402
    choose_k,
    cluster_papers,
    embed_texts_spacy,
    embed_texts_tfidf,
    extract_cluster_topics,
    sort_papers,
    top_papers,
)
from demo_data import demo_queries_available, load_fixture_docs  # noqa: E402
from disagreement import (  # noqa: E402
    BADGE_BODY,
    BADGE_TITLE,
    detect_possible_disagreement,
    make_nli_judge,
)
from export import docs_to_rows, to_bibtex, to_csv  # noqa: E402
from sample_queries import SAMPLE_QUERIES, normalize_query  # noqa: E402
from enrichment import enrich_documents  # noqa: E402

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
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.top-papers {
    font-size: 0.8rem;
    color: #8b949e;
    margin-bottom: 1rem;
    line-height: 1.5;
    border-left: 2px solid #30363d;
    padding-left: 0.75rem;
}
.top-papers strong { color: #c9d1d9; }

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

/* Disagreement badge */
.divergence-badge {
    background: linear-gradient(90deg, #3d1a00, #5a2800);
    border: 1px solid #f0883e55;
    border-radius: 6px;
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.9rem;
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

/* Mode pill */
.mode-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
}
.mode-demo { background: #1b4332; color: #74c69d; border: 1px solid #2d6a4f; }
.mode-live { background: #1f2a44; color: #79c0ff; border: 1px solid #1f6feb55; }

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
  clusters them by topic similarity, and flags <strong>possible disagreement</strong>
  between snippets for human review.</p>
</div>
""",
    unsafe_allow_html=True,
)


def esc(value: Any) -> str:
    """HTML-escape any value that will be injected into unsafe_allow_html markup."""
    return html.escape(str(value), quote=True)


def safe_url(url: str) -> str:
    """Only allow http(s) links."""
    return url if urlparse(url or "").scheme in ("http", "https") else ""


# ---------------------------------------------------------------------------
# Sidebar config
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    env_key = os.getenv("SERPAPI_API_KEY", "")
    if env_key and env_key == "your_serpapi_api_key_here":
        env_key = ""
    api_key = env_key

    demo_available = demo_queries_available()
    demo_mode = st.checkbox(
        "Demo mode (recorded results, no API key)",
        value=not bool(api_key),
        help="Replays real SerpApi responses recorded earlier for the three sample "
        "questions. Uses no API credits and works offline.",
    )

    if demo_mode:
        st.info("Demo mode: only the three sample questions are available.", icon="🧪")
    elif api_key:
        st.success("🔑 API key loaded from environment", icon="✅")
    else:
        api_key = st.text_input(
            "SerpApi API Key",
            value="",
            type="password",
            help="Get yours at serpapi.com",
        )

    num_results = st.slider("Papers to retrieve", min_value=5, max_value=20, value=10, disabled=demo_mode)

    auto_k = st.checkbox(
        "Choose cluster count automatically (silhouette)",
        value=True,
        help="Tries k = 2…6 and keeps the best silhouette score. With 10–20 papers "
        "this is a noisy guide; untick to set k yourself.",
    )
    num_clusters = st.slider("Topic clusters (k)", min_value=2, max_value=6, value=3, disabled=auto_k)
    sort_by = st.selectbox(
        "Sort papers within a cluster",
        ["relevance", "citations", "year"],
        format_func={"relevance": "Search relevance", "citations": "Citations (high → low)", "year": "Indexed year (new → old)"}.get,
    )
    use_cache = st.checkbox("Enable disk cache (saves API credits)", value=True, disabled=demo_mode)

    enrich_abstracts = st.checkbox(
        "Show full abstracts (OpenAlex / arXiv)",
        value=False,
        help="Replaces short snippets with full abstracts (free, no key). Off by default: in our "
        "ablation, full abstracts lowered disagreement-detection scores for every detector tested. "
        "Useful for reading, not (yet) for detection.",
    )

    backend = st.selectbox(
        "Disagreement detector",
        ["phrases", "nli", "llm"],
        format_func={
            "phrases": "Tier 1: Phrase matching (fast, free)",
            "nli": "Tier 2: Cross-Encoder NLI (DeBERTa, CPU)",
            "llm": "Tier 3: LLM Judge (OpenAI / LangChain)",
        }.get,
        help="Tier 1 uses directional phrase patterns. Tier 2 uses a local DeBERTa cross-encoder on CPU. Tier 3 uses an LLM judge.",
    )

    st.divider()
    st.markdown("### 📖 About")
    st.caption(
        "Powered by **langchain-serpapi-provider** + **OpenAlex / arXiv** + **spaCy / TF-IDF** + **KMeans**.\n\n"
        "• **Indexed Year:** Reflects the canonical edition/year indexed by Google Scholar.\n\n"
        "• **Possible disagreement:** A weak signal comparing empirical outcomes. "
        "Differences may reflect distinct cohorts or dosing rather than direct contradiction. Always check the papers."
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
def load_judge(backend_choice: str):
    if backend_choice == "nli":
        return make_nli_judge()
    if backend_choice == "llm":
        from disagreement import make_llm_judge
        return make_llm_judge()
    return None


def get_embeddings(texts: List[str]):
    nlp = load_spacy()
    if nlp is not None:
        return embed_texts_spacy(nlp, texts)
    return embed_texts_tfidf(texts)


# ---------------------------------------------------------------------------
# Query flow: fetch → analyze (stored in session state) → render
# ---------------------------------------------------------------------------

def fetch_docs(query: str) -> list:
    search_query = normalize_query(query)  # trailing "?" breaks Scholar exact matching
    if demo_mode:
        docs = load_fixture_docs(search_query, num_results=20)
    else:
        from langchain_serpapi_provider import SerpApiCache, SerpApiRetriever

        retriever = SerpApiRetriever(
            api_key=api_key,
            engine="google_scholar",
            num_results=num_results,
            cache=SerpApiCache(enabled=True) if use_cache else None,
        )
        docs = retriever.invoke(search_query)

    if enrich_abstracts and docs:
        docs = enrich_documents(docs)
    return docs


def analyze(docs: list) -> Dict[str, Any]:
    texts = [f"{d.metadata.get('title', '')} {d.page_content}" for d in docs]
    embeddings = get_embeddings(texts)
    if auto_k:
        k, scores = choose_k(embeddings)
    else:
        k, scores = min(num_clusters, len(docs)), {}
    labels = cluster_papers(embeddings, k)

    judge = load_judge(backend) if backend in ("nli", "llm") else None
    groups: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        groups[int(label)].append(idx)

    clusters = []
    for label, idxs in sorted(groups.items()):
        result = detect_possible_disagreement(
            [texts[i] for i in idxs], backend=backend, judge=judge
        )
        clusters.append(
            {
                "indices": idxs,
                "topic": extract_cluster_topics(texts, idxs),
                "disagreement": result,
            }
        )
    return {"docs": docs, "clusters": clusters, "k": k, "silhouette": scores, "demo": demo_mode}


def run_research(query: str) -> None:
    try:
        with st.spinner("🔍 Loading papers…" if demo_mode else "🔍 Fetching papers from Google Scholar…"):
            docs = fetch_docs(query)
        if not docs:
            st.error("No results returned. Try a different query or check your API key.")
            st.session_state.pop("result", None)
            return
        with st.spinner("🧠 Embedding and clustering…"):
            st.session_state["result"] = analyze(docs)
        st.session_state["result_query"] = query
    except FileNotFoundError as exc:
        st.session_state.pop("result", None)
        st.error(str(exc))
    except ImportError as exc:
        st.session_state.pop("result", None)
        st.error(f"Missing optional dependency: {exc}")
    except Exception as exc:  # SerpApiError subclasses and anything unexpected
        st.session_state.pop("result", None)
        st.error(f"{type(exc).__name__}: {exc}")


def stat_box(value: Any, label: str) -> str:
    return (
        f'<div class="stat-box"><div class="stat-value">{esc(value)}</div>'
        f'<div class="stat-label">{esc(label)}</div></div>'
    )


def render_result(result: Dict[str, Any]) -> None:
    docs = result["docs"]
    clusters = result["clusters"]

    pill = (
        '<span class="mode-pill mode-demo">DEMO MODE — recorded SerpApi results</span>'
        if result["demo"]
        else '<span class="mode-pill mode-live">LIVE — SerpApi Google Scholar</span>'
    )
    st.markdown(pill, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    total_citations = sum(d.metadata.get("citation_count", 0) or 0 for d in docs)
    sil = result["silhouette"].get(result["k"])
    with col1:
        st.markdown(stat_box(len(docs), "Papers Retrieved"), unsafe_allow_html=True)
    with col2:
        st.markdown(stat_box(f"{total_citations:,}", "Total Citations"), unsafe_allow_html=True)
    with col3:
        label = f"Topic Clusters (silhouette {sil:.2f})" if sil is not None else "Topic Clusters"
        st.markdown(stat_box(len(clusters), label), unsafe_allow_html=True)

    # Cue lookup + export rows
    cluster_names: List[str] = [""] * len(docs)
    cues_by_doc: Dict[int, Any] = {}
    for c_no, cl in enumerate(clusters, 1):
        name = f"Cluster {c_no}: {cl['topic']}"
        for local, idx in enumerate(cl["indices"]):
            cluster_names[idx] = name
            cues_by_doc[idx] = cl["disagreement"].per_paper_cues[local]
    export_cues = [cues_by_doc.get(i, ([], [])) for i in range(len(docs))]

    st.write("")
    dl1, dl2, _ = st.columns([1, 1, 2])
    dl1.download_button(
        "⬇️ Export CSV",
        data=to_csv(docs_to_rows(docs, cluster_names, export_cues)),
        file_name="papers.csv",
        mime="text/csv",
        use_container_width=True,
    )
    dl2.download_button(
        "⬇️ Export BibTeX",
        data=to_bibtex(docs),
        file_name="papers.bib",
        mime="text/plain",
        use_container_width=True,
        help="Entries are @misc built from Scholar snippets; verify metadata before citing.",
    )
    st.markdown("---")

    CLUSTER_ICONS = ["🔵", "🟣", "🟢", "🟡", "🟠", "🔴"]
    for c_no, cl in enumerate(clusters):
        icon = CLUSTER_ICONS[c_no % len(CLUSTER_ICONS)]
        flagged = cl["disagreement"].flagged
        cluster_docs = [docs[i] for i in cl["indices"]]
        cue_map = {id(docs[i]): cl["disagreement"].per_paper_cues[n] for n, i in enumerate(cl["indices"])}

        badge_html = ""
        if flagged:
            badge_html = (
                '<div class="divergence-badge">⚠️ <span>'
                f"<strong>{esc(BADGE_TITLE)}</strong> {esc(BADGE_BODY)}"
                "</span></div>"
            )

        top = top_papers(cluster_docs, 3)
        top_html = "<br>".join(
            f"<strong>{esc(p.metadata.get('title', '(untitled)'))}</strong>"
            + (f" ({p.metadata['citation_count']:,} cites)" if p.metadata.get("citation_count") is not None else "")
            for p in top
        )

        papers_html = ""
        for paper in sort_papers(cluster_docs, sort_by):
            pos_cues, neg_cues = cue_map[id(paper)]
            meta = paper.metadata
            title = meta.get("title", "(untitled)")
            url = safe_url(meta.get("url", ""))
            snippet = paper.page_content or ""
            year = meta.get("publication_year")
            cites = meta.get("citation_count")
            authors = meta.get("authors")
            pub_info = meta.get("publication_info", "")

            url_html = f'<div class="paper-url"><a href="{esc(url)}" target="_blank" rel="noopener noreferrer">↗ View paper</a></div>' if url else ""
            meta_parts = []
            if authors:
                meta_parts.append(f"👤 {esc(authors)}")
            if year:
                meta_parts.append(f"📅 Indexed: {esc(year)}")
            if cites is not None:
                meta_parts.append(f"📚 {cites:,} citations")
            if pub_info:
                meta_parts.append(f"📖 {esc(pub_info)}")
            meta_html = "".join(f"<span>{p}</span>" for p in meta_parts)

            cues_html = ""
            if flagged:
                cue_items = []
                if pos_cues:
                    cue_items.append(f'<span style="background: #1b4332; color: #74c69d; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;">Affirmative cue: {esc(", ".join(pos_cues[:2]))}</span>')
                if neg_cues:
                    cue_items.append(f'<span style="background: #4a154b; color: #e0aaff; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;">Cautionary / Null cue: {esc(", ".join(neg_cues[:2]))}</span>')
                if cue_items:
                    cues_html = f'<div style="margin-top: 6px; display: flex; gap: 8px;">{"".join(cue_items)}</div>'

            is_enriched = meta.get("is_enriched", False)
            title_pill = (
                ' <span style="background: rgba(31, 111, 235, 0.2); color: #58a6ff; '
                'border: 1px solid rgba(56, 139, 253, 0.4); padding: 2px 8px; '
                'border-radius: 12px; font-size: 0.72rem; vertical-align: middle;">Enriched Abstract</span>'
                if is_enriched else ""
            )
            snippet_label = "Abstract" if is_enriched else "Snippet"
            shown = snippet[:400] + ("…" if len(snippet) > 400 else "")
            papers_html += (
                f'<div class="paper-card">'
                f'  <div class="paper-title">{esc(title)}{title_pill}</div>'
                f'  <div class="paper-snippet"><strong>{snippet_label}:</strong> {esc(shown)}</div>'
                f'  <div class="paper-meta">{meta_html}</div>'
                f"  {cues_html}"
                f"  {url_html}"
                f"</div>"
            )

        n = len(cluster_docs)
        st.markdown(
            f'<div class="cluster-card">'
            f'  <div class="cluster-title">{icon} Cluster {c_no + 1}: {esc(cl["topic"])} &mdash; {n} paper{"s" if n != 1 else ""}</div>'
            f'  <div class="top-papers">Most-cited in this cluster (check the label against these):<br>{top_html}</div>'
            f"  {badge_html}"
            f"  {papers_html}"
            f"</div>",
            unsafe_allow_html=True,
        )

    with st.expander("⚠️ Limitations & Scientific Framing"):
        st.markdown(
            "- **Abstract & Snippet Evidence:** Google Scholar snippets are short (100–250 chars) and often cut "
            "mid-sentence. Full abstracts (optional) give more context for reading, but in our ablation they did "
            "not improve, and usually reduced, disagreement-detection scores.\n"
            "- **Weak detector:** the best free detector reached about 0.27 recall on an AI-labeled, unaudited "
            "test set of 84 pairs. It misses most disagreements; see `eval/README.md`.\n"
            "- **Possible Disagreement ≠ Proven Conflict:** Two papers can take opposing directional stances "
            "because they examine different cohorts (e.g. children vs older adults) or different dosages. "
            "The detector flags candidate disagreements for human verification; always read the full papers.\n"
            "- **Detector Tiers:** Tier 1 (Phrase Matching) is fast and free. Tier 2 (Cross-Encoder NLI) evaluates "
            "sentence entailment locally on CPU. Tier 3 uses an LLM judge for meta-analytic synthesis.\n"
            "- **Small-N Clustering:** KMeans on 10–20 papers is noisy; optimal k is selected via silhouette score "
            "with a minimum cluster size constraint.\n"
            "- **SerpApi Credits:** Each live search uses one API credit. Repeat searches are served from disk cache "
            "in < 1 ms."
        )


# ---------------------------------------------------------------------------
# Query input & quick examples
# ---------------------------------------------------------------------------

st.markdown("<p style='color: #8b949e; font-size: 0.85rem; margin-bottom: 6px;'><strong>Try a sample research question:</strong></p>", unsafe_allow_html=True)
sample_cols = st.columns(3)

if "current_query" not in st.session_state:
    st.session_state.current_query = SAMPLE_QUERIES[0][1]

for col, (label, sq) in zip(sample_cols, SAMPLE_QUERIES):
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
    if not demo_mode and not api_key:
        st.error("Please enter your SerpApi API key in the sidebar, or tick Demo mode.")
    elif not query.strip():
        st.warning("Please enter a research question.")
    else:
        run_research(query.strip())

# Results live in session state so sorting / downloads don't wipe them on rerun.
if "result" in st.session_state:
    render_result(st.session_state["result"])
