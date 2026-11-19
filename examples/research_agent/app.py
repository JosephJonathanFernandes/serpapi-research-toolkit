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
    SPLIT_BADGE_BODY,
    SPLIT_BADGE_TITLE,
    STANCE_DISPLAY,
    STANCES,
    make_nli_stance_classifier,
    make_phrase_stance_classifier,
    make_stance_classifier,
    score_text,
    stance_split,
    summarize_stances,
)
from export import docs_to_rows, to_bibtex, to_csv  # noqa: E402
from sample_queries import SAMPLE_CLAIMS, SAMPLE_QUERIES, normalize_query  # noqa: E402
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

/* Claim & Stance UI */
.claim-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    margin-top: 1rem;
}
.claim-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #8b949e; margin-bottom: 0.25rem; }
.claim-text { font-size: 1.05rem; font-weight: 600; color: #e6edf3; margin-bottom: 0.75rem; }
.claim-note { font-size: 0.78rem; color: #8b949e; margin-top: 0.5rem; }
.stance-bar {
    display: flex;
    height: 10px;
    border-radius: 6px;
    overflow: hidden;
    background: #21262d;
    margin-bottom: 0.45rem;
}
.seg-supports { background: #2ea043; }
.seg-null { background: #da3633; }
.seg-mixed { background: #d29922; }
.seg-unclear { background: #484f58; }
.stance-legend { font-size: 0.82rem; color: #c9d1d9; margin-bottom: 0.6rem; }
.stance-groups { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.75rem; margin: 0.75rem 0; }
.stance-group { background: #0d1117; border: 1px solid #21262d; border-radius: 8px; padding: 0.65rem 0.9rem; font-size: 0.82rem; }
.stance-group ul { margin: 0.35rem 0 0; padding-left: 1.1rem; color: #c9d1d9; }
.stance-group li { margin-bottom: 0.25rem; }
.stance-group .cites { color: #8b949e; font-size: 0.76rem; }
.sg-title { font-weight: 600; color: #e6edf3; }
.stance-pill {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 0.4rem;
    vertical-align: middle;
}
.pill-supports { background: #1b4332; color: #74c69d; border: 1px solid #2d6a4f; }
.pill-null { background: #491c1c; color: #ffa198; border: 1px solid #8e2a2a; }
.pill-mixed { background: #3d2e00; color: #f2cc60; border: 1px solid #6e5400; }
.pill-unclear { background: #21262d; color: #8b949e; border: 1px solid #30363d; }
.enriched-pill {
    background: rgba(31, 111, 235, 0.2);
    color: #58a6ff;
    border: 1px solid rgba(56, 139, 253, 0.4);
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.72rem;
    vertical-align: middle;
}
details summary { cursor: pointer; color: #58a6ff; font-size: 0.85rem; margin: 0.5rem 0; }

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
  <p>Enter a research question and a hypothesis — the agent retrieves academic papers via SerpApi,
  clusters them by topic similarity, and summarizes <strong>stance counts per cluster</strong>
  (supports, null/contrary, mixed, unclear) with top-cited papers under each stance.</p>
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
        "Use full abstracts (OpenAlex / arXiv)",
        value=False,
        help="Replaces short snippets with full abstracts (free, no key). The stance classifiers were "
        "validated on full abstracts (SciFact); Scholar snippets often cut off the findings sentences. "
        "Slower: one lookup per paper.",
    )

    backend = st.selectbox(
        "Stance classifier",
        ["phrases", "nli", "llm"],
        format_func={
            "phrases": "Tier 1: Phrase cues (instant, free, weak)",
            "nli": "Tier 2: NLI cross-encoder (CPU, recommended)",
            "llm": "Tier 3: LLM (OpenAI key or local Ollama)",
        }.get,
        help="Each paper is classified as supporting the claim, null/contrary, mixed, or unclear. "
        "Tier 1 ignores the claim and only looks at outcome wording. Tier 2 checks whether the "
        "paper's findings entail or contradict the claim. Measured accuracy on SciFact (human "
        "labels) is in eval/scifact/results.md.",
    )

    st.divider()
    st.markdown("### 📖 About")
    st.caption(
        "Powered by **langchain-serpapi-provider** + **OpenAlex / arXiv** + **spaCy / TF-IDF** + **KMeans**.\n\n"
        "• **Indexed Year:** Reflects the canonical edition/year indexed by Google Scholar.\n\n"
        "• **Stance counts:** automatic labels of what each paper's findings say about your claim. "
        "Papers can differ in cohort, dose or outcome. Always check the papers."
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
def load_classifier(backend_choice: str):
    if backend_choice == "nli":
        return make_nli_stance_classifier()
    if backend_choice == "llm":
        return make_stance_classifier()
    return make_phrase_stance_classifier()


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


def analyze(docs: list, claim: str) -> Dict[str, Any]:
    texts = [f"{d.metadata.get('title', '')} {d.page_content}" for d in docs]
    embeddings = get_embeddings(texts)
    if auto_k:
        k, scores = choose_k(embeddings)
    else:
        k, scores = min(num_clusters, len(docs)), {}
    labels = cluster_papers(embeddings, k)

    classify = load_classifier(backend)
    stances = [classify(claim, t) for t in texts]
    cues = [score_text(t) for t in texts]

    groups: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        groups[int(label)].append(idx)

    clusters = []
    for label, idxs in sorted(groups.items()):
        counts = summarize_stances([stances[i] for i in idxs])
        clusters.append(
            {
                "indices": idxs,
                "topic": extract_cluster_topics(texts, idxs),
                "counts": counts,
                "split": stance_split(counts),
            }
        )
    return {
        "docs": docs,
        "clusters": clusters,
        "k": k,
        "silhouette": scores,
        "demo": demo_mode,
        "claim": claim,
        "stances": stances,
        "cues": cues,
        "backend": backend,
    }


def run_research(query: str, claim: str) -> None:
    try:
        with st.spinner("🔍 Loading papers…" if demo_mode else "🔍 Fetching papers from Google Scholar…"):
            docs = fetch_docs(query)
        if not docs:
            st.error("No results returned. Try a different query or check your API key.")
            st.session_state.pop("result", None)
            return
        with st.spinner("🧠 Clustering and classifying stance…"):
            st.session_state["result"] = analyze(docs, claim)
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


def stance_bar(counts: Dict[str, int]) -> str:
    """Proportional coloured bar + legend for one set of stance counts."""
    total = sum(counts.values()) or 1
    segs = "".join(
        f'<div class="seg seg-{s.lower()}" style="width:{100 * counts[s] / total:.1f}%" '
        f'title="{counts[s]} {STANCE_DISPLAY[s][1]}"></div>'
        for s in STANCES
        if counts[s]
    )
    legend = " &nbsp;·&nbsp; ".join(
        f"{STANCE_DISPLAY[s][0]} <strong>{counts[s]}</strong> {STANCE_DISPLAY[s][1]}" for s in STANCES if counts[s]
    )
    return f'<div class="stance-bar">{segs}</div><div class="stance-legend">{legend}</div>'


def stance_pill(stance: str) -> str:
    icon, text = STANCE_DISPLAY.get(stance, STANCE_DISPLAY["UNCLEAR"])
    return f'<span class="stance-pill pill-{stance.lower()}">{icon} {esc(text)}</span>'


def render_result(result: Dict[str, Any]) -> None:
    docs = result["docs"]
    clusters = result["clusters"]
    stances = result["stances"]

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

    # Overall stance summary toward the claim
    overall = summarize_stances(stances)
    tier = {"phrases": "Tier 1 phrase cues", "nli": "Tier 2 NLI", "llm": "Tier 3 LLM"}.get(result["backend"], result["backend"])
    st.markdown(
        f'<div class="claim-card"><div class="claim-label">Claim checked ({esc(tier)})</div>'
        f'<div class="claim-text">“{esc(result["claim"])}”</div>{stance_bar(overall)}'
        f'<div class="claim-note">Automatic labels of what each paper\'s text says about this claim. '
        f"They can be wrong; open the papers before concluding anything.</div></div>",
        unsafe_allow_html=True,
    )

    cluster_names: List[str] = [""] * len(docs)
    for c_no, cl in enumerate(clusters, 1):
        for idx in cl["indices"]:
            cluster_names[idx] = f"Cluster {c_no}: {cl['topic']}"

    st.write("")
    dl1, dl2, _ = st.columns([1, 1, 2])
    dl1.download_button(
        "⬇️ Export CSV",
        data=to_csv(docs_to_rows(docs, cluster_names, result["cues"], stances)),
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
        idxs = cl["indices"]
        counts = cl["counts"]

        badge_html = ""
        if cl["split"]:
            badge_html = (
                '<div class="divergence-badge">⚠️ <span>'
                f"<strong>{esc(SPLIT_BADGE_TITLE)}</strong> {esc(SPLIT_BADGE_BODY)}"
                "</span></div>"
            )

        # Most-cited papers under each stance (supports / null first: that's the comparison)
        groups_html = ""
        for s in ("SUPPORTS", "NULL", "MIXED"):
            members = [docs[i] for i in idxs if stances[i] == s]
            if not members:
                continue
            emoji, text = STANCE_DISPLAY[s]
            items = "".join(
                f"<li>{esc(p.metadata.get('title', '(untitled)'))}"
                + (f" <span class='cites'>({p.metadata['citation_count']:,} cites)</span>" if p.metadata.get("citation_count") is not None else "")
                + "</li>"
                for p in top_papers(members, 3)
            )
            groups_html += f'<div class="stance-group"><div class="sg-title">{emoji} {esc(text)} ({len(members)})</div><ul>{items}</ul></div>'
        if not groups_html:
            groups_html = '<div class="top-papers">No paper in this cluster states findings about the claim (all unclear).</div>'

        papers_html = ""
        by_id = {id(docs[i]): i for i in idxs}
        for paper in sort_papers([docs[i] for i in idxs], sort_by):
            i = by_id[id(paper)]
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

            is_enriched = meta.get("is_enriched", False)
            title_pill = ' <span class="enriched-pill">Full abstract</span>' if is_enriched else ""
            snippet_label = "Abstract" if is_enriched else "Snippet"
            shown = snippet[:400] + ("…" if len(snippet) > 400 else "")
            papers_html += (
                f'<div class="paper-card">'
                f'  <div class="paper-title">{stance_pill(stances[i])} {esc(title)}{title_pill}</div>'
                f'  <div class="paper-snippet"><strong>{snippet_label}:</strong> {esc(shown)}</div>'
                f'  <div class="paper-meta">{meta_html}</div>'
                f"  {url_html}"
                f"</div>"
            )

        n = len(idxs)
        st.markdown(
            f'<div class="cluster-card">'
            f'  <div class="cluster-title">{icon} Cluster {c_no + 1}: {esc(cl["topic"])} &mdash; {n} paper{"s" if n != 1 else ""}</div>'
            f"  {stance_bar(counts)}"
            f"  {badge_html}"
            f'  <div class="stance-groups">{groups_html}</div>'
            f'  <details><summary>All {n} papers in this cluster</summary>{papers_html}</details>'
            f"</div>",
            unsafe_allow_html=True,
        )

    with st.expander("⚠️ Limitations & Scientific Framing"):
        st.markdown(
            "- **Stance toward a shared claim:** Instead of only predicting pairwise contradictions, each paper is "
            "classified for its stance toward your stated hypothesis (`supports`, `null / contrary`, `mixed`, `unclear`) "
            "and aggregated per topic cluster.\n"
            "- **External Validation (SciFact):** On the human-labeled SciFact dev benchmark (340 claim–abstract pairs), "
            "Tier 2 NLI stance classification (`cross-encoder/nli-deberta-v3-small`) reaches **0.51 CONTRADICT F1** "
            "(precision 0.56, recall 0.46) and **0.42 Macro-F1**, compared to **0.03–0.07 CONTRADICT F1** for claim-agnostic "
            "phrase cues; see `eval/scifact/results.md`.\n"
            "- **Pairwise Benchmark Caveats:** On our internal 84-pair test split (AI-drafted, unaudited labels; "
            "`eval/audit/` provides blind human-audit sheets), the dev-only selected phrase lexicon V3 reaches "
            "**0.18 F1** (precision 0.75, recall 0.10), while lexicon V2 (tuned with exposure to the same topics) "
            "reaches 0.40 F1 (an optimistic upper bound).\n"
            "- **Split Findings ≠ Proven Conflict:** Two papers can report opposing stances because they examine "
            "different cohorts, dosages, or endpoints. Always inspect the full papers.\n"
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
if "current_claim" not in st.session_state:
    st.session_state.current_claim = SAMPLE_CLAIMS.get(normalize_query(SAMPLE_QUERIES[0][1]), SAMPLE_QUERIES[0][1])

for col, (label, sq) in zip(sample_cols, SAMPLE_QUERIES):
    if col.button(f"💡 {label}", help=sq, use_container_width=True):
        st.session_state.current_query = sq
        st.session_state.current_claim = SAMPLE_CLAIMS.get(normalize_query(sq), sq)
        st.rerun()

query = st.text_area(
    "Research question (Google Scholar query)",
    value=st.session_state.current_query,
    height=68,
    help="Search terms sent to Google Scholar via SerpApi.",
)

default_claim = SAMPLE_CLAIMS.get(normalize_query(query), st.session_state.current_claim or query)
claim = st.text_input(
    "Hypothesis / claim to test against retrieved papers",
    value=default_claim,
    help="Tier 2 (NLI) and Tier 3 (LLM) classify whether each paper's findings support or report null/contrary evidence for this claim.",
)

if st.button("🔬 Run Research Agent", use_container_width=True):
    if not demo_mode and not api_key:
        st.error("Please enter your SerpApi API key in the sidebar, or tick Demo mode.")
    elif not query.strip():
        st.warning("Please enter a research question.")
    else:
        run_research(query.strip(), (claim.strip() or query.strip()))

# Results live in session state so sorting / downloads don't wipe them on rerun.
if "result" in st.session_state:
    render_result(st.session_state["result"])
