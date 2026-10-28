"""Tests for the demo app's pure-logic modules (no Streamlit needed)."""

from __future__ import annotations

import numpy as np
import pytest
from langchain_core.documents import Document

pytest.importorskip("sklearn")

from analysis import (  # noqa: E402
    choose_k,
    cluster_papers,
    embed_texts_tfidf,
    extract_cluster_topics,
    l2_normalize,
    sort_papers,
    top_papers,
)
from demo_data import demo_queries_available, load_fixture_docs  # noqa: E402
from disagreement import (  # noqa: E402
    BADGE_TITLE,
    detect_possible_disagreement,
    phrase_pair_disagrees,
    score_text,
)
from export import bibtex_key, to_bibtex, to_csv, docs_to_rows  # noqa: E402
from sample_queries import SAMPLE_QUERIES, fixture_path  # noqa: E402


def _doc(title="T", snippet="s", **meta):
    return Document(page_content=snippet, metadata={"title": title, **meta})


# --------------------------------------------------------------------------
# Disagreement
# --------------------------------------------------------------------------

class TestDisagreement:
    def test_badge_wording_is_hedged(self):
        assert "Possible disagreement" in BADGE_TITLE
        assert "human review" in BADGE_TITLE
        assert "Mixed findings" not in BADGE_TITLE

    def test_cues_extracted(self):
        pos, neg = score_text("Training significantly improved outcomes and outperformed controls")
        assert "outperformed" in pos
        _, neg = score_text("We found no significant effect and failed to replicate")
        assert "no significant effect" in neg and "failed to replicate" in neg

    def test_cluster_flagged_when_two_different_snippets_disagree(self):
        r = detect_possible_disagreement(
            ["Treated group outperformed controls", "Large study with no significant effect"]
        )
        assert r.flagged and r.pairs == [(0, 1)]

    def test_single_snippet_with_both_cues_not_flagged(self):
        r = detect_possible_disagreement(["It outperformed A but showed no significant effect on B"])
        assert not r.flagged

    def test_same_direction_not_flagged(self):
        r = detect_possible_disagreement(["outperformed controls", "significantly improved scores"])
        assert not r.flagged

    def test_pair_helper(self):
        assert phrase_pair_disagrees("outperformed controls", "no effect was seen")
        assert not phrase_pair_disagrees("outperformed controls", "outperforms baselines")

    def test_nli_backend_uses_injected_judge(self):
        judge = lambda a, b: 0.9 if "A" in a and "B" in b else 0.1
        r = detect_possible_disagreement(["A claim", "B claim", "C"], backend="nli", judge=judge)
        assert r.flagged and (0, 1) in r.pairs and r.backend == "nli"

    def test_nli_threshold(self):
        r = detect_possible_disagreement(["x", "y"], backend="nli", judge=lambda a, b: 0.4)
        assert not r.flagged

    def test_unknown_backend(self):
        with pytest.raises(ValueError):
            detect_possible_disagreement(["a", "b"], backend="magic")

    def test_llm_backend(self):
        from disagreement import LLMJudgeVerdict
        judge = lambda a, b: LLMJudgeVerdict(relationship="CONTRADICT", score=0.9, rationale="Opposing claims")
        r = detect_possible_disagreement(["paper 1", "paper 2"], backend="llm", judge=judge)
        assert r.flagged and (0, 1) in r.pairs and r.backend == "llm"


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------

def _blobs(k=3, per=6, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.eye(dim)[:k] * 5
    return np.vstack([c + rng.normal(0, 0.15, (per, dim)) for c in centers])


class TestClustering:
    def test_l2_normalize_unit_rows(self):
        out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
        assert np.allclose(out[0], [0.6, 0.8]) and np.allclose(out[1], [0, 0])

    def test_choose_k_recovers_true_cluster_count(self):
        k, scores = choose_k(_blobs(k=3), k_max=5)
        assert k == 3 and scores[3] == max(scores.values())

    def test_choose_k_tiny_input(self):
        assert choose_k(np.zeros((3, 4))) == (2, {})
        assert choose_k(np.zeros((1, 4))) == (1, {})

    def test_cluster_papers_labels(self):
        labels = cluster_papers(_blobs(k=3), 3)
        assert len(labels) == 18 and len(set(labels)) == 3

    def test_cluster_papers_k_clamped(self):
        assert len(set(cluster_papers(np.random.rand(3, 4), 10))) <= 3

    def test_topic_label(self):
        texts = ["deep learning vision", "deep learning speech", "insulin fasting study", "insulin fasting trial"]
        label = extract_cluster_topics(texts, [2, 3])
        assert "Insulin" in label or "Fasting" in label

    def test_tfidf_embedding_shape(self):
        emb = embed_texts_tfidf(["alpha beta gamma", "beta gamma delta", "epsilon zeta eta", "zeta eta theta"])
        assert emb.shape[0] == 4


class TestSorting:
    DOCS = [
        _doc("a", citation_count=5, publication_year=2010),
        _doc("b", citation_count=50, publication_year=2001),
        _doc("c", citation_count=None, publication_year=2022),
    ]

    def test_relevance_keeps_order(self):
        assert [d.metadata["title"] for d in sort_papers(self.DOCS)] == ["a", "b", "c"]

    def test_citations_desc_none_last(self):
        assert [d.metadata["title"] for d in sort_papers(self.DOCS, "citations")] == ["b", "a", "c"]

    def test_year_desc(self):
        assert [d.metadata["title"] for d in sort_papers(self.DOCS, "year")] == ["c", "a", "b"]

    def test_top_papers(self):
        assert [d.metadata["title"] for d in top_papers(self.DOCS, 2)] == ["b", "a"]


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

class TestExport:
    def test_csv_roundtrip_and_formula_guard(self):
        import csv, io

        docs = [_doc("=HYPERLINK(\"x\")", "snip, with comma", citation_count=3, publication_year=2020,
                     authors="A B", url="https://x")]
        text = to_csv(docs_to_rows(docs, clusters=["C1"], cues=[(["outperformed"], [])]))
        rows = list(csv.DictReader(io.StringIO(text)))
        assert rows[0]["title"].startswith("'=")
        assert rows[0]["citations"] == "3" and rows[0]["cluster"] == "C1"
        assert rows[0]["affirmative_cues"] == "outperformed"

    def test_bibtex_basic(self):
        d = _doc("Attention Is All You Need", "snip", authors="Ashish Vaswani, Noam Shazeer",
                 publication_year=2017, url="https://arxiv.org/abs/1706.03762",
                 publication_info="A Vaswani, N Shazeer - Advances in neural ..., 2017 - proceedings.neurips.cc",
                 citation_count=100000)
        bib = to_bibtex([d])
        assert bib.startswith("@misc{vaswani2017attention,")
        assert "author = {Ashish Vaswani and Noam Shazeer}" in bib
        assert "howpublished = {Advances in neural ...}" in bib
        assert "verify before citing" in bib

    def test_bibtex_keys_unique(self):
        taken = set()
        m = {"authors": "A Smith", "publication_year": 2020, "title": "Same title"}
        assert bibtex_key(m, taken) != bibtex_key(m, taken)

    def test_bibtex_escapes_specials(self):
        bib = to_bibtex([_doc("R&D {costs} 100%")])
        assert r"R\&D \{costs\} 100\%" in bib

    def test_bibtex_missing_author_year(self):
        assert to_bibtex([_doc("Only a title")]).startswith("@misc{anonndonly")

    def test_empty(self):
        assert to_bibtex([]) == ""


# --------------------------------------------------------------------------
# Demo mode fixtures (recorded from live SerpApi)
# --------------------------------------------------------------------------

class TestDemoFixtures:
    def test_every_sample_query_has_a_fixture(self):
        for _, q in SAMPLE_QUERIES:
            assert fixture_path(q).exists(), f"missing fixture for {q!r}"
        assert len(demo_queries_available()) == len(SAMPLE_QUERIES)

    def test_fixtures_load_through_real_schema(self):
        for _, q in SAMPLE_QUERIES:
            docs = load_fixture_docs(q)
            assert len(docs) >= 5
            assert all(d.metadata["source_engine"] == "google_scholar" for d in docs)
            assert all(d.metadata.get("title") for d in docs)

    def test_question_mark_normalized(self):
        _, q = SAMPLE_QUERIES[0]
        assert load_fixture_docs(q + "?")

    def test_unknown_query_raises_helpful_error(self):
        with pytest.raises(FileNotFoundError, match="sample questions"):
            load_fixture_docs("never recorded query")

    def test_fixtures_contain_no_secrets(self):
        for _, q in SAMPLE_QUERIES:
            text = fixture_path(q).read_text(encoding="utf-8")
            assert "api_key" not in text


# --------------------------------------------------------------------------
# Enrichment tests
# --------------------------------------------------------------------------

class TestEnrichment:
    def test_clean_scholar_title(self):
        from enrichment import clean_scholar_title
        dirty = "[HTML] [PDF] A Study of Executive Function... "
        assert clean_scholar_title(dirty) == "A Study of Executive Function"

    def test_extract_doi_from_url(self):
        from enrichment import extract_doi_from_url
        url = "https://www.frontiersin.org/articles/10.3389/fpsyg.2015.00164/full"
        assert extract_doi_from_url(url) == "10.3389/fpsyg.2015.00164"
        assert extract_doi_from_url("https://example.com/no-doi") is None

    def test_reconstruct_openalex_abstract(self):
        from enrichment import _reconstruct_openalex_abstract
        inv = {"Hello": [0], "world": [1], "test": [2]}
        text = _reconstruct_openalex_abstract(inv)
        assert text == "Hello world test"
        assert _reconstruct_openalex_abstract(None) is None


# --------------------------------------------------------------------------
# Embedding comparison tests
# --------------------------------------------------------------------------

class TestEmbeddingComparison:
    def test_compare_embeddings_tfidf(self):
        from analysis import compare_embeddings
        texts = [
            "Bilingualism enhances cognitive control and executive functioning.",
            "Second language learning improves task switching performance.",
            "No evidence of cognitive advantage in bilingual children.",
            "Null replication of bilingual executive function effects.",
            "Dietary energy restriction improves metabolic health in adults.",
            "Intermittent fasting reduces insulin resistance in obese patients.",
        ]
        res = compare_embeddings(texts, k_min=2, k_max=3)
        assert "tfidf_svd" in res
        assert res["tfidf_svd"]["available"] is True
        assert res["tfidf_svd"]["best_k"] in (2, 3)

