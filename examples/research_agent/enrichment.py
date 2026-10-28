"""Abstract enrichment service using OpenAlex and arXiv APIs (free, zero API keys required).

Google Scholar search results only provide short (100-250 char) snippets that
are often truncated mid-sentence. This module enriches retrieved papers with
full abstracts from OpenAlex (250M+ works) and arXiv, caching results to disk.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

USER_AGENT = "serpapi-research-toolkit (https://github.com/JosephJonathanFernandes/serpapi-research-toolkit; mailto:research@example.com)"


def _reconstruct_openalex_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> Optional[str]:
    """Reconstruct paragraph text from OpenAlex's abstract_inverted_index."""
    if not inverted_index:
        return None
    positions: Dict[int, str] = {}
    for word, pos_list in inverted_index.items():
        for pos in pos_list:
            positions[pos] = word
    if not positions:
        return None
    return " ".join(positions[p] for p in sorted(positions))


def fetch_arxiv_abstract(arxiv_id: str, timeout: int = 5) -> Optional[str]:
    """Fetch abstract directly from arXiv API for a given arXiv ID."""
    clean_id = re.sub(r"v\d+$", "", arxiv_id.strip())
    url = f"http://export.arxiv.org/api/query?id_list={urllib.parse.quote(clean_id)}&max_results=1"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            xml_data = resp.read()
        root = ET.fromstring(xml_data)
        # Atom XML namespace
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", ns)
        if entry is not None:
            summary = entry.find("atom:summary", ns)
            if summary is not None and summary.text:
                return re.sub(r"\s+", " ", summary.text).strip()
    except Exception as exc:
        logger.debug("arXiv lookup failed for %s: %s", arxiv_id, exc)
    return None


def clean_scholar_title(title: str) -> str:
    """Clean Google Scholar title noise (e.g., [HTML], [PDF], ellipses, special chars)."""
    t = re.sub(r"^(\[[A-Z]+\]\s*)+", "", title)
    t = re.sub(r"[\s\.\u2026]+$", "", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def extract_doi_from_url(url: str) -> Optional[str]:
    """Attempt to extract a DOI pattern (10.XXXX/...) from an article URL."""
    match = re.search(r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", url)
    if match:
        doi = match.group(1).rstrip(").,;")
        doi = re.sub(r"/(full|pdf|abstract|epdf|html)$", "", doi, flags=re.IGNORECASE)
        return doi
    return None


def fetch_openalex_by_doi(doi: str, timeout: int = 5) -> Optional[str]:
    """Fetch abstract from OpenAlex using an exact DOI."""
    url = f"https://api.openalex.org/works/https://doi.org/{urllib.parse.quote(doi)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            work = json.loads(resp.read().decode("utf-8"))
        inv = work.get("abstract_inverted_index")
        return _reconstruct_openalex_abstract(inv)
    except Exception as exc:
        logger.debug("OpenAlex DOI lookup failed for %s: %s", doi, exc)
    return None


def fetch_openalex_abstract(title: str, timeout: int = 5) -> Optional[str]:
    """Search OpenAlex for a paper title and reconstruct its full abstract."""
    clean_title = clean_scholar_title(title)
    if not clean_title or len(clean_title) < 8:
        return None
    query = urllib.parse.quote(clean_title)
    url = f"https://api.openalex.org/works?search={query}&per-page=5"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results = data.get("results", [])
        if not results:
            return None

        clean_words = set(clean_title.lower().split())
        for work in results:
            oa_title = work.get("title", "") or ""
            oa_words = set(re.sub(r"[^\w\s]", " ", oa_title.lower()).split())
            if clean_words and oa_words:
                overlap = len(clean_words & oa_words) / max(len(clean_words), 1)
                if overlap < 0.35 and len(clean_words) > 3:
                    continue
            inv = work.get("abstract_inverted_index")
            if inv:
                text = _reconstruct_openalex_abstract(inv)
                if text and len(text) > 80:
                    return text
    except Exception as exc:
        logger.debug("OpenAlex lookup failed for title %r: %s", title, exc)
    return None


def enrich_document_abstract(doc: Document, timeout: int = 5) -> Document:
    """Enrich a Document with a full abstract if available.

    Preserves the original snippet in ``metadata['snippet']`` and replaces
    ``page_content`` with the full abstract when found.
    """
    url = doc.metadata.get("url") or ""
    title = doc.metadata.get("title") or ""
    original_text = doc.page_content

    # If the text is already comprehensive (> 400 chars), no need to enrich
    if len(original_text) > 400:
        return doc

    abstract: Optional[str] = None

    # 1. Try DOI if URL contains one
    doi = extract_doi_from_url(url)
    if doi:
        abstract = fetch_openalex_by_doi(doi, timeout=timeout)

    # 2. Try arXiv if URL points to arXiv
    if not abstract:
        arxiv_match = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+)", url)
        if arxiv_match:
            abstract = fetch_arxiv_abstract(arxiv_match.group(1), timeout=timeout)

    # 3. Try OpenAlex by title
    if not abstract and title:
        abstract = fetch_openalex_abstract(title, timeout=timeout)

    if abstract and len(abstract) > len(original_text):
        metadata = dict(doc.metadata)
        metadata["original_snippet"] = original_text
        metadata["is_enriched"] = True
        return Document(page_content=abstract, metadata=metadata)

    return doc


def enrich_documents(docs: List[Document], max_workers: int = 5) -> List[Document]:
    """Enrich a list of documents concurrently using a thread pool."""
    from concurrent.futures import ThreadPoolExecutor

    if not docs:
        return []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(enrich_document_abstract, docs))
