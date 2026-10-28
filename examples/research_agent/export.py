"""CSV and BibTeX export for retrieved papers (no Streamlit dependency)."""

from __future__ import annotations

import csv
import io
import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence

CSV_FIELDS = [
    "cluster",
    "title",
    "authors",
    "year",
    "citations",
    "publication_info",
    "url",
    "snippet",
    "affirmative_cues",
    "null_or_contrary_cues",
]


def docs_to_rows(
    docs: Sequence[Any],
    clusters: Optional[Sequence[Optional[str]]] = None,
    cues: Optional[Sequence[Any]] = None,
) -> List[Dict[str, Any]]:
    """Flatten Documents into dict rows. ``cues[i]`` is ``(pos_cues, neg_cues)``."""
    rows = []
    for i, d in enumerate(docs):
        m = d.metadata
        pos, neg = (cues[i] if cues else ([], []))
        rows.append(
            {
                "cluster": clusters[i] if clusters else "",
                "title": m.get("title", ""),
                "authors": m.get("authors", "") or "",
                "year": m.get("publication_year", "") or "",
                "citations": m.get("citation_count", "") if m.get("citation_count") is not None else "",
                "publication_info": m.get("publication_info", "") or "",
                "url": m.get("url", "") or "",
                "snippet": d.page_content or "",
                "affirmative_cues": "; ".join(pos),
                "null_or_contrary_cues": "; ".join(neg),
            }
        )
    return rows


def _csv_safe(value: Any) -> Any:
    """Neutralize spreadsheet formula injection (=, +, -, @ at cell start)."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@"):
        return "'" + value
    return value


def to_csv(rows: Sequence[Dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _csv_safe(row.get(k, "")) for k in CSV_FIELDS})
    return buf.getvalue()


# ---------------------------------------------------------------------------
# BibTeX
# ---------------------------------------------------------------------------

_BIBTEX_ESCAPES = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_"}


def _escape(text: str) -> str:
    return "".join(_BIBTEX_ESCAPES.get(c, c) for c in text)


def _ascii_slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]", "", text)


def _venue(publication_info: str) -> str:
    """Pull the venue out of a Scholar summary like 'A Vaswani - Advances in NIPS, 2017 - site'."""
    parts = [p.strip() for p in (publication_info or "").split(" - ")]
    if len(parts) < 2:
        return ""
    venue = re.sub(r",?\s*\b(?:19|20)\d{2}\b\s*$", "", parts[1]).strip(" ,")
    return venue


def bibtex_key(meta: Dict[str, Any], taken: set) -> str:
    authors = meta.get("authors") or ""
    first = authors.split(",")[0].strip().split()[-1] if authors.strip() else "anon"
    year = meta.get("publication_year") or "nd"
    word = next((w for w in re.findall(r"[A-Za-z]{4,}", meta.get("title", ""))), "paper")
    base = f"{_ascii_slug(first).lower() or 'anon'}{year}{_ascii_slug(word).lower()}"
    key, n = base, 2
    while key in taken:
        key, n = f"{base}{n}", n + 1
    taken.add(key)
    return key


def to_bibtex(docs: Sequence[Any]) -> str:
    """Render Documents as ``@misc`` BibTeX entries.

    Google Scholar snippets do not say whether a result is an article, a book
    or a preprint, so every entry is ``@misc`` with a note to verify the
    metadata before citing. Authors may be truncated by Scholar.
    """
    taken: set = set()
    entries = []
    for d in docs:
        m = d.metadata
        key = bibtex_key(m, taken)
        fields = [("title", m.get("title", ""))]
        if m.get("authors"):
            names = [a.strip() for a in m["authors"].split(",") if a.strip()]
            fields.append(("author", " and ".join(names)))
        if m.get("publication_year"):
            fields.append(("year", str(m["publication_year"])))
        venue = _venue(m.get("publication_info", ""))
        if venue:
            fields.append(("howpublished", venue))
        if m.get("url"):
            fields.append(("url", m["url"]))
        if m.get("citation_count") is not None:
            fields.append(("note", f"Cited by {m['citation_count']} per Google Scholar. Metadata taken from a search snippet; verify before citing."))
        body = ",\n".join(
            f"  {name} = {{{value if name == 'url' else _escape(str(value))}}}" for name, value in fields
        )
        entries.append(f"@misc{{{key},\n{body}\n}}")
    return "\n\n".join(entries) + ("\n" if entries else "")
