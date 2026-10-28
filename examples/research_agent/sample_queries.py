"""Sample research questions and their recorded-fixture locations (no heavy imports)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

FIXTURE_DIR = Path(__file__).parent / "fixtures"

SAMPLE_QUERIES: List[Tuple[str, str]] = [
    ("Bilingual Advantage", "bilingual advantage executive function cognitive control"),
    ("Intermittent Fasting", "intermittent fasting insulin sensitivity metabolic health"),
    ("LLM Theory of Mind", "large language models theory of mind reasoning"),
]


def normalize_query(query: str) -> str:
    """Same normalization the app applies before searching Google Scholar."""
    return query.strip().rstrip("?").strip()


def fixture_path(query: str, directory: Path = FIXTURE_DIR) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", normalize_query(query).lower()).strip("_")
    return directory / f"{slug}.json"


# Contested literatures used (together with SAMPLE_QUERIES) to build the labeled
# evaluation set in ``eval/``. Snippets are recorded from live Google Scholar.
EVAL_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "eval" / "fixtures"
EVAL_EXTRA_QUERIES: List[str] = [
    "ego depletion self-control replication",
    "power posing hormones behavior",
    "growth mindset intervention academic achievement",
    "vitamin D supplementation depression",
    "screen time adolescent well-being",
    "omega-3 fatty acids cardiovascular disease prevention",
    "mindfulness meditation anxiety randomized trial",
]
