"""
Possible-disagreement detection for clusters of Google Scholar snippets.

This is a *weak* signal and is labeled that way in the UI:

* Scholar snippets are short, often cut mid-sentence, and frequently come from
  background or definition text rather than a paper's conclusions.
* The default backend is phrase matching: free and deterministic, but it only
  sees surface wording. It cannot tell that two papers study different
  populations or outcomes.
* An optional NLI backend (``backend="nli"``) scores snippet pairs for
  contradiction with a cross-encoder model. It is heavier (needs
  ``transformers`` + ``torch`` and a model download) and has its own failure
  modes; see ``eval/README.md`` for measured numbers.

Neither backend proves that the literature disagrees. Always open the papers.
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# UI wording (single source of truth, used by the app and the README)
# ---------------------------------------------------------------------------

BADGE_TITLE = "Possible disagreement \u2014 human review recommended."
BADGE_BODY = (
    "Snippets in this cluster contain contrasting outcome wording (for example one "
    "reports an advantage and another reports no effect). This is based on short, "
    "possibly truncated snippets, so it may reflect different populations, "
    "outcomes or background text rather than a real dispute. On our small, AI-labeled "
    "(unaudited) test set the best free detector reached about 0.27 recall, so it misses "
    "most disagreements, and absence of this flag means nothing. Open the papers to check."
)

# ---------------------------------------------------------------------------
# Phrase-matching backend
# ---------------------------------------------------------------------------

# Multi-word patterns avoid shallow single-word false positives
# (e.g. matching "limiting" in "limiting calories").
#
# ``*_V1`` is the original lexicon, written before any evaluation. ``*_V2`` adds
# broader generic phrasing after the first eval showed V1 never fired on real
# snippets. V2 was authored by someone who had read the eval snippets, so its
# eval numbers are optimistic; treat them as an upper bound on new topics.
AFFIRMATIVE_PATTERNS_V1 = [
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

NULL_OR_CONTRARY_PATTERNS_V1 = [
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

AFFIRMATIVE_PATTERNS_V2 = AFFIRMATIVE_PATTERNS_V1 + [
    r"\bplays?\s+a\s+causal\s+role\b",
    r"\b(?:positively|significantly)\s+(?:influence|affect|impact|predict)s?\b",
    r"\b(?:improves?|improved|increases?|increased|enhances?|promotes?|boosts?)\s+(?!the\b|a\b|an\b|\d)\w+",
    r"\bfound\s+improvements?\b",
    r"\bnow\s+shown\b",
    r"\bpattern\s+holds\b",
    r"\breplicat(?:ing|ed|es)\s+the\s+(?:classic\s+)?\w+\s+effect\b",
]

NULL_OR_CONTRARY_PATTERNS_V2 = NULL_OR_CONTRARY_PATTERNS_V1 + [
    r"\bno\s+(?:clear\s+|convincing\s+|significant\s+|meaningful\s+)?evidence\b",
    r"\b(?:does|do|did)\s+not\s+(?:elicit|alter|improve|matter|affect|lead|differ|replicate|change|influence|increase|reduce)\b",
    r"\b(?:little|minimal|negligible)\s+(?:impact|effect|benefit|difference)\b",
    r"\bbut\s+not\b",
    r"\bfail(?:s|ed|ure)?\s+to\s+(?:replicate|materiali[sz]e|show|improve|detect|support)\b",
    r"\bunsuccessful\s+attempts?\b",
    r"\bnot\s+(?:associated|related)\b",
]

# Default lexicon used by the app.
AFFIRMATIVE_PATTERNS = AFFIRMATIVE_PATTERNS_V2
NULL_OR_CONTRARY_PATTERNS = NULL_OR_CONTRARY_PATTERNS_V2

_PATTERN_SETS = {
    "v1": (AFFIRMATIVE_PATTERNS_V1, NULL_OR_CONTRARY_PATTERNS_V1),
    "v2": (AFFIRMATIVE_PATTERNS_V2, NULL_OR_CONTRARY_PATTERNS_V2),
}

Cues = Tuple[List[str], List[str]]


def _matches(patterns: Sequence[str], text: str) -> List[str]:
    found: List[str] = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            s = re.sub(r"\s+", " ", m.group(0).lower().strip())
            if s not in found:
                found.append(s)
    return found


def score_text(text: str, patterns: str = "v2") -> Cues:
    """Return (affirmative_cues, null_or_contrary_cues) found in *text*."""
    if not text:
        return [], []
    pos_pats, neg_pats = _PATTERN_SETS[patterns]
    return _matches(pos_pats, text), _matches(neg_pats, text)


def phrase_pair_disagrees(text_a: str, text_b: str, patterns: str = "v2") -> bool:
    """One snippet carries an affirmative cue and the *other* a null/contrary cue."""
    pos_a, neg_a = score_text(text_a, patterns)
    pos_b, neg_b = score_text(text_b, patterns)
    return bool((pos_a and neg_b) or (pos_b and neg_a))


# ---------------------------------------------------------------------------
# Optional NLI backend
# ---------------------------------------------------------------------------

#: ``judge(text_a, text_b) -> probability that the two snippets contradict``
ContradictionJudge = Callable[[str, str], float]

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
DEFAULT_NLI_THRESHOLD = 0.5


def make_nli_judge(model_name: str = DEFAULT_NLI_MODEL) -> ContradictionJudge:
    """Build a contradiction judge from a Hugging Face NLI cross-encoder.

    Needs ``pip install transformers torch``; downloads the model on first use.
    The pair is scored in both directions and the larger contradiction
    probability is returned.
    """
    try:
        from transformers import pipeline  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional deps
        raise ImportError(
            "The NLI backend needs `transformers` and `torch`: pip install transformers torch"
        ) from exc

    clf = pipeline("text-classification", model=model_name, top_k=None)

    def _contradiction(a: str, b: str) -> float:
        scores = clf({"text": a[:512], "text_pair": b[:512]})
        if scores and isinstance(scores[0], list):
            scores = scores[0]
        for item in scores:
            if str(item["label"]).lower().startswith("contradiction"):
                return float(item["score"])
        return 0.0

    def judge(a: str, b: str) -> float:
        return max(_contradiction(a, b), _contradiction(b, a))

    return judge


# ---------------------------------------------------------------------------
# Tier 3: Optional LLM Judge
# ---------------------------------------------------------------------------

@dataclass
class LLMJudgeVerdict:
    relationship: str  # "CONTRADICT", "AGREE", "UNRELATED"
    score: float       # contradiction score 0.0 - 1.0
    rationale: str     # brief explanation


LLMJudge = Callable[[str, str], LLMJudgeVerdict]


def make_llm_judge(
    llm: Optional[Any] = None,
    model_name: str = "gpt-4o-mini",
) -> LLMJudge:
    """Build a contradiction judge using an LLM (Tier 3).

    Accepts any LangChain BaseChatModel or a callable returning a string/ChatResult.
    If *llm* is None, attempts to construct ``ChatOpenAI(model=model_name, temperature=0)``.
    Returns a callable ``judge(text_a, text_b) -> LLMJudgeVerdict``.
    """
    if llm is None:
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The Tier 3 LLM judge needs `langchain-openai`: pip install langchain-openai. "
                "Alternatively, pass an initialized chat model to `make_llm_judge(llm=...)`."
            ) from exc
        llm = ChatOpenAI(model=model_name, temperature=0)

    prompt_template = (
        "You are an expert scientific meta-analyst evaluating academic claims.\n"
        "Given two papers (titles and abstracts or search snippets), classify their empirical relationship:\n\n"
        "PAPER A:\n{paper_a}\n\n"
        "PAPER B:\n{paper_b}\n\n"
        "Classify into exactly one of:\n"
        "- CONTRADICT: The papers make directly conflicting empirical claims or opposite conclusions on the same question/intervention.\n"
        "- AGREE: The papers report consistent, mutually supporting, or replicating findings on the same topic.\n"
        "- UNRELATED: The papers investigate different questions, mechanisms, or populations, or are not directly comparable.\n\n"
        "Respond in valid JSON only with keys:\n"
        '{{"relationship": "CONTRADICT" | "AGREE" | "UNRELATED", "contradiction_score": <float 0.0-1.0>, "rationale": "<one sentence>"}}'
    )

    def judge(text_a: str, text_b: str) -> LLMJudgeVerdict:
        formatted = prompt_template.format(paper_a=text_a[:1500], paper_b=text_b[:1500])
        if hasattr(llm, "invoke"):
            resp = llm.invoke(formatted)
            content = resp.content if hasattr(resp, "content") else str(resp)
        else:
            content = str(llm(formatted))

        rel = "UNRELATED"
        score = 0.0
        rationale = ""
        try:
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
                rel = parsed.get("relationship", "UNRELATED").upper()
                score = float(parsed.get("contradiction_score", 0.0))
                rationale = parsed.get("rationale", "")
        except Exception:
            if "CONTRADICT" in content.upper():
                rel = "CONTRADICT"
                score = 0.8
            elif "AGREE" in content.upper():
                rel = "AGREE"
                score = 0.0
            rationale = content[:150]

        return LLMJudgeVerdict(relationship=rel, score=score, rationale=rationale)

    return judge


# ---------------------------------------------------------------------------
# Cluster-level detection
# ---------------------------------------------------------------------------

@dataclass
class DisagreementResult:
    flagged: bool
    per_paper_cues: List[Cues]
    #: index pairs (within the cluster) that triggered the flag
    pairs: List[Tuple[int, int]] = field(default_factory=list)
    backend: str = "phrases"


def detect_possible_disagreement(
    texts: Sequence[str],
    backend: str = "phrases",
    judge: Optional[Union[ContradictionJudge, LLMJudge]] = None,
    nli_threshold: float = DEFAULT_NLI_THRESHOLD,
) -> DisagreementResult:
    """Flag a cluster when two *different* snippets appear to disagree.

    ``backend="phrases"`` (Tier 1, default, free): one snippet has an affirmative cue
    and another has a null/contrary cue. A single snippet containing both is
    not enough.

    ``backend="nli"`` (Tier 2): any snippet pair whose contradiction probability from
    *judge* is at least ``nli_threshold``. A judge is built with
    :func:`make_nli_judge` if none is passed.

    ``backend="llm"`` (Tier 3): any snippet pair evaluated by an LLM judge that returns
    a ``CONTRADICT`` relationship or contradiction score >= ``nli_threshold``.
    """
    per_paper = [score_text(t) for t in texts]
    pairs: List[Tuple[int, int]] = []

    if backend == "phrases":
        for i, j in itertools.combinations(range(len(texts)), 2):
            (pos_i, neg_i), (pos_j, neg_j) = per_paper[i], per_paper[j]
            if (pos_i and neg_j) or (pos_j and neg_i):
                pairs.append((i, j))
    elif backend == "nli":
        nli_judge = judge or make_nli_judge()
        for i, j in itertools.combinations(range(len(texts)), 2):
            val = nli_judge(texts[i], texts[j])
            prob = val.score if isinstance(val, LLMJudgeVerdict) else float(val)
            if prob >= nli_threshold:
                pairs.append((i, j))
    elif backend == "llm":
        llm_judge = judge or make_llm_judge()
        for i, j in itertools.combinations(range(len(texts)), 2):
            verdict = llm_judge(texts[i], texts[j])
            if isinstance(verdict, LLMJudgeVerdict):
                if verdict.relationship == "CONTRADICT" or verdict.score >= nli_threshold:
                    pairs.append((i, j))
            elif float(verdict) >= nli_threshold:
                pairs.append((i, j))
    else:
        raise ValueError(f"Unknown backend {backend!r}; use 'phrases', 'nli', or 'llm'.")

    return DisagreementResult(flagged=bool(pairs), per_paper_cues=per_paper, pairs=pairs, backend=backend)


# ---------------------------------------------------------------------------
# Findings extraction + stance-toward-claim detection
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(r"\b(?:results?|conclusions?|findings)\s*[:\-]\s*", re.IGNORECASE)


def extract_findings(text: str, max_sentences: int = 3, max_chars: int = 600) -> str:
    """Return the part of an abstract that states findings.

    Uses a structured-abstract ``Results:``/``Conclusion:`` section if present,
    otherwise the last ``max_sentences`` sentences (findings come last in most
    abstracts, background first). Short texts are returned unchanged.
    """
    if not text or len(text) <= max_chars:
        return text or ""
    m = _SECTION_RE.search(text)
    if m:
        return text[m.start():][:max_chars]
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    tail = " ".join(sentences[-max_sentences:])
    return tail[-max_chars:]


STANCES = ("SUPPORTS", "NULL", "MIXED", "UNCLEAR")

STANCE_PROMPT = (
    "You classify what a paper's own FINDINGS say about a claim.\n\n"
    "CLAIM: {claim}\n\nPAPER TEXT:\n{text}\n\n"
    "Answer with exactly one stance:\n"
    "- SUPPORTS: the paper reports evidence in favour of the claim.\n"
    "- NULL: the paper reports no effect, a failure to replicate, or evidence against the claim.\n"
    "- MIXED: effects depend on conditions or evidence is split.\n"
    "- UNCLEAR: the text does not state findings about the claim (background, protocol, commentary).\n\n"
    'Respond as JSON only: {{"stance": "SUPPORTS|NULL|MIXED|UNCLEAR", "rationale": "<one sentence>"}}'
)

StanceClassifier = Callable[[str, str], str]


def make_stance_classifier(llm: Optional[Any] = None, model_name: str = "gpt-4o-mini") -> StanceClassifier:
    """Build ``classify(claim, text) -> stance`` from a LangChain chat model."""
    if llm is None:
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Pass a chat model via make_stance_classifier(llm=...) or pip install langchain-openai."
            ) from exc
        llm = ChatOpenAI(model=model_name, temperature=0)

    def classify(claim: str, text: str) -> str:
        resp = llm.invoke(STANCE_PROMPT.format(claim=claim, text=text[:1500]))
        content = resp.content if hasattr(resp, "content") else str(resp)
        m = re.search(r"\{.*\}", content, re.DOTALL)
        try:
            stance = str(json.loads(m.group(0)).get("stance", "UNCLEAR")).upper() if m else "UNCLEAR"
        except Exception:
            stance = "UNCLEAR"
        return stance if stance in STANCES else "UNCLEAR"

    return classify


def stances_oppose(a: str, b: str) -> bool:
    """Two papers contradict when one SUPPORTS the claim and the other reports NULL."""
    return {a, b} == {"SUPPORTS", "NULL"}
