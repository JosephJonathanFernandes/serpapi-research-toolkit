"""
External validation of the stance classifiers on SciFact (Wadden et al., EMNLP 2020).

SciFact pairs expert-written scientific claims with paper abstracts. Each
(claim, abstract) pair is labeled by human annotators as SUPPORT, CONTRADICT,
or NEI (not enough info). We use the public **dev** split (300 claims, 340
claim-abstract pairs; the test split's labels are hidden).

This answers one question the project's own 120-pair benchmark can't: how good
are the stance classifiers on labels nobody on this project wrote?

Mapping from our stance labels to SciFact labels:
    SUPPORTS -> SUPPORT,  NULL -> CONTRADICT,  MIXED / UNCLEAR -> NEI

Data licence: SciFact is CC BY-NC 2.0 (corpus from S2ORC, ODC-By 1.0). The data
is downloaded on demand into eval/scifact/data/ and is NOT committed.

Usage:
    python eval/scifact/run_scifact.py --backend phrases-v1
    python eval/scifact/run_scifact.py --backend nli
    python eval/scifact/run_scifact.py --backend llm               # OpenAI key or local Ollama
    python eval/scifact/run_scifact.py --backend llm --limit 100   # stratified subsample
    python eval/scifact/run_scifact.py --report                    # rebuild results.md only
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import sys
import tarfile
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "examples" / "research_agent"))

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DATA_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
DATA_DIR = HERE / "data" / "data"
RESULTS_JSON = HERE / "results.json"
RESULTS_MD = HERE / "results.md"
CACHE_DIR = HERE / "cache"

LABELS = ("SUPPORT", "CONTRADICT", "NEI")
STANCE_TO_SCIFACT = {"SUPPORTS": "SUPPORT", "NULL": "CONTRADICT", "MIXED": "NEI", "UNCLEAR": "NEI"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def ensure_data() -> None:
    if (DATA_DIR / "claims_dev.jsonl").exists():
        return
    print(f"Downloading SciFact from {DATA_URL} (CC BY-NC 2.0) ...")
    raw = urllib.request.urlopen(DATA_URL, timeout=120).read()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        members = [m for m in tar.getmembers() if not Path(m.name).name.startswith("._")]
        tar.extractall(HERE / "data", members=members)  # noqa: S202 - trusted AllenAI archive


def load_pairs() -> List[Dict]:
    """(claim, abstract, gold) pairs from the SciFact dev split."""
    ensure_data()
    corpus = {}
    with open(DATA_DIR / "corpus.jsonl", encoding="utf-8") as fh:
        for line in fh:
            doc = json.loads(line)
            corpus[doc["doc_id"]] = doc
    pairs = []
    with open(DATA_DIR / "claims_dev.jsonl", encoding="utf-8") as fh:
        for line in fh:
            claim = json.loads(line)
            evidence = claim.get("evidence", {})
            for doc_id in claim["cited_doc_ids"]:
                ev = evidence.get(str(doc_id))
                if ev:
                    gold = (ev[0] if isinstance(ev, list) else ev)["label"]
                else:
                    gold = "NEI"
                doc = corpus[doc_id]
                pairs.append(
                    {
                        "key": f"{claim['id']}-{doc_id}",
                        "claim": claim["claim"],
                        "title": doc["title"],
                        "abstract": " ".join(s.strip() for s in doc["abstract"]),
                        "gold": gold,
                    }
                )
    return pairs


def stratified_sample(pairs: List[Dict], limit: int, seed: int = 0) -> List[Dict]:
    """Deterministic subsample that keeps the label proportions."""
    rng = random.Random(seed)
    by_label: Dict[str, List[Dict]] = {}
    for p in pairs:
        by_label.setdefault(p["gold"], []).append(p)
    out = []
    for label, items in sorted(by_label.items()):
        k = max(1, round(limit * len(items) / len(pairs)))
        out.extend(rng.sample(items, min(k, len(items))))
    return sorted(out, key=lambda p: p["key"])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def score(golds: List[str], preds: List[str]) -> Dict:
    confusion = {g: Counter() for g in LABELS}
    for g, p in zip(golds, preds):
        confusion[g][p] += 1
    per_class = {}
    for lab in LABELS:
        tp = confusion[lab][lab]
        n_pred = sum(confusion[g][lab] for g in LABELS)
        n_gold = sum(confusion[lab].values())
        prec = tp / n_pred if n_pred else 0.0
        rec = tp / n_gold if n_gold else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[lab] = {
            "tp": tp, "n_pred": n_pred, "n_gold": n_gold,
            "precision": prec, "recall": rec, "f1": f1,
            "ci_p": wilson(tp, n_pred), "ci_r": wilson(tp, n_gold),
        }
    acc = sum(g == p for g, p in zip(golds, preds)) / len(golds) if golds else 0.0
    return {
        "n": len(golds),
        "accuracy": acc,
        "macro_f1": sum(c["f1"] for c in per_class.values()) / len(LABELS),
        "per_class": per_class,
        "confusion": {g: dict(confusion[g]) for g in LABELS},
    }


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def build_classifier(backend: str, model: str | None) -> Tuple[Callable[[str, str], str], str]:
    from disagreement import (
        make_nli_stance_classifier,
        make_phrase_stance_classifier,
        make_stance_classifier,
    )

    if backend.startswith("phrases-"):
        return make_phrase_stance_classifier(backend.split("-", 1)[1]), f"phrase lexicon {backend.split('-', 1)[1]}"
    if backend == "nli":
        return make_nli_stance_classifier(), "cross-encoder/nli-deberta-v3-small (claim as hypothesis)"
    if backend == "llm":
        from llm_backends import default_chat_model

        llm = default_chat_model(model)
        desc = repr(llm) if "Ollama" in type(llm).__name__ else f"{type(llm).__name__}({getattr(llm, 'model_name', model)})"
        return make_stance_classifier(llm=llm), desc
    if backend == "majority":
        return (lambda claim, text: "SUPPORTS"), "always SUPPORT (majority class)"
    raise ValueError(backend)


def run(backend: str, limit: int | None, model: str | None) -> Dict:
    pairs = load_pairs()
    if limit:
        pairs = stratified_sample(pairs, limit)
    classify, desc = build_classifier(backend, model)

    cache_path = CACHE_DIR / f"{backend}{'-' + model.replace(':', '_') if model else ''}.json"
    cache: Dict[str, str] = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    use_cache = backend == "llm"

    golds, preds = [], []
    t0 = time.time()
    for i, p in enumerate(pairs, 1):
        text = f"{p['title']}. {p['abstract']}"
        if use_cache and p["key"] in cache:
            stance = cache[p["key"]]
        else:
            stance = classify(p["claim"], text)
            if use_cache:
                cache[p["key"]] = stance
                if i % 5 == 0:
                    CACHE_DIR.mkdir(exist_ok=True)
                    cache_path.write_text(json.dumps(cache, indent=0), encoding="utf-8")
        golds.append(p["gold"])
        preds.append(STANCE_TO_SCIFACT.get(stance, "NEI"))
        if use_cache and i % 10 == 0:
            print(f"  {i}/{len(pairs)}  ({time.time() - t0:.0f}s)", flush=True)
    if use_cache:
        CACHE_DIR.mkdir(exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=0), encoding="utf-8")

    res = score(golds, preds)
    res.update({"backend": backend, "description": desc, "limit": limit, "seconds": round(time.time() - t0, 1)})
    return res


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt_ci(ci) -> str:
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]" if not any(math.isnan(x) for x in ci) else "n/a"


def write_report(all_results: Dict[str, Dict]) -> None:
    order = ["majority", "phrases-v1", "phrases-v2", "phrases-v3", "nli", "llm"]
    rows = [all_results[k] for k in order if k in all_results] + [
        v for k, v in all_results.items() if k not in order
    ]
    lines = [
        "# SciFact external validation (stance toward a claim)",
        "",
        "Generated by `python eval/scifact/run_scifact.py`. Do not edit by hand.",
        "",
        "SciFact dev split: human-written claims, human labels (SUPPORT / CONTRADICT / NEI) for each",
        "cited abstract. None of these labels were written by this project, and none of these topics",
        "were seen while writing any phrase lexicon. Mapping: SUPPORTS→SUPPORT, NULL→CONTRADICT,",
        "MIXED/UNCLEAR→NEI.",
        "",
        "| Backend | n | Accuracy | Macro-F1 | CONTRADICT P [95% CI] | CONTRADICT R [95% CI] | CONTRADICT F1 | SUPPORT F1 |",
        "|:---|---:|---:|---:|:---:|:---:|---:|---:|",
    ]
    for r in rows:
        c, s = r["per_class"]["CONTRADICT"], r["per_class"]["SUPPORT"]
        n = f"{r['n']}" + (" (stratified sample)" if r.get("limit") else "")
        lines.append(
            f"| {r['backend']} — {r['description']} | {n} | {r['accuracy']:.2f} | {r['macro_f1']:.2f} | "
            f"{c['precision']:.2f} {fmt_ci(c['ci_p'])} (of {c['n_pred']}) | {c['recall']:.2f} {fmt_ci(c['ci_r'])} | "
            f"{c['f1']:.2f} | {s['f1']:.2f} |"
        )
    lines += ["", "## Confusion matrices (rows = gold, columns = predicted)", ""]
    for r in rows:
        lines.append(f"**{r['backend']}**")
        lines.append("")
        lines.append("| gold \\ pred | SUPPORT | CONTRADICT | NEI |")
        lines.append("|:---|---:|---:|---:|")
        for g in LABELS:
            row = r["confusion"][g]
            lines.append(f"| {g} | {row.get('SUPPORT', 0)} | {row.get('CONTRADICT', 0)} | {row.get('NEI', 0)} |")
        lines.append("")
    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["majority", "phrases-v1", "phrases-v2", "phrases-v3", "nli", "llm"])
    ap.add_argument("--limit", type=int, default=None, help="Stratified subsample size (for slow backends).")
    ap.add_argument("--model", default=None, help="Chat model name (e.g. mistral, gpt-4o-mini).")
    ap.add_argument("--report", action="store_true", help="Only rebuild results.md from results.json.")
    args = ap.parse_args()

    all_results = json.loads(RESULTS_JSON.read_text(encoding="utf-8")) if RESULTS_JSON.exists() else {}
    if not args.report:
        if not args.backend:
            ap.error("--backend is required unless --report is given")
        try:
            res = run(args.backend, args.limit, args.model)
        except (ImportError, RuntimeError) as exc:
            print(f"NOT EVALUATED: {exc}")
            return 2
        all_results[args.backend] = res
        RESULTS_JSON.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        c = res["per_class"]["CONTRADICT"]
        print(
            f"{args.backend}: n={res['n']} acc={res['accuracy']:.2f} macroF1={res['macro_f1']:.2f} "
            f"CONTRADICT P={c['precision']:.2f} R={c['recall']:.2f} F1={c['f1']:.2f} ({res['seconds']}s)"
        )
    write_report(all_results)
    print(f"Wrote {RESULTS_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
