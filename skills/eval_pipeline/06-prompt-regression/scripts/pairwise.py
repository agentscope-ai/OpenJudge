#!/usr/bin/env python3
"""Position-debiased pairwise win-rate with bootstrap CI.

Standalone — stdlib only (numpy optional). No OpenJudge dependency: it consumes the
verdicts of ANY pairwise judge, as long as each row says which named model was A vs B and
a score where >= 0.5 means A won. Run each query twice with A/B swapped to debias position.

INPUT (--comparisons file.jsonl), one object per line:
    {"id": "q1", "model_a": "baseline", "model_b": "candidate", "score": 1.0, "dimension": "relevance"}
  - score in [0,1]: >= 0.5 => model_a wins, < 0.5 => model_b wins (ties: feed 0.5).
  - "dimension" is optional; if present, results are reported per dimension.

USAGE
    python pairwise.py --comparisons comps.jsonl
    python pairwise.py --comparisons comps.jsonl --candidate candidate --baseline baseline
    python pairwise.py --comparisons comps.jsonl --json
    python pairwise.py --self-test

EXIT CODE: 0 if candidate is BETTER, 2 otherwise (worse/tied/inconclusive), 1 on usage error.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

MIN_SAMPLES = 10          # below this, CI is too wide to conclude (see SKILL.md)
TIE_CI_WIDTH = 0.30       # CI brackets 0.5 and is narrower than this => genuine tie

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def winners(comparisons: list[dict[str, Any]]) -> list[str]:
    """One named winner per comparison row (respects per-row A/B order = swap-safe)."""
    out = []
    for c in comparisons:
        score = float(c["score"])
        out.append(c["model_a"] if score >= 0.5 else c["model_b"])
    return out


def _bootstrap_ci(wins: list[str], target: str, n_iter: int = 1000, seed: int = 0) -> tuple[float, float]:
    n = len(wins)
    if n == 0:
        return 0.0, 0.0
    if _np is not None:
        rng = _np.random.default_rng(seed)
        rates = [sum(1 for i in rng.integers(0, n, n) if wins[i] == target) / n for _ in range(n_iter)]
        return float(_np.percentile(rates, 2.5)), float(_np.percentile(rates, 97.5))
    rnd = random.Random(seed)
    rates = sorted(sum(1 for _ in range(n) if wins[rnd.randrange(n)] == target) / n for _ in range(n_iter))

    def pct(p: float) -> float:
        k = (len(rates) - 1) * p / 100
        lo, hi = int(k), min(int(k) + 1, len(rates) - 1)
        return rates[lo] + (rates[hi] - rates[lo]) * (k - lo)

    return pct(2.5), pct(97.5)


def verdict_for(wins: list[str], candidate: str, baseline: str, n_iter: int = 1000) -> dict[str, Any]:
    n = len(wins)
    cand = sum(1 for w in wins if w == candidate) / n if n else 0.0
    base = sum(1 for w in wins if w == baseline) / n if n else 0.0
    tie = 1 - cand - base
    if n < MIN_SAMPLES:
        verdict = f"INSUFFICIENT_EVIDENCE (n={n} < {MIN_SAMPLES})"
        ci = [0.0, 1.0]
    else:
        lo, hi = _bootstrap_ci(wins, candidate, n_iter)
        ci = [lo, hi]
        if lo > 0.5:
            verdict = "candidate BETTER"
        elif hi < 0.5:
            verdict = "candidate WORSE"
        elif (hi - lo) < TIE_CI_WIDTH:
            verdict = "TIED (CI brackets 0.5, narrow)"
        else:
            verdict = "INCONCLUSIVE (CI too wide — need more samples)"
    return {"n": n, "candidate_win_rate": round(cand, 3), "baseline_win_rate": round(base, 3),
            "tie_rate": round(tie, 3), "candidate_ci95": [round(ci[0], 3), round(ci[1], 3)], "verdict": verdict}


def analyze(comparisons: list[dict[str, Any]], candidate: str, baseline: str, n_iter: int = 1000) -> dict[str, Any]:
    dims = sorted({c.get("dimension", "overall") for c in comparisons})
    by_dim = {}
    for d in dims:
        subset = [c for c in comparisons if c.get("dimension", "overall") == d]
        by_dim[d] = verdict_for(winners(subset), candidate, baseline, n_iter)
    return {"candidate": candidate, "baseline": baseline, "by_dimension": by_dim}


def render(report: dict[str, Any]) -> str:
    lines = [f"Pairwise: candidate='{report['candidate']}' vs baseline='{report['baseline']}'",
             f"{'Dimension':<16}{'Cand':>7}{'Base':>7}{'Tie':>7}  {'95% CI':>14}  Verdict",
             "-" * 78]
    for d, v in report["by_dimension"].items():
        ci = f"[{v['candidate_ci95'][0]:.2f},{v['candidate_ci95'][1]:.2f}]"
        lines.append(f"{d:<16}{v['candidate_win_rate']:>7.0%}{v['baseline_win_rate']:>7.0%}"
                     f"{v['tie_rate']:>7.0%}  {ci:>14}  {v['verdict']}")
    return "\n".join(lines)


def _self_test() -> None:
    comps = []
    # Candidate clearly better on relevance: wins both orders for 15 queries (30 rows).
    for i in range(15):
        comps.append({"id": f"q{i}", "model_a": "baseline", "model_b": "candidate", "score": 0.0, "dimension": "relevance"})
        comps.append({"id": f"q{i}", "model_a": "candidate", "model_b": "baseline", "score": 1.0, "dimension": "relevance"})
    w = winners(comps)
    assert all(x == "candidate" for x in w), w[:5]
    r = analyze(comps, "candidate", "baseline", n_iter=200)
    assert r["by_dimension"]["relevance"]["verdict"] == "candidate BETTER", r["by_dimension"]["relevance"]
    # Too few samples → insufficient.
    few = [{"model_a": "baseline", "model_b": "candidate", "score": 1.0} for _ in range(6)]
    assert "INSUFFICIENT" in analyze(few, "candidate", "baseline")["by_dimension"]["overall"]["verdict"]
    # 50/50 → tied/inconclusive, never "better".
    tie = []
    for i in range(20):
        tie.append({"model_a": "baseline", "model_b": "candidate", "score": 1.0 if i % 2 else 0.0})
    assert "BETTER" not in analyze(tie, "candidate", "baseline", n_iter=200)["by_dimension"]["overall"]["verdict"]
    print("self-test OK")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Position-debiased pairwise win-rate with bootstrap CI.")
    ap.add_argument("--comparisons", type=Path, help="JSONL with model_a,model_b,score[,dimension].")
    ap.add_argument("--candidate", default="candidate", help="Candidate model name (default: candidate).")
    ap.add_argument("--baseline", default="baseline", help="Baseline model name (default: baseline).")
    ap.add_argument("--n-iter", type=int, default=1000)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        _self_test()
        return 0
    if not args.comparisons:
        ap.error("provide --comparisons (or --self-test)")
    comps = _read_jsonl(args.comparisons)
    if not comps:
        print("No comparisons found.", file=sys.stderr)
        return 1
    report = analyze(comps, args.candidate, args.baseline, args.n_iter)
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else render(report))
    better = any(v["verdict"] == "candidate BETTER" for v in report["by_dimension"].values())
    worse = any(v["verdict"] == "candidate WORSE" for v in report["by_dimension"].values())
    return 0 if better and not worse else 2


if __name__ == "__main__":
    raise SystemExit(main())
