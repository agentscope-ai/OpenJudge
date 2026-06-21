#!/usr/bin/env python3
"""RAG diagnostic matrix: is the problem retrieval or generation?

Standalone — stdlib only. No OpenJudge dependency: it consumes per-trace judgments you've
already produced (with any faithfulness judge and any retrieval signal). A single "RAG
accuracy" number hides whether you need to fix chunking/embeddings (retrieval) or the
prompt/model (generation); this 2x2 matrix separates them.

INPUT (--traces traces.jsonl), one object per line. Provide booleans, or scores+thresholds:
    {"retrieval_good": true, "faithful": false}
  or
    {"faithfulness_score": 5, "recall_at_k": 0.8}      # with --faithful-threshold / --recall-threshold
  - faithfulness_score on a 1-5 scale (>= --faithful-threshold => faithful; default 4)
  - recall_at_k in [0,1] (>= --recall-threshold => retrieval_good; default 0.5)
  - if neither retrieval signal is present, rows are treated as retrieval_good=unknown and
    only the generation axis is reported.

USAGE
  python rag_diagnostic.py --traces traces.jsonl
  python rag_diagnostic.py --traces traces.jsonl --faithful-threshold 4 --recall-threshold 0.5
  python rag_diagnostic.py --traces traces.jsonl --json
  python rag_diagnostic.py --self-test

EXIT: 0 always (diagnostic), 1 on usage error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def _faithful(row: dict[str, Any], threshold: float) -> bool:
    if "faithful" in row:
        return bool(row["faithful"])
    if "faithfulness_score" in row:
        return float(row["faithfulness_score"]) >= threshold
    raise ValueError("trace needs 'faithful' (bool) or 'faithfulness_score' (number)")


def _retrieval_good(row: dict[str, Any], threshold: float) -> bool | None:
    if "retrieval_good" in row:
        return bool(row["retrieval_good"])
    if "recall_at_k" in row:
        return float(row["recall_at_k"]) >= threshold
    return None


def analyze(traces: list[dict[str, Any]], faithful_threshold: float = 4.0,
            recall_threshold: float = 0.5) -> dict[str, Any]:
    n = len(traces)
    faith = [_faithful(t, faithful_threshold) for t in traces]
    retr = [_retrieval_good(t, recall_threshold) for t in traces]
    have_retrieval = all(r is not None for r in retr)

    gen = {"faithful": round(sum(faith) / n, 3) if n else 0.0,
           "hallucinating": round(sum(1 for f in faith if not f) / n, 3) if n else 0.0}

    result: dict[str, Any] = {"n": n, "generation": gen, "has_retrieval_signal": have_retrieval}
    if not have_retrieval:
        result["primary_issue"] = (
            "generation: {:.0%} of answers hallucinate".format(gen["hallucinating"])
            if gen["hallucinating"] > 0.1 else "generation looks healthy; add a retrieval signal to localize further")
        return result

    # 2x2 matrix
    cells = {"good_faithful": 0, "good_hallucinating": 0, "poor_faithful": 0, "poor_hallucinating": 0}
    for f, r in zip(faith, retr):
        key = ("good" if r else "poor") + ("_faithful" if f else "_hallucinating")
        cells[key] += 1
    matrix = {k: round(v / n, 3) for k, v in cells.items()} if n else cells
    # Primary issue heuristic
    if matrix["good_hallucinating"] >= matrix["poor_hallucinating"] and matrix["good_hallucinating"] > 0.1:
        primary = (f"generation: {matrix['good_hallucinating']:.0%} hallucinate DESPITE good retrieval "
                   "-> fix the generation prompt/model, not retrieval")
    elif matrix["poor_hallucinating"] > 0.15:
        primary = (f"retrieval: {matrix['poor_hallucinating']:.0%} have poor retrieval AND hallucinate "
                   "-> fix retrieval (chunking/embeddings) first")
    else:
        primary = "system largely healthy"
    result.update({"matrix": matrix, "primary_issue": primary})
    return result


def render(report: dict[str, Any]) -> str:
    lines = [f"RAG diagnostic  (n={report['n']})",
             f"  generation: faithful={report['generation']['faithful']:.0%}  "
             f"hallucinating={report['generation']['hallucinating']:.0%}"]
    if report["has_retrieval_signal"]:
        m = report["matrix"]
        lines += [
            "                     Faithful   Hallucinating",
            f"  Retrieval Good      {m['good_faithful']:>7.0%}      {m['good_hallucinating']:>7.0%}",
            f"  Retrieval Poor      {m['poor_faithful']:>7.0%}      {m['poor_hallucinating']:>7.0%}",
        ]
    else:
        lines.append("  (no retrieval signal — generation axis only; add recall_at_k or retrieval_good)")
    lines.append(f"  PRIMARY ISSUE: {report['primary_issue']}")
    return "\n".join(lines)


def _self_test() -> None:
    # 62% good/faithful, 10% good/hallucinating, 8% poor/faithful, 20% poor/hallucinating
    traces = (
        [{"retrieval_good": True, "faithful": True}] * 62
        + [{"retrieval_good": True, "faithful": False}] * 10
        + [{"retrieval_good": False, "faithful": True}] * 8
        + [{"retrieval_good": False, "faithful": False}] * 20
    )
    r = analyze(traces)
    assert r["matrix"]["good_hallucinating"] == 0.1, r["matrix"]
    assert "retrieval" in r["primary_issue"] and "20%" in r["primary_issue"], r["primary_issue"]
    # score-based + no retrieval signal => generation-only
    sc = [{"faithfulness_score": 5}] * 9 + [{"faithfulness_score": 2}] * 1
    rs = analyze(sc, faithful_threshold=4)
    assert rs["has_retrieval_signal"] is False
    assert rs["generation"]["hallucinating"] == 0.1
    # good retrieval but generation broken
    gb = [{"retrieval_good": True, "faithful": False}] * 5 + [{"retrieval_good": True, "faithful": True}] * 5
    assert "generation" in analyze(gb)["primary_issue"]
    print("self-test OK")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RAG retrieval-vs-generation diagnostic matrix.")
    ap.add_argument("--traces", type=Path, help="JSONL with faithful|faithfulness_score[,retrieval_good|recall_at_k].")
    ap.add_argument("--faithful-threshold", type=float, default=4.0, help="faithfulness_score >= this => faithful.")
    ap.add_argument("--recall-threshold", type=float, default=0.5, help="recall_at_k >= this => retrieval_good.")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        _self_test()
        return 0
    if not args.traces:
        ap.error("provide --traces (or --self-test)")
    traces = _read_jsonl(args.traces)
    if not traces:
        print("No traces found.", file=sys.stderr)
        return 1
    report = analyze(traces, args.faithful_threshold, args.recall_threshold)
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
