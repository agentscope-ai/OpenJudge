#!/usr/bin/env python3
"""Evaluation dataset coverage check.

Standalone — stdlib only. No OpenJudge dependency. Validates that a dataset has enough
samples per dimension and per difficulty stratum BEFORE you trust any per-slice metric.
A dimension with too few cases is being guessed at, not measured.

INPUT (--dataset dataset.jsonl), rows in the standard eval shape:
    {"query": "...", "response": "...",
     "metadata": {"dimension": "order_accuracy", "difficulty": "boundary"}}
  (dimension/difficulty may also be top-level keys.)

THRESHOLDS (see eval-data-principles): >=5 per dimension to measure at all; >=10 per
(dimension x stratum) cell for a reliable per-stratum number; target strata mix ~60/30/10.

USAGE
  python coverage_check.py --dataset dataset.jsonl
  python coverage_check.py --dataset dataset.jsonl --json
  python coverage_check.py --self-test

EXIT: 0 if coverage is adequate, 2 if thin cells found, 1 on usage error.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

MIN_PER_DIMENSION = 5
MIN_PER_CELL = 10
EXPECTED_STRATA = {"easy": 0.55, "boundary": 0.30, "adversarial": 0.10}  # rough target mix


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def _field(row: dict[str, Any], key: str) -> str:
    meta = row.get("metadata", {})
    return str(meta.get(key, row.get(key, "")) or "unspecified")


def analyze(dataset: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(dataset)
    dim_counts = Counter(_field(r, "dimension") for r in dataset)
    strat_counts = Counter(_field(r, "difficulty") for r in dataset)
    cell_counts: Counter = Counter((_field(r, "dimension"), _field(r, "difficulty")) for r in dataset)

    thin_dims = sorted(d for d, c in dim_counts.items() if c < MIN_PER_DIMENSION)
    thin_cells = sorted(f"{d}/{s}" for (d, s), c in cell_counts.items() if c < MIN_PER_CELL)

    strata_mix = {s: round(c / n, 2) for s, c in strat_counts.items()} if n else {}
    adversarial_share = strata_mix.get("adversarial", 0.0)
    warnings = []
    if adversarial_share < 0.10:
        warnings.append(f"adversarial share {adversarial_share:.0%} < 10% minimum")
    if "unspecified" in dim_counts:
        warnings.append(f"{dim_counts['unspecified']} rows have no dimension label")

    adequate = not thin_dims and not thin_cells
    return {
        "n": n,
        "dimensions": dict(dim_counts),
        "strata": dict(strat_counts),
        "strata_mix": strata_mix,
        "thin_dimensions": thin_dims,
        "thin_cells": thin_cells,
        "warnings": warnings,
        "verdict": "adequate" if adequate else "thin_coverage",
    }


def render(report: dict[str, Any]) -> str:
    lines = [f"Coverage check  (n={report['n']})", "  dimensions:"]
    for d, c in sorted(report["dimensions"].items()):
        flag = "  <-- THIN (<5)" if d in report["thin_dimensions"] else ""
        lines.append(f"    {d:<24} {c}{flag}")
    lines.append("  strata mix: " + ", ".join(f"{s}={p:.0%}" for s, p in sorted(report["strata_mix"].items())))
    if report["thin_cells"]:
        lines.append(f"  thin (dim x stratum) cells (<{MIN_PER_CELL}): " + ", ".join(report["thin_cells"]))
    for w in report["warnings"]:
        lines.append(f"  ! {w}")
    lines.append(f"  VERDICT: {report['verdict']}")
    if report["verdict"] != "adequate":
        lines.append("    add samples to thin dimensions/cells before trusting their per-slice metrics")
    return "\n".join(lines)


def _self_test() -> None:
    data = []
    data += [{"metadata": {"dimension": "order_accuracy", "difficulty": "easy"}}] * 12
    data += [{"metadata": {"dimension": "order_accuracy", "difficulty": "boundary"}}] * 11
    data += [{"metadata": {"dimension": "order_accuracy", "difficulty": "adversarial"}}] * 10
    data += [{"metadata": {"dimension": "tone", "difficulty": "easy"}}] * 3   # thin dimension
    r = analyze(data)
    assert "tone" in r["thin_dimensions"], r["thin_dimensions"]
    assert r["verdict"] == "thin_coverage"
    # A well-covered single dimension.
    good = ([{"metadata": {"dimension": "d", "difficulty": "easy"}}] * 11
            + [{"metadata": {"dimension": "d", "difficulty": "boundary"}}] * 10
            + [{"metadata": {"dimension": "d", "difficulty": "adversarial"}}] * 10)
    rg = analyze(good)
    assert rg["verdict"] == "adequate", rg
    print("self-test OK")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluation dataset coverage check.")
    ap.add_argument("--dataset", type=Path, help="JSONL eval dataset.")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        _self_test()
        return 0
    if not args.dataset:
        ap.error("provide --dataset (or --self-test)")
    data = _read_jsonl(args.dataset)
    if not data:
        print("Empty dataset.", file=sys.stderr)
        return 1
    report = analyze(data)
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else render(report))
    return 0 if report["verdict"] == "adequate" else 2


if __name__ == "__main__":
    raise SystemExit(main())
