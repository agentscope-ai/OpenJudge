#!/usr/bin/env python3
"""Judge calibration report: does an automatic judge agree with human labels?

Standalone — needs only the Python standard library (uses numpy if available, else a
stdlib bootstrap). Works on any judge whose verdicts and human labels are pass/fail.

It computes the confusion matrix, TPR/TNR/accuracy/F1, Cohen's kappa, Gwet's AC1,
directional bias, bootstrap 95% CIs, an optional per-stratum breakdown, and a calibration
gate verdict (calibrated / not_calibrated / insufficient_evidence).

INPUT — paired judge vs human, one JSON object per line. Two accepted shapes:

  A) a single paired file (--pairs pairs.jsonl), rows like:
       {"id": "s1", "judge": "fail", "human": "fail", "stratum": "boundary"}
     ("judge"/"human" may be "pass"/"fail" or 1/0; "stratum" is optional)

  B) two files joined on id:
       --verdicts verdicts.jsonl  rows: {"id": "s1", "verdict": "fail", "stratum": "..."}
       --labels   labels.jsonl    rows: {"id": "s1", "label": "fail"}

POSITIVE CLASS: defaults to "fail" — in judge/error detection the thing you care about
catching is a failure, so TPR = fraction of real failures the judge catches.

USAGE
  python calibration.py --pairs pairs.jsonl
  python calibration.py --verdicts verdicts.jsonl --labels labels.jsonl --stratum-key stratum
  python calibration.py --pairs pairs.jsonl --json        # machine-readable output
  python calibration.py --self-test                       # run built-in checks

EXIT CODE: 0 if calibrated, 2 if not_calibrated/insufficient_evidence, 1 on usage error.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable

# Calibration gate thresholds (mirror 03-align-human/SKILL.md).
TPR_MIN = 0.8
TNR_MIN = 0.8
KAPPA_PHASE2_MIN = 0.6
N_PER_CLASS_MIN = 10
N_TOTAL_MIN = 50

try:  # numpy is optional; the stdlib path gives identical results, just slower.
    import numpy as _np
except Exception:  # pragma: no cover - exercised only when numpy is absent
    _np = None


def _norm(value: Any) -> str:
    """Normalize a verdict/label to 'pass' or 'fail'."""
    if isinstance(value, bool):
        return "pass" if value else "fail"
    if isinstance(value, (int, float)):
        return "pass" if value >= 0.5 else "fail"
    s = str(value).strip().lower()
    if s in {"pass", "p", "1", "true", "yes", "ok", "good"}:
        return "pass"
    if s in {"fail", "f", "0", "false", "no", "bad"}:
        return "fail"
    raise ValueError(f"Cannot interpret verdict/label: {value!r} (use pass/fail or 1/0)")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def load_pairs(
    pairs: Path | None,
    verdicts: Path | None,
    labels: Path | None,
    stratum_key: str = "stratum",
) -> list[dict[str, str]]:
    """Return a list of {'id','judge','human','stratum'} dicts."""
    out: list[dict[str, str]] = []
    if pairs is not None:
        for row in _read_jsonl(pairs):
            out.append({
                "id": str(row.get("id", len(out))),
                "judge": _norm(row["judge"]),
                "human": _norm(row["human"]),
                "stratum": str(row.get(stratum_key, "")) or "all",
            })
        return out

    if verdicts is None or labels is None:
        raise ValueError("Provide --pairs, or both --verdicts and --labels.")
    label_by_id = {str(r["id"]): r.get("label", r.get("human")) for r in _read_jsonl(labels)}
    for row in _read_jsonl(verdicts):
        rid = str(row["id"])
        if rid not in label_by_id:
            continue
        out.append({
            "id": rid,
            "judge": _norm(row.get("verdict", row.get("judge"))),
            "human": _norm(label_by_id[rid]),
            "stratum": str(row.get(stratum_key, "")) or "all",
        })
    return out


def confusion(pairs: list[dict[str, str]], positive: str = "fail") -> dict[str, int]:
    tp = fp = tn = fn = 0
    for p in pairs:
        judge_pos = p["judge"] == positive
        human_pos = p["human"] == positive
        if judge_pos and human_pos:
            tp += 1
        elif judge_pos and not human_pos:
            fp += 1
        elif not judge_pos and not human_pos:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def core_metrics(pairs: list[dict[str, str]], positive: str = "fail") -> dict[str, float]:
    c = confusion(pairs, positive)
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    tpr = _safe_div(tp, tp + fn)            # recall on the positive (caught failures)
    tnr = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    f1 = _safe_div(2 * precision * tpr, precision + tpr)
    accuracy = _safe_div(tp + tn, tp + fp + tn + fn)
    return {"tpr": tpr, "tnr": tnr, "precision": precision, "f1": f1, "accuracy": accuracy}


def cohens_kappa(pairs: list[dict[str, str]]) -> float:
    n = len(pairs)
    if n == 0:
        return 0.0
    p_o = _safe_div(sum(1 for p in pairs if p["judge"] == p["human"]), n)
    j_pass = _safe_div(sum(1 for p in pairs if p["judge"] == "pass"), n)
    h_pass = _safe_div(sum(1 for p in pairs if p["human"] == "pass"), n)
    p_e = j_pass * h_pass + (1 - j_pass) * (1 - h_pass)
    return _safe_div(p_o - p_e, 1 - p_e) if p_e != 1 else 1.0


def gwet_ac1(pairs: list[dict[str, str]]) -> float:
    """Robust to class imbalance (kappa's paradox)."""
    n = len(pairs)
    if n == 0:
        return 0.0
    p_o = _safe_div(sum(1 for p in pairs if p["judge"] == p["human"]), n)
    # mean proportion of "pass" across the two raters
    pi = (sum(1 for p in pairs if p["judge"] == "pass") + sum(1 for p in pairs if p["human"] == "pass")) / (2 * n)
    p_e = 2 * pi * (1 - pi)
    return _safe_div(p_o - p_e, 1 - p_e) if p_e != 1 else 1.0


def directional_bias(pairs: list[dict[str, str]], positive: str = "fail") -> float:
    """P(judge=fail | human=pass) - P(judge=pass | human=fail).

    > 0.1 : judge stricter than humans (over-flags). < -0.1 : more lenient.
    """
    neg = "pass" if positive == "fail" else "fail"
    human_neg = [p for p in pairs if p["human"] == neg]
    human_pos = [p for p in pairs if p["human"] == positive]
    over = _safe_div(sum(1 for p in human_neg if p["judge"] == positive), len(human_neg))
    under = _safe_div(sum(1 for p in human_pos if p["judge"] == neg), len(human_pos))
    return over - under


def bootstrap_ci(
    pairs: list[dict[str, str]],
    metric_fn: Callable[[list[dict[str, str]]], float],
    n_iter: int = 1000,
    ci: float = 95.0,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (mean, lo, hi) for metric_fn over bootstrap resamples."""
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0, 0.0
    lo_pct, hi_pct = (100 - ci) / 2, 100 - (100 - ci) / 2
    if _np is not None:
        rng = _np.random.default_rng(seed)
        vals = [metric_fn([pairs[i] for i in rng.integers(0, n, n)]) for _ in range(n_iter)]
        return float(_np.mean(vals)), float(_np.percentile(vals, lo_pct)), float(_np.percentile(vals, hi_pct))
    rnd = random.Random(seed)
    vals = sorted(metric_fn([pairs[rnd.randrange(n)] for _ in range(n)]) for _ in range(n_iter))
    mean = sum(vals) / len(vals)

    def pct(p: float) -> float:
        k = (len(vals) - 1) * p / 100
        lo_i, hi_i = int(k), min(int(k) + 1, len(vals) - 1)
        return vals[lo_i] + (vals[hi_i] - vals[lo_i]) * (k - lo_i)

    return mean, pct(lo_pct), pct(hi_pct)


def per_stratum(pairs: list[dict[str, str]], positive: str = "fail") -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    strata = sorted({p["stratum"] for p in pairs})
    for s in strata:
        subset = [p for p in pairs if p["stratum"] == s]
        if len(subset) >= N_PER_CLASS_MIN:
            m = core_metrics(subset, positive)
            out[s] = {"n": len(subset), "tpr": m["tpr"], "tnr": m["tnr"]}
    return out


def calibration_gate(pairs: list[dict[str, str]], metrics: dict[str, float], kappa: float,
                     positive: str = "fail") -> dict[str, Any]:
    """Apply the SKILL.md hard gate. Returns verdict + blocking reasons."""
    n = len(pairs)
    n_pos = sum(1 for p in pairs if p["human"] == positive)
    n_neg = n - n_pos
    reasons: list[str] = []
    if n < N_TOTAL_MIN:
        reasons.append(f"only {n} labels (need >= {N_TOTAL_MIN})")
    if min(n_pos, n_neg) < N_PER_CLASS_MIN:
        reasons.append(f"min class count {min(n_pos, n_neg)} (need >= {N_PER_CLASS_MIN} per class)")
    if reasons:
        return {"verdict": "insufficient_evidence", "blocking": reasons,
                "next": f"collect more labels: aim for >= {N_TOTAL_MIN} total, >= {N_PER_CLASS_MIN} per class"}
    not_ready = []
    if metrics["tpr"] < TPR_MIN:
        not_ready.append(f"TPR {metrics['tpr']:.2f} < {TPR_MIN}")
    if metrics["tnr"] < TNR_MIN:
        not_ready.append(f"TNR {metrics['tnr']:.2f} < {TNR_MIN}")
    if not_ready:
        return {"verdict": "not_calibrated", "blocking": not_ready,
                "next": "refine the judge prompt (add borderline few-shot) then re-measure"}
    phase = 2 if kappa >= KAPPA_PHASE2_MIN else 1
    return {"verdict": "calibrated", "blocking": [],
            "human_reduction_phase": phase,
            "next": "phase 2 (assisted) ok" if phase == 2 else "calibrated but kappa<0.6: keep humans primary"}


def analyze(pairs: list[dict[str, str]], positive: str = "fail", n_iter: int = 1000) -> dict[str, Any]:
    metrics = core_metrics(pairs, positive)
    kappa = cohens_kappa(pairs)
    ac1 = gwet_ac1(pairs)
    bias = directional_bias(pairs, positive)
    tpr_ci = bootstrap_ci(pairs, lambda s: core_metrics(s, positive)["tpr"], n_iter)
    tnr_ci = bootstrap_ci(pairs, lambda s: core_metrics(s, positive)["tnr"], n_iter)
    gate = calibration_gate(pairs, metrics, kappa, positive)
    return {
        "n": len(pairs),
        "positive_class": positive,
        "confusion": confusion(pairs, positive),
        "metrics": metrics,
        "kappa": kappa,
        "gwet_ac1": ac1,
        "kappa_ac1_gap": abs(kappa - ac1),
        "bias": bias,
        "tpr_ci95": [tpr_ci[1], tpr_ci[2]],
        "tnr_ci95": [tnr_ci[1], tnr_ci[2]],
        "per_stratum": per_stratum(pairs, positive),
        "gate": gate,
    }


def render(report: dict[str, Any]) -> str:
    m, g = report["metrics"], report["gate"]
    lines = [
        f"Calibration report  (n={report['n']}, positive='{report['positive_class']}')",
        f"  confusion: {report['confusion']}",
        f"  TPR={m['tpr']:.2f} (95% CI {report['tpr_ci95'][0]:.2f}-{report['tpr_ci95'][1]:.2f})  "
        f"TNR={m['tnr']:.2f} (95% CI {report['tnr_ci95'][0]:.2f}-{report['tnr_ci95'][1]:.2f})",
        f"  F1={m['f1']:.2f}  accuracy={m['accuracy']:.2f}",
        f"  kappa={report['kappa']:.2f}  Gwet's AC1={report['gwet_ac1']:.2f}"
        + ("  [gap>0.15: class imbalance — trust AC1]" if report["kappa_ac1_gap"] > 0.15 else ""),
        f"  bias={report['bias']:+.2f} "
        + ("(stricter than humans)" if report["bias"] > 0.1 else
           "(more lenient than humans)" if report["bias"] < -0.1 else "(no significant bias)"),
    ]
    if report["per_stratum"]:
        lines.append("  per-stratum:")
        for s, v in report["per_stratum"].items():
            lines.append(f"    {s:12s} n={int(v['n']):3d}  TPR={v['tpr']:.2f}  TNR={v['tnr']:.2f}")
    lines.append(f"  VERDICT: {g['verdict']}")
    if g["blocking"]:
        lines.append("    blocking: " + "; ".join(g["blocking"]))
    lines.append("    next: " + g["next"])
    return "\n".join(lines)


def _self_test() -> None:
    # A strong judge over >=50 labels: catches 27/30 failures, clears 28/30 passes.
    pairs = []
    for i in range(30):
        pairs.append({"id": f"f{i}", "judge": "fail" if i < 27 else "pass", "human": "fail", "stratum": "boundary"})
    for i in range(30):
        pairs.append({"id": f"p{i}", "judge": "pass" if i < 28 else "fail", "human": "pass", "stratum": "easy"})
    r = analyze(pairs, n_iter=200)
    assert r["confusion"] == {"tp": 27, "fp": 2, "tn": 28, "fn": 3}, r["confusion"]
    assert abs(r["metrics"]["tpr"] - 0.9) < 1e-9, r["metrics"]["tpr"]
    assert abs(r["metrics"]["tnr"] - 28 / 30) < 1e-9, r["metrics"]["tnr"]
    assert r["gate"]["verdict"] == "calibrated", r["gate"]
    # Perfect agreement → kappa == 1.
    perfect = [{"id": str(i), "judge": "fail", "human": "fail", "stratum": "all"} for i in range(10)]
    perfect += [{"id": str(i + 10), "judge": "pass", "human": "pass", "stratum": "all"} for i in range(10)]
    assert abs(cohens_kappa(perfect) - 1.0) < 1e-9
    # Too few labels → insufficient_evidence.
    few = [{"id": str(i), "judge": "fail", "human": "fail", "stratum": "all"} for i in range(18)]
    assert analyze(few, n_iter=50)["gate"]["verdict"] == "insufficient_evidence"
    # Kappa paradox: skewed marginals + some disagreement → high agreement (0.90) but
    # kappa paradoxically low while Gwet's AC1 stays high.
    imbal = [{"id": f"a{i}", "judge": "pass", "human": "pass", "stratum": "all"} for i in range(85)]
    imbal += [{"id": f"b{i}", "judge": "fail", "human": "fail", "stratum": "all"} for i in range(5)]
    imbal += [{"id": f"c{i}", "judge": "pass", "human": "fail", "stratum": "all"} for i in range(5)]
    imbal += [{"id": f"d{i}", "judge": "fail", "human": "pass", "stratum": "all"} for i in range(5)]
    assert gwet_ac1(imbal) > cohens_kappa(imbal) + 0.3, (gwet_ac1(imbal), cohens_kappa(imbal))
    print("self-test OK")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Judge calibration report (TPR/TNR/kappa/AC1/CI/gate).")
    ap.add_argument("--pairs", type=Path, help="JSONL with id,judge,human[,stratum].")
    ap.add_argument("--verdicts", type=Path, help="JSONL with id,verdict[,stratum].")
    ap.add_argument("--labels", type=Path, help="JSONL with id,label.")
    ap.add_argument("--positive", default="fail", choices=["fail", "pass"],
                    help="Positive class for TPR (default: fail = caught problems).")
    ap.add_argument("--stratum-key", default="stratum", help="Field name for stratum (default: stratum).")
    ap.add_argument("--n-iter", type=int, default=1000, help="Bootstrap iterations.")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of a text report.")
    ap.add_argument("--self-test", action="store_true", help="Run built-in checks and exit.")
    args = ap.parse_args(argv)

    if args.self_test:
        _self_test()
        return 0
    if not args.pairs and not (args.verdicts and args.labels):
        ap.error("provide --pairs, or both --verdicts and --labels (or --self-test)")

    pairs = load_pairs(args.pairs, args.verdicts, args.labels, args.stratum_key)
    if not pairs:
        print("No paired samples found (check ids match).", file=sys.stderr)
        return 1
    report = analyze(pairs, positive=args.positive, n_iter=args.n_iter)
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else render(report))
    return 0 if report["gate"]["verdict"] == "calibrated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
