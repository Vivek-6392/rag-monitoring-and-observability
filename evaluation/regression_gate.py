"""
Regression gate — compares current run metrics against thresholds.
Used both in CI (pytest) and in the Streamlit evaluation page.
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from typing import Optional

# ── Configurable thresholds ──────────────────────────────────────
THRESHOLDS = {
    "p95_latency_ms":        3000,   # fail if p95 > 3 s
    "avg_quality":           0.60,   # fail if avg quality < 0.60
    "avg_faithfulness":      0.55,
    "avg_answer_relevancy":  0.55,
    "avg_context_precision": 0.55,
}

# How much degradation vs baseline triggers failure (fraction)
REGRESSION_TOLERANCE = {
    "p95_latency_ms":  0.20,   # +20% latency allowed
    "avg_quality":    -0.10,   # -10% quality allowed
}


@dataclass
class GateResult:
    passed: bool
    checks: list[dict]
    summary: str
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        import numpy as np

        def sanitize(obj):
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize(v) for v in obj]
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            else:
                return obj

        return sanitize(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def run_gate(
    *,
    p95_latency_ms: Optional[float] = None,
    avg_quality: Optional[float] = None,
    avg_faithfulness: Optional[float] = None,
    avg_answer_relevancy: Optional[float] = None,
    avg_context_precision: Optional[float] = None,
    baseline: Optional[dict] = None,
) -> GateResult:
    """
    Run all threshold checks.
    Optionally compare against a `baseline` dict with the same keys.
    """
    checks = []
    overall_pass = True

    def _check(name: str, value: Optional[float], threshold: float, higher_is_better: bool):
        nonlocal overall_pass
        if value is None:
            checks.append({"name": name, "value": None, "threshold": threshold, "passed": True, "note": "skipped"})
            return
        if higher_is_better:
            passed = value >= threshold
        else:
            passed = value <= threshold
        if not passed:
            overall_pass = False
        checks.append({
            "name": name,
            "value": round(value, 4),
            "threshold": threshold,
            "passed": passed,
            "higher_is_better": higher_is_better,
        })

    _check("p95_latency_ms",        p95_latency_ms,        THRESHOLDS["p95_latency_ms"],        higher_is_better=False)
    _check("avg_quality",           avg_quality,           THRESHOLDS["avg_quality"],           higher_is_better=True)
    _check("avg_faithfulness",      avg_faithfulness,      THRESHOLDS["avg_faithfulness"],      higher_is_better=True)
    _check("avg_answer_relevancy",  avg_answer_relevancy,  THRESHOLDS["avg_answer_relevancy"],  higher_is_better=True)
    _check("avg_context_precision", avg_context_precision, THRESHOLDS["avg_context_precision"], higher_is_better=True)

    # Regression check against baseline
    if baseline:
        for metric, tolerance in REGRESSION_TOLERANCE.items():
            curr = locals().get(metric)
            base = baseline.get(metric)
            if curr is None or base is None or base == 0:
                continue
            change = (curr - base) / abs(base)
            if tolerance < 0:
                # quality metric: regression = drop below tolerance
                regression = change < tolerance
            else:
                # latency metric: regression = rise above tolerance
                regression = change > tolerance
            if regression:
                overall_pass = False
            checks.append({
                "name": f"{metric}_regression",
                "baseline": round(base, 4),
                "current": round(curr, 4),
                "change_pct": round(change * 100, 1),
                "allowed_pct": round(tolerance * 100, 1),
                "passed": not regression,
            })

    failed = [c for c in checks if not c.get("passed", True)]
    summary = "✅ All checks passed." if overall_pass else f"❌ {len(failed)} check(s) failed: {[c['name'] for c in failed]}"

    return GateResult(passed=overall_pass, checks=checks, summary=summary)