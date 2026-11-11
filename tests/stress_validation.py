"""Opt-in, reproducible randomized validation of the QT-Joint implementation.

Run from the project root (or any working directory):
    python tests/stress_validation.py
    python tests/stress_validation.py --output result/stress_validation.json

Defaults reproduce 90 coordinate-oracle cases and 60 joint-solver cases.
This script is deliberately not named test_*.py, so normal unittest discovery
does not run the optional randomized stress workload.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np


TEST_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIRECTORY.parent
for directory in (PROJECT_ROOT, TEST_DIRECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import qulow_joint as qt
from test_paper_alignment import NAMES, random_components, reference_coordinate_scan


def positive_integer(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("seed count must be positive")
    return parsed


def safe_numbers(value):
    """Keep failure reports valid JSON even when a failing run returns NaN/Inf."""
    if isinstance(value, dict):
        return {str(key): safe_numbers(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_numbers(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else str(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def run_oracle_cases(seed_count, seed_offset):
    candidates = (2, 3, 4, 6)
    delta = 0.03
    policies = ((False, 0.0), (False, 0.1), (True, 0.1))
    failures = []
    total = 0
    for seed in range(seed_offset, seed_offset + seed_count):
        for symmetric, clip_ratio in policies:
            total += 1
            case = {"seed": seed, "symmetric": symmetric, "clip_ratio": clip_ratio}
            try:
                target, factors, core = random_components(seed)
                masters = dict(zip(NAMES, tuple(factors) + (core,)))
                expected = reference_coordinate_scan(
                    target, masters, {name: max(candidates) for name in NAMES},
                    candidates, delta, sweeps=2,
                    symmetric=symmetric, clip_ratio=clip_ratio,
                )
                _, _, _, states = qt.select_quantizers_global(
                    target, factors, core, candidates, delta=delta,
                    coordinate_sweeps=2, symmetric=symmetric, clip_ratio=clip_ratio,
                )
                actual = {name: states[name].bit_width for name in NAMES}
                if actual != expected:
                    failures.append(dict(case, reason="bit_configuration_mismatch",
                                         expected=expected, actual=actual))
            except Exception as error:
                failures.append(dict(case, reason="exception", error_type=type(error).__name__,
                                     error=str(error)))
    return {
        "cases": total, "passed": total - len(failures), "failed": len(failures),
        "seed_count": seed_count, "seed_offset": seed_offset,
        "candidate_bits": list(candidates), "delta": delta, "coordinate_sweeps": 2,
        "policies": [{"symmetric": symmetric, "clip_ratio": clip_ratio}
                     for symmetric, clip_ratio in policies],
        "failures": failures,
    }


def run_solver_cases(seed_count, seed_offset):
    candidates = (2, 3, 5)
    delta = 0.01
    policies = ((False, 0.0), (False, 0.2), (True, 0.2))
    failures = []
    total = 0
    for seed in range(seed_offset, seed_offset + seed_count):
        for symmetric, clip_ratio in policies:
            total += 1
            case = {"seed": seed, "symmetric": symmetric, "clip_ratio": clip_ratio,
                    "use_skew_filter": bool(seed % 2)}
            try:
                data = np.random.default_rng(seed).normal(size=(5, 4, 3))
                result = qt.joint_lowrank_quantize(
                    data, ranks=(2, 2, 2), candidate_bits=candidates, delta=delta,
                    max_iter=5, learning_rate=10.0, clip_ratio=clip_ratio,
                    symmetric=symmetric, use_skew_filter=bool(seed % 2), verbose=False,
                )
                objectives = np.asarray([row["objective"] for row in result.history])
                final_objective = float(result.final_metrics["objective"])
                independent_bits = sum(
                    array.size * result.quantizers[name].bit_width
                    for name, array in zip(NAMES, tuple(result.master_factors) + (result.master_core,))
                )
                expected = float(np.sum((data - result.reconstructed) ** 2) + delta * independent_bits)
                reasons = []
                if not np.all(np.isfinite(objectives)) or not np.isfinite(final_objective):
                    reasons.append("nonfinite_objective")
                # Strict monotonicity checks have NO positive allowance for an increase.
                if np.any(np.diff(objectives) > 0.0):
                    reasons.append("accepted_history_objective_increased")
                if objectives.size == 0 or final_objective > objectives[-1]:
                    reasons.append("final_objective_increased")
                if result.model_bits != independent_bits:
                    reasons.append("ideal_bit_cost_mismatch")
                # Independent reduction order may differ by floating-point roundoff.
                if not np.isclose(final_objective, expected, rtol=1e-12, atol=1e-10):
                    reasons.append("reported_objective_units_or_value_mismatch")
                if reasons:
                    failures.append(dict(case, reasons=reasons, history=objectives.tolist(),
                                         reported_final_objective=final_objective,
                                         independently_evaluated_objective=expected,
                                         reported_model_bits=result.model_bits,
                                         independently_counted_model_bits=independent_bits))
            except Exception as error:
                failures.append(dict(case, reason="exception", error_type=type(error).__name__,
                                     error=str(error)))
    return {
        "cases": total, "passed": total - len(failures), "failed": len(failures),
        "seed_count": seed_count, "seed_offset": seed_offset,
        "shape": [5, 4, 3], "ranks": [2, 2, 2], "candidate_bits": list(candidates),
        "delta": delta, "learning_rate": 10.0, "max_iter": 5,
        "skew_filter_rule": "enabled for odd seeds",
        "policies": [{"symmetric": symmetric, "clip_ratio": clip_ratio}
                     for symmetric, clip_ratio in policies],
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--oracle-seeds", type=positive_integer, default=30)
    parser.add_argument("--solver-seeds", type=positive_integer, default=20)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    args = parser.parse_args()
    if args.seed_offset < 0:
        parser.error("--seed-offset must be non-negative")

    started = time.perf_counter()
    oracle = run_oracle_cases(args.oracle_seeds, args.seed_offset)
    solver = run_solver_cases(args.solver_seeds, args.seed_offset)
    total = oracle["cases"] + solver["cases"]
    failed = oracle["failed"] + solver["failed"]
    report = safe_numbers({
        "schema_version": 1,
        "status": "passed" if failed == 0 else "failed",
        "total_cases": total, "passed_cases": total - failed, "failed_cases": failed,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "elapsed_seconds": time.perf_counter() - started,
        "coordinate_oracle": oracle, "joint_solver": solver,
    })
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
