"""Recompute three summaries from retained historical records; run no experiments."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("PingData", "PlanetLab", "TaxiBJ_in", "TaxiBJ_out", "FT-AED", "UCSD-Ped2")
OLD_BITS = (2, 3, 4, 5, 6, 7, 8, 10, 12, 16)
FULL_BITS = tuple(range(2, 33))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def retained(rows):
    return [row for row in rows if row.get("dataset") in DATASETS]


def main_qt(rows, manifest, methods=("QT-Joint",)):
    return [row for row in rows if row.get("method") in methods
            and row.get("dataset") in DATASETS
            and row.get("seed", 7) == 7 and "_s7_" in row.get("run_id", "")
            and row["run_id"].endswith("_auto")
            and row.get("shape") == manifest[row["dataset"]]["shape"]]


def maximum(rows, key, default=None):
    return max((row[key] for row in rows if isinstance(row.get(key), (int, float))), default=default)


def minimum_errors(rows):
    return {name: min(row["relative_error"] for row in rows if row["dataset"] == name)
            for name in DATASETS if any(row["dataset"] == name for row in rows)}


def fraction(rows, key):
    values = [row[key] for row in rows if key in row]
    return sum(bool(value) for value in values) / len(values) if values else None


def metadata(sources):
    return {
        "summary_status": "Recomputed from retained historical records; this script runs no experiments or tests.",
        "summarized_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": list(DATASETS),
        "main_QT_selection": "seed 7, run_id ending _auto, method QT-Joint for its stated bit set, full manifest shape",
        "source_files": sources,
        "mechanism_validation_status": "Historical numerical validation records; separate from dataset performance measurements.",
    }


def historical_tests(path):
    path = Path(path)
    if path.suffix == ".json":
        result = read(path)
        return {"evidence_status": "historical", "source": str(path.relative_to(ROOT)), **result}
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    counts = re.findall(r"Ran (\d+) tests?", text)
    return {"evidence_status": "historical", "source": str(path.relative_to(ROOT)),
            "tests": int(counts[-1]) if counts else None, "passed": bool(counts and re.search(r"\bOK\b", text))}


def mechanism_summary(folder):
    candidates = read(folder / "scoring_candidates.json")
    rankings = read(folder / "scoring_rankings.json")
    timings = read(folder / "scoring_timing.json")
    grids = read(folder / "exhaustive.json")
    unchecked = read(folder / "no_acceptance.json")
    cases = {row["case"]: row["combinations"] for row in grids}
    speedups = [row["speedup"] for row in timings]
    summary = {
        "scoring_candidate_count": len(candidates),
        "scoring_max_absolute_difference": maximum(candidates, "abs_difference"),
        "scoring_max_energy_normalized_difference": maximum(candidates, "energy_normalized_difference"),
        "scoring_best_candidate_agreement": fraction(rankings, "argmin_equal"),
        "scoring_strict_order_agreement": min((row["strict_order_agreement"] for row in rankings
                                               if isinstance(row.get("strict_order_agreement"), (int, float))), default=None),
        "scoring_min_speedup": min(speedups, default=None),
        "scoring_max_speedup": max(speedups, default=None),
        "scoring_scan_bits_equal": all(row["bits_equal_every_repeat"] for row in timings),
        "full_enumeration_cases": len(cases),
        "combinations_per_grid": next(iter(set(cases.values()))) if len(set(cases.values())) == 1 else None,
        "total_enumerated_combinations": sum(cases.values()),
        "exhaustive_max_coordinate_gap": maximum(grids, "coordinate_gap"),
        "exhaustive_max_relative_global_gap": maximum(grids, "relative_global_gap"),
        "exhaustive_non_global_endpoints": sum(row["relative_global_gap"] > 1e-10 for row in grids),
        "grid_oracle_max_abs_difference": maximum(grids, "oracle_max_abs_validation_difference"),
        "no_acceptance_increases": sum(bool(row["unchecked_increased"]) for row in unchecked),
        "no_acceptance_cases": len(unchecked),
        "scoring_objective_argmin_agreement": fraction(rankings, "objective_argmin_equal"),
        "scoring_max_objective_best_gap": maximum(rankings, "objective_best_gap"),
    }
    if any("best_sse_within_energy_tolerance" in row for row in rankings):
        summary["scoring_tolerance_sse_agreement"] = all(
            row["best_sse_within_energy_tolerance"] for row in rankings
            if "best_sse_within_energy_tolerance" in row)
    return {key: value for key, value in summary.items() if value is not None}


def audit_summary(audits):
    return {
        "audited_records": len(audits),
        "history_objective_increases": sum(row.get("objective_history_increases", row.get("outer_objective_increases", 0)) for row in audits),
        "parameter_sse_increases": sum(row.get("parameter_sse_increases", 0) for row in audits),
        "coordinate_objective_increases": sum(row.get("coordinate_objective_increases", 0) for row in audits),
        "maximum_objective_unit_difference": max((abs(value) for row in audits
            for value in [row.get("objective_units_difference", row.get("objective_unit_difference"))]
            if isinstance(value, (int, float))), default=None),
    }


def summarize_old(result_root):
    folder = result_root / "evaluation"
    records = retained(read(folder / "all_records.json"))
    comparison = retained(read(folder / "main_records.json"))
    manifest = read(folder / "dataset_manifest.json")
    main = main_qt(records, manifest)
    ids = {row["run_id"] for row in records}
    audits = [row for row in read(folder / "theory_audit.json") if row["run_id"] in ids]
    summary = metadata(["all_records.json", "main_records.json", "dataset_manifest.json", "theory_audit.json", "mechanisms/*.json"])
    summary.update(record_count=len(records), main_record_count=len(comparison), main_QT_configurations=len(main),
                   candidate_bits=list(OLD_BITS), methods=sorted({row["method"] for row in comparison}),
                   budgets=sorted({row["budget"] for row in read(folder / "budget_results.json")}),
                   regression_tests=historical_tests(folder / "unit_tests_py312.log"),
                   stress_validation={"evidence_status": "historical", **read(folder / "stress_validation.json")},
                   maximum_reload_difference=maximum(records, "reload_max_abs_difference"),
                   minimum_QT_error_by_dataset=minimum_errors(main))
    summary.update(audit_summary(audits))
    summary.update(mechanism_summary(folder / "mechanisms"))
    summary["max_iteration_records"] = [
        {key: row[key] for key in ("run_id", "dataset", "relative_error", "cr")}
        for row in main if row.get("search", {}).get("stop_reason") == "max_iter"]
    summary["main_configurations_by_dataset"] = dict(Counter(row["dataset"] for row in main))
    assert len(main) == len(DATASETS) * 8
    write(folder / "analysis_summary.json", summary)
    return summary


def summarize_full(result_root):
    folder = result_root / "evaluation_H2_32"
    records = retained(read(folder / "all_new_records.json"))
    combined = retained(read(folder / "combined_main_records.json"))
    reused = retained(read(folder / "reused_references.json"))
    pairs = retained(read(folder / "paired_old_new.json"))
    manifest = read(folder / "dataset_manifest.json")
    main = main_qt(records, manifest)
    main_ids = {row["run_id"] for row in main}
    main_pairs = [row for row in pairs if row["run_id"] in main_ids]
    comparable = [row for row in main_pairs if row["ranks_same"] and row["delta_same"]]
    ids = {row["run_id"] for row in records}
    audits = [row for row in read(folder / "theory_audit.json") if row["run_id"] in ids]
    histogram = Counter(bit for row in main for bit in row["bits"].values())
    validation = read(folder / "H31_validation.json")
    summary = metadata(["all_new_records.json", "combined_main_records.json", "reused_references.json", "paired_old_new.json", "dataset_manifest.json", "theory_audit.json", "mechanisms/*.json"])
    summary.update(candidate_bits=list(FULL_BITS), previous_candidate_bits=list(OLD_BITS),
                   fresh_configurations=len(records), fresh_main_configurations=len(main),
                   reused_reference_configurations=len(reused), combined_main_configurations=len(combined),
                   paired_configurations=len(pairs), paired_main_configurations=len(main_pairs),
                   main_same_parameter_configurations=len(comparable),
                   methods=sorted({row["method"] for row in combined}),
                   budgets=sorted({row["budget"] for row in read(folder / "budget_results.json")}),
                   regression_tests=historical_tests(folder / "regression_tests.json"),
                   H31_validation={"evidence_status": "historical", "packing_widths": len(validation["packing_cases"]),
                                   "coordinate_reference_cases": len(validation["full31_coordinate_oracle_cases"]),
                                   "passed": validation["passed"]},
                   minimum_main_error=minimum_errors(main),
                   final_bit_histogram_main={str(bit): histogram[bit] for bit in FULL_BITS},
                   main_components_above_16_bits=sum(count for bit, count in histogram.items() if bit > 16),
                   main_new_intermediate_bits=sum(count for bit, count in histogram.items() if bit not in OLD_BITS),
                   main_same_parameter_file_smaller=sum(row["new_file_bytes"] < row["old_file_bytes"] for row in comparable),
                   main_same_parameter_file_larger=sum(row["new_file_bytes"] > row["old_file_bytes"] for row in comparable),
                   main_same_parameter_file_equal=sum(row["new_file_bytes"] == row["old_file_bytes"] for row in comparable),
                   main_same_parameter_objective_larger=sum(row["new_objective"] > row["old_objective"] + 1e-10 * max(1.0, abs(row["old_objective"])) for row in comparable),
                   objective_comparison_relative_tolerance=1e-10,
                   maximum_reload_difference=maximum(records, "reload_max_abs_difference"))
    summary.update(audit_summary(audits))
    summary["objective_increases"] = summary.pop("history_objective_increases")
    summary["max_objective_unit_difference"] = summary.pop("maximum_objective_unit_difference")
    mechanism = mechanism_summary(folder / "mechanisms")
    mechanism["scoring_candidates"] = mechanism.pop("scoring_candidate_count")
    mechanism["scoring_literal_sse_argmin_agreement"] = mechanism.pop("scoring_best_candidate_agreement")
    mechanism["scoring_min_strict_order_agreement"] = mechanism.pop("scoring_strict_order_agreement")
    mechanism["exhaustive_max_global_relative_gap"] = mechanism.pop("exhaustive_max_relative_global_gap")
    summary.update(mechanism)
    summary["main_configurations_by_dataset"] = dict(Counter(row["dataset"] for row in main))
    assert len(main) == len(DATASETS) * 8 and sum(histogram.values()) == len(main) * 4
    write(folder / "analysis_summary.json", summary)
    return summary


def summarize_paper(result_root):
    folder = result_root / "paper_comparison"
    old = read(folder / "comparison_summary.json")
    records = retained(read(folder / "comparison_records.json"))
    qt = main_qt(records, read(result_root / "evaluation" / "dataset_manifest.json"))
    tkqt = [row for row in records if row["method"] == "TKQT-independent"]
    fits = []
    for fit in read(folder / "fit_diagnostics.json"):
        names = [name for name in DATASETS if fit["fit_id"].startswith(name + "_")]
        if names:
            name = names[0]
            fits.append({**fit, "dataset": name, "label": fit["fit_id"][len(name) + 1:].rsplit("_r", 1)[0]})
    summary = metadata(["comparison_records.json", "fit_diagnostics.json", "reference_tests.json"])
    summary.update({key: old[key] for key in ("paper_title", "paper_method", "doi", "reference_type", "limitations")})
    summary.update(paper_sha256=hashlib.sha256((folder / "source_paper.pdf").read_bytes()).hexdigest(),
                   tkqt_configurations=len(tkqt), reused_qt_configurations=len(qt),
                   budgets=sorted({row["budget"] for row in read(folder / "budget_results.json")}),
                   default_records=[row["run_id"] for row in tkqt if row.get("paper_default_precision") and row.get("native_rank_default")],
                   decompositions=len(fits), decompositions_converged=sum(bool(fit["converged"]) for fit in fits),
                   unconverged_fits=[fit for fit in fits if not fit["converged"]],
                   reference_tests=historical_tests(folder / "reference_tests.json"),
                   qt_candidate_bits=list(OLD_BITS), minimum_QT_error_by_dataset=minimum_errors(qt),
                   minimum_TKQT_error_by_dataset=minimum_errors(tkqt))
    assert len(qt) == len(DATASETS) * 8
    write(folder / "comparison_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=ROOT / "result")
    args = parser.parse_args()
    old = summarize_old(args.result_root)
    full = summarize_full(args.result_root)
    paper = summarize_paper(args.result_root)
    print(json.dumps({"old_records": old["record_count"], "old_main_QT": old["main_QT_configurations"],
                      "full_records": full["fresh_configurations"], "full_main_QT": full["fresh_main_configurations"],
                      "TKQT_records": paper["tkqt_configurations"], "fits": paper["decompositions"],
                      "converged_fits": paper["decompositions_converged"]}, indent=2))


if __name__ == "__main__":
    main()
