"""Command-line runner for the QT-Joint specification dated 2026-11-10.

The objective uses SSE and ideal component bits; actual compression ratios
include every saved file required by the decoder, including normalization.
"""
import argparse
import json
import os
import time
import warnings

import numpy as np

from qulow_joint import (
    configure_utf8_output, joint_lowrank_quantize, load_mat_tensor,
    load_matrix_sequence, load_quantized_model, normalize_minmax,
    save_history, tucker_reconstruct, write_csv,
)

PAPER_BITS = tuple(range(2, 33))
COMPONENTS = ("G", "A", "B", "C")


def parse_ranks(value):
    if value is None or str(value).lower() == "auto":
        return None
    try:
        ranks = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ranks must contain integers") from exc
    if len(ranks) != 3 or any(rank < 1 for rank in ranks):
        raise argparse.ArgumentTypeError("ranks must look like 8,8,8, or auto")
    return ranks


def parse_bits(value):
    try:
        bits = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bits must contain integers") from exc
    if not bits or any(bit < 2 or bit > 32 for bit in bits):
        raise argparse.ArgumentTypeError("bits must be a nonempty list in [2, 32]")
    return tuple(sorted(set(bits)))


def parse_component_values(value, converter, label):
    """One setting for all components, or a complete G,A,B,C mapping."""
    if "=" not in value:
        try:
            return converter(value.strip())
        except (ValueError, KeyError) as exc:
            raise argparse.ArgumentTypeError(f"invalid {label}: {value}") from exc
    result = {}
    try:
        for item in value.split(","):
            key, setting = item.split("=", 1)
            key = key.strip().upper()
            if key not in COMPONENTS or key in result:
                raise ValueError("unknown or repeated component")
            result[key] = converter(setting.strip())
    except (ValueError, KeyError) as exc:
        raise argparse.ArgumentTypeError(
            f"{label} must be scalar or G=...,A=...,B=...,C=..."
        ) from exc
    if set(result) != set(COMPONENTS):
        raise argparse.ArgumentTypeError(f"{label} mapping must specify G,A,B,C")
    return result


def parse_clip_ratio(value):
    def convert(item):
        number = float(item)
        if not np.isfinite(number) or not 0 <= number < 1:
            raise ValueError("c must lie in [0, 1)")
        return number
    return parse_component_values(value, convert, "clip ratio c")


def parse_quantization_type(value):
    return parse_component_values(
        value, lambda item: {"symmetric": True, "asymmetric": False}[item.lower()],
        "quantization type",
    )


def parse_skew_thresholds(value):
    try:
        thresholds = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("skew thresholds must be numeric") from exc
    if (len(thresholds) != 3
        or not all(np.isfinite(number) for number in thresholds)
        or not 0 <= thresholds[0] < thresholds[1] < thresholds[2]):
        raise argparse.ArgumentTypeError("use 0 <= gamma1 < gamma2 < gamma3")
    return thresholds


def add_solver_arguments(parser):
    parser.add_argument("--ranks", type=parse_ranks, default=None,
                        help="fixed dimensions, e.g. 8,8,8; default: joint HOSVD proxy search")
    parser.add_argument("--bits", type=parse_bits, default=PAPER_BITS,
                        help="finite bit set; current default is every integer from 2 through 32")
    objective = parser.add_mutually_exclusive_group()
    objective.add_argument("--delta", type=float, default=None,
                           help="SSE per ideal stored bit; core default: 1e-4")
    objective.add_argument("--rate-weight", type=float, default=None,
                           help="deprecated old lambda; maps to delta=lambda/32")
    parser.add_argument("--clip-ratio", "--clip-fraction", dest="clip_ratio",
                        type=parse_clip_ratio, default=0.0,
                        help="fixed total two-sided c in [0,1), or G=0,A=0,B=0,C=0")
    parser.add_argument("--quantization-type", type=parse_quantization_type,
                        default=False,
                        help="fixed asymmetric (default), symmetric, or named mapping")
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--parameter-steps", type=int, default=1)
    parser.add_argument("--max-backtracks", type=int, default=20)
    parser.add_argument("--backtrack-factor", type=float, default=0.5)
    parser.add_argument("--rank-max", type=int, default=64,
                        help="candidate rank cap; proxy optimality is limited to this cap")
    parser.add_argument("--quantizer-update-interval", type=int, default=1)
    parser.add_argument("--quantizer-sweeps", type=int, default=2)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-absolute-improvement", type=float, default=1e-10)
    parser.add_argument("--gradient-clip-norm", type=float, default=None,
                        help="optional positive proposal-gradient norm cap")
    parser.add_argument("--skew-filter", action="store_true",
                        help="optional heuristic bit screening, disabled by default")
    parser.add_argument("--skew-thresholds", type=parse_skew_thresholds,
                        default=(0.5, 1.0, 2.0))
    full_scan = parser.add_mutually_exclusive_group()
    full_scan.add_argument("--full-scan-at-end", "--final-full-scan", dest="full_scan_at_end",
                           action="store_true", default=True)
    full_scan.add_argument("--no-full-scan-at-end", "--no-final-full-scan", dest="full_scan_at_end",
                           action="store_false")
    parser.add_argument("--seed", type=int, default=7)
    # Recognize obsolete flags only to give a precise migration error.
    parser.add_argument("--l2-weight", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--rank-error-weight", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--rank-storage-weight", type=float, default=None, help=argparse.SUPPRESS)


def add_normalization_argument(parser):
    parser.add_argument("--normalization", choices=("none", "scale", "global", "axis0"),
                        default="none",
                        help="none (default); scale=max-abs; global/axis0=legacy min-max")


def solver_kwargs(args):
    if args.l2_weight != 0:
        raise ValueError("nonzero --l2-weight changes the paper objective; remove it")
    if args.rank_error_weight is not None or args.rank_storage_weight is not None:
        raise ValueError("separate old rank weights are unsupported; use the common --delta")
    result = dict(
        ranks=args.ranks, candidate_bits=args.bits, delta=args.delta,
        clip_ratio=args.clip_ratio, symmetric=args.quantization_type,
        max_iter=args.max_iter, learning_rate=args.learning_rate,
        parameter_steps=args.parameter_steps, max_backtracks=args.max_backtracks,
        backtrack_factor=args.backtrack_factor, rank_max=args.rank_max,
        quantizer_update_interval=args.quantizer_update_interval,
        quantizer_coordinate_sweeps=args.quantizer_sweeps, patience=args.patience,
        min_absolute_improvement=args.min_absolute_improvement,
        gradient_clip_norm=args.gradient_clip_norm, use_skew_filter=args.skew_filter,
        skew_thresholds=args.skew_thresholds, full_scan_at_end=args.full_scan_at_end,
        seed=args.seed,
    )
    if args.rate_weight is not None:
        warnings.warn(
            "--rate-weight is deprecated: old lambda maps to delta=lambda/32; "
            "old clipping/rank/storage rules are not retained.", UserWarning,
        )
        result["rate_weight"] = args.rate_weight
    return result


def load_input(args):
    if args.mat:
        return load_mat_tensor(args.mat, args.mat_key)
    if args.matrix_sequence:
        directory, prefix, count = args.matrix_sequence
        return load_matrix_sequence(directory, prefix, int(count))
    raise ValueError("choose --mat or --matrix-sequence")


def normalize_input(data, strategy):
    if np.iscomplexobj(data):
        raise ValueError("input must be real-valued")
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 3 or not data.size or not np.all(np.isfinite(data)):
        raise ValueError("input must be a nonempty finite 3D tensor")
    if strategy == "none":
        return data, np.asarray(0.0), np.asarray(1.0)
    if strategy == "scale":
        maximum = float(np.max(np.abs(data)))
        scale = maximum if maximum > 0 else 1.0
        return data / scale, np.asarray(0.0), np.asarray(scale)
    if strategy == "global":
        return normalize_minmax(data, axis=None)
    if strategy == "axis0":
        return normalize_minmax(data, axis=0)
    raise ValueError(f"unknown normalization strategy: {strategy}")


def error_metrics(reference, estimate):
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if reference.shape != estimate.shape:
        raise ValueError("error metric arrays must have identical shapes")
    residual = reference - estimate
    sse = float(np.sum(residual ** 2))
    norm = float(np.linalg.norm(reference.reshape(-1)))
    relative = np.sqrt(sse) / norm if norm > 0 else (0.0 if sse == 0 else None)
    return {
        "sse": sse, "mse": sse / reference.size,
        "rmse": float(np.sqrt(sse / reference.size)),
        "mae": float(np.mean(np.abs(residual))),
        "max_abs_error": float(np.max(np.abs(residual))),
        "relative_f_error": None if relative is None else float(relative),
        "reference_f_norm": norm,
    }


def write_json(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"not JSON serializable: {type(item).__name__}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2,
                  default=convert, allow_nan=False)
        handle.write("\n")


def export_and_verify(result, directory, minimum, scale, strategy, prefix="joint_qulow"):
    """Save and reload decoder inputs; diagnostic files are excluded from rate."""
    os.makedirs(directory, exist_ok=True)
    model_path = os.path.join(directory, f"{prefix}_model.npz")
    normalization_path = os.path.join(directory, "normalization.npz")
    start = time.perf_counter()
    result.save_npz(model_path)
    np.savez_compressed(normalization_path, strategy=np.asarray(strategy),
                        minimum=np.asarray(minimum, dtype=np.float64),
                        scale=np.asarray(scale, dtype=np.float64))
    encoding_seconds = time.perf_counter() - start
    start = time.perf_counter()
    factors, core, _, metadata = load_quantized_model(model_path)
    reconstructed = tucker_reconstruct(core, factors)
    with np.load(normalization_path, allow_pickle=False) as normalization:
        raw_reconstruction = reconstructed * normalization["scale"] + normalization["minimum"]
    decoding_seconds = time.perf_counter() - start
    if not np.allclose(reconstructed, result.reconstructed, rtol=1e-12, atol=1e-14):
        raise RuntimeError("packed model does not reproduce the evaluated reconstruction")
    model_bytes = os.path.getsize(model_path)
    normalization_bytes = os.path.getsize(normalization_path)
    original_bytes = int(np.prod(result.original_shape)) * 4
    return reconstructed, raw_reconstruction, {
        "model_file": os.path.basename(model_path),
        "normalization_file": os.path.basename(normalization_path),
        "format_version": metadata["format_version"],
        "original_float32_bytes": original_bytes,
        "ideal_model_bits": int(result.model_bits),
        "ideal_model_compression_ratio": float(result.model_compression_ratio),
        "encoded_payload_and_metadata_bits": int(result.compressed_bits),
        "encoded_compression_ratio": float(result.compression_ratio),
        "actual_model_bytes": model_bytes,
        "actual_normalization_bytes": normalization_bytes,
        "actual_decoder_bytes": model_bytes + normalization_bytes,
        "actual_compression_ratio": original_bytes / (model_bytes + normalization_bytes),
        "encoding_seconds": encoding_seconds,
        "decoding_seconds": decoding_seconds,
        "reload_max_abs_difference": float(np.max(np.abs(reconstructed - result.reconstructed))),
    }


def main():
    configure_utf8_output()
    parser = argparse.ArgumentParser(description="QT-Joint: SSE + delta * ideal model bits.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mat", help="3D tensor in a .mat file")
    source.add_argument("--matrix-sequence", nargs=3, metavar=("DIR", "PREFIX", "COUNT"),
                        help="load matrices as a time sequence on axis 0")
    parser.add_argument("--mat-key")
    add_solver_arguments(parser)
    add_normalization_argument(parser)
    parser.add_argument("--error-budget", type=float, default=None,
                        help="raw-scale relative Frobenius post-check, e.g. 0.1; no auto-tuning")
    parser.add_argument("--output-dir", default="result")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        kwargs = solver_kwargs(args)
        if args.error_budget is not None and (
            not np.isfinite(args.error_budget) or args.error_budget < 0
        ):
            raise ValueError("--error-budget must be finite and nonnegative")
        start = time.perf_counter()
        raw = load_input(args)
        data, minimum, scale = normalize_input(raw, args.normalization)
        input_seconds = time.perf_counter() - start
        start = time.perf_counter()
        result = joint_lowrank_quantize(data, verbose=not args.quiet, **kwargs)
        optimization_seconds = time.perf_counter() - start
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))

    deployed, raw_deployed, storage = export_and_verify(
        result, args.output_dir, minimum, scale, args.normalization
    )
    raw_metrics = error_metrics(raw, raw_deployed)
    optimized_metrics = error_metrics(data, deployed)
    budget_pass = None
    if args.error_budget is not None:
        relative = raw_metrics["relative_f_error"]
        budget_pass = relative is not None and relative <= args.error_budget
    summary = {
        "method": "QT-Joint",
        "objective_definition": "optimization-domain SSE + delta * sum(n_Z * b_Z)",
        "normalization": args.normalization,
        "normalization_note": "delta uses optimization-domain units; min-max relative error differs from raw relative error",
        "ranks": result.ranks,
        "bits": {name: state.bit_width for name, state in result.quantizers.items()},
        "optimized_domain_metrics": optimized_metrics, "raw_domain_metrics": raw_metrics,
        "storage": storage, "final_metrics": result.final_metrics,
        "search_diagnostics": result.search_diagnostics,
        "input_seconds": input_seconds, "optimization_seconds": optimization_seconds,
        "error_budget": args.error_budget,
        "error_budget_domain": "raw_relative_Frobenius",
        "error_budget_satisfied": budget_pass,
        "error_budget_note": "post-validation only; weighted optimization is not an error-bound guarantee",
    }
    save_history(os.path.join(args.output_dir, "joint_qulow_history.csv"), result)
    write_csv(os.path.join(args.output_dir, "rank_selection.csv"),
              [item.__dict__ for item in result.rank_diagnostics])
    write_json(os.path.join(args.output_dir, "summary.json"), summary)
    write_json(os.path.join(args.output_dir, "run_config.json"), vars(args))
    print("ranks:", result.ranks)
    print("bits:", summary["bits"])
    print("raw SSE: {:.6e}".format(raw_metrics["sse"]))
    print("raw relative Frobenius error:", raw_metrics["relative_f_error"])
    print("optimization-domain relative Frobenius error:", optimized_metrics["relative_f_error"])
    print("ideal model compression ratio: {:.4f}x".format(storage["ideal_model_compression_ratio"]))
    print("actual compression ratio (model + normalization): {:.4f}x".format(storage["actual_compression_ratio"]))
    if args.error_budget is not None:
        print("raw-domain error budget satisfied:", budget_pass)
    print("summary:", os.path.abspath(os.path.join(args.output_dir, "summary.json")))


if __name__ == "__main__":
    main()

