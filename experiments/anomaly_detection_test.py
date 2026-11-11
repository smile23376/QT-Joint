"""Anomaly preservation experiment, not a new module of the QT-Joint method.

The low-rank-plus-sparse branch is an explicitly optional external extension.
Oracle top-k scores use the known clean baseline and injected anomaly count;
they must not be interpreted as deployable unsupervised detection accuracy.
"""
import argparse
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main import (
    add_normalization_argument, add_solver_arguments, error_metrics,
    export_and_verify, normalize_input, solver_kwargs, write_json,
)
from qulow_joint import (
    anomaly_metrics, configure_utf8_output, joint_lowrank_quantize,
    load_mat_tensor, robust_low_sparse_split, sparse_coo_storage_bits, write_csv,
)


def inject_sparse_anomalies(data, fraction, magnitude, seed):
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("--anomaly-fraction must be in [0,1]")
    if not np.isfinite(magnitude) or magnitude <= 0:
        raise ValueError("--anomaly-magnitude must be finite and positive")
    rng = np.random.RandomState(seed)
    mask = rng.rand(*data.shape) < fraction
    signed = rng.choice([-1.0, 1.0], size=data.shape)
    scale = float(np.std(data))
    if fraction > 0 and scale == 0:
        raise ValueError("constant data has zero anomaly scale; choose nonconstant clean input")
    observed = np.asarray(data, dtype=np.float64).copy()
    observed[mask] += signed[mask] * magnitude * scale
    return observed, mask


def save_sparse_payload(path, sparse, value_bits=32, zero_tolerance=1e-12):
    """Cast and sparsify before evaluating; return the exact decoded payload."""
    if value_bits not in (32, 64):
        raise ValueError("sparse values can be stored at 32 or 64 bits")
    if not np.isfinite(zero_tolerance) or zero_tolerance < 0:
        raise ValueError("sparse zero tolerance must be finite and nonnegative")
    dtype = np.float32 if value_bits == 32 else np.float64
    stored = np.asarray(sparse, dtype=dtype).copy()
    if not np.all(np.isfinite(stored)):
        raise ValueError("sparse values overflow the selected storage dtype")
    stored[np.abs(stored) <= zero_tolerance] = 0
    indices = np.argwhere(stored != 0).astype(np.int64)
    values = stored[tuple(indices.T)] if len(indices) else np.empty(0, dtype=dtype)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, shape=np.asarray(stored.shape, dtype=np.int64),
                        indices=indices, values=values)
    decoded = load_sparse_payload(path, expected_shape=stored.shape)
    if not np.array_equal(decoded, stored.astype(np.float64)):
        raise RuntimeError("sparse payload reload mismatch")
    return decoded


def load_sparse_payload(path, expected_shape=None):
    with np.load(path, allow_pickle=False) as loaded:
        shape_array = np.asarray(loaded["shape"])
        if shape_array.shape != (3,) or not np.issubdtype(shape_array.dtype, np.integer):
            raise ValueError("sparse payload shape must contain three integers")
        shape = tuple(int(value) for value in shape_array)
        if any(value <= 0 for value in shape):
            raise ValueError("sparse payload shape must be positive")
        if expected_shape is not None and shape != tuple(expected_shape):
            raise ValueError("sparse payload and Tucker reconstruction shapes differ")
        indices = np.asarray(loaded["indices"])
        values = np.asarray(loaded["values"])
        if (indices.ndim != 2 or indices.shape[1] != 3 or values.ndim != 1
                or len(indices) != len(values)
                or not np.issubdtype(indices.dtype, np.integer)
                or not np.all(np.isfinite(values))):
            raise ValueError("invalid sparse COO payload")
        if len(indices) and (np.any(indices < 0) or np.any(indices >= np.asarray(shape))):
            raise ValueError("sparse index lies outside its tensor shape")
        if len(indices) and len(np.unique(indices, axis=0)) != len(indices):
            raise ValueError("sparse payload contains duplicate indices")
        reconstructed = np.zeros(shape, dtype=np.float64)
        if len(indices):
            reconstructed[tuple(indices.T)] = values
        return reconstructed


def oracle_preservation_metrics(observed, reconstructed, mask, clean):
    metrics = anomaly_metrics(observed, reconstructed, anomaly_mask=mask, baseline=clean)
    for old, new in (
        ("anomaly_precision", "oracle_topk_precision"),
        ("anomaly_recall", "oracle_topk_recall"),
        ("anomaly_f1", "oracle_topk_f1"),
        ("anomaly_iou", "oracle_topk_iou"),
    ):
        metrics[new] = metrics.pop(old)
    metrics["oracle_topk_count"] = int(np.sum(mask))
    metrics["uses_oracle_clean_baseline"] = True
    return metrics


def main():
    configure_utf8_output()
    parser = argparse.ArgumentParser(description="QT-Joint anomaly preservation and optional sparse extension.")
    parser.add_argument("--mat", required=True, help="clean 3D tensor in a .mat file")
    parser.add_argument("--mat-key")
    add_solver_arguments(parser)
    add_normalization_argument(parser)
    parser.set_defaults(max_iter=60, ranks=(8, 8, 8), seed=11)
    parser.add_argument("--anomaly-fraction", type=float, default=0.01)
    parser.add_argument("--anomaly-magnitude", type=float, default=8.0)
    parser.add_argument("--sparse-value-bits", type=int, choices=(32, 64), default=32)
    parser.add_argument("--sparse-zero-tolerance", type=float, default=1e-12)
    parser.add_argument("--split-iter", type=int, default=30)
    parser.add_argument("--output-dir", default=os.path.join("result", "anomaly_joint"))
    args = parser.parse_args()
    try:
        kwargs = solver_kwargs(args)
        if args.split_iter < 1:
            raise ValueError("--split-iter must be positive")
        clean = load_mat_tensor(args.mat, args.mat_key)
        # Validate clean input before injection, even when normalization is off.
        clean, _, _ = normalize_input(clean, "none")
        observed, mask = inject_sparse_anomalies(
            clean, args.anomaly_fraction, args.anomaly_magnitude, args.seed
        )
        data, minimum, scale = normalize_input(observed, args.normalization)
        started = time.perf_counter()
        direct = joint_lowrank_quantize(data, verbose=False, **kwargs)
        direct_seconds = time.perf_counter() - started
        # Use precisely the direct branch's structure for the extension comparison.
        started = time.perf_counter()
        low, sparse, split_history = robust_low_sparse_split(
            data, direct.ranks, max_iter=args.split_iter
        )
        split_seconds = time.perf_counter() - started
        extension_kwargs = dict(kwargs, ranks=direct.ranks)
        started = time.perf_counter()
        sparse_aware = joint_lowrank_quantize(low, verbose=False, **extension_kwargs)
        extension_seconds = time.perf_counter() - started
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))

    direct_dir = os.path.join(args.output_dir, "direct")
    extension_dir = os.path.join(args.output_dir, "sparse_extension")
    _, direct_raw, direct_storage = export_and_verify(
        direct, direct_dir, minimum, scale, args.normalization
    )
    extension_dq, _, extension_storage = export_and_verify(
        sparse_aware, extension_dir, minimum, scale, args.normalization
    )
    sparse_path = os.path.join(extension_dir, "sparse_payload.npz")
    sparse_started = time.perf_counter()
    try:
        sparse_decoded = save_sparse_payload(
            sparse_path, sparse, args.sparse_value_bits, args.sparse_zero_tolerance
        )
    except ValueError as exc:
        parser.error(str(exc))
    sparse_export_seconds = time.perf_counter() - sparse_started
    extension_raw = (extension_dq + sparse_decoded) * scale + minimum
    # This is a theoretical bit-packed COO estimate, not the saved NPZ size.
    sparse_bits = sparse_coo_storage_bits(
        sparse_decoded, value_bit_width=args.sparse_value_bits, zero_tolerance=0.0
    )
    sparse_bytes = os.path.getsize(sparse_path)
    original_bytes = int(observed.size) * 4
    rows = []
    for name, result, reconstruction, storage, extra_bits, extra_bytes, seconds in (
        ("QT-Joint", direct, direct_raw, direct_storage, 0, 0, direct_seconds),
        ("Sparse extension + QT-Joint", sparse_aware, extension_raw,
         extension_storage, sparse_bits, sparse_bytes, extension_seconds),
    ):
        effective_ideal_bits = int(result.model_bits) + extra_bits
        raw_errors = error_metrics(observed, reconstruction)
        decoder_bytes = storage["actual_decoder_bytes"] + extra_bytes
        row = {
            "method": name, "is_core_QT_Joint": extra_bytes == 0,
            "ranks": str(result.ranks),
            "tucker_ideal_bits": int(result.model_bits),
            "tucker_encoded_payload_and_metadata_bits": int(result.compressed_bits),
            "sparse_theoretical_COO_bits": extra_bits,
            "effective_theoretical_bits": effective_ideal_bits,
            "theoretical_compression_ratio": original_bytes * 8 / effective_ideal_bits,
            "actual_tucker_model_bytes": storage["actual_model_bytes"],
            "actual_normalization_bytes": storage["actual_normalization_bytes"],
            "actual_sparse_bytes": extra_bytes,
            "actual_decoder_bytes": decoder_bytes,
            "actual_compression_ratio": original_bytes / decoder_bytes,
            "sparse_density": float(np.count_nonzero(sparse_decoded) / data.size) if extra_bytes else 0.0,
            "raw_sse": raw_errors["sse"], "raw_mse": raw_errors["mse"],
            "raw_relative_f_error": raw_errors["relative_f_error"],
            "optimization_seconds": seconds,
            "external_sparse_split_seconds": split_seconds if extra_bytes else 0.0,
            "encoding_seconds": storage["encoding_seconds"],
            "decoding_seconds": storage["decoding_seconds"],
            "sparse_export_and_reload_seconds": sparse_export_seconds if extra_bytes else 0.0,
        }
        row.update(oracle_preservation_metrics(observed, reconstruction, mask, clean))
        rows.append(row)
        print(row)

    write_csv(os.path.join(args.output_dir, "anomaly_joint_metrics.csv"), rows)
    write_csv(os.path.join(args.output_dir, "sparse_split_history.csv"), split_history)
    write_json(os.path.join(args.output_dir, "run_config.json"), {
        **vars(args),
        "experiment_scope": "core QT-Joint versus an external low+sparse extension",
        "oracle_metric_caveat": "top-k uses true injected count and clean baseline; not a deployable detector",
        "extension_structure": "uses the direct branch ranks; sparse cast/threshold are applied before evaluation",
    })
    write_json(os.path.join(args.output_dir, "summary.json"), {
        "rows": rows,
        "direct_search_diagnostics": direct.search_diagnostics,
        "extension_search_diagnostics": sparse_aware.search_diagnostics,
        "storage_note": "actual compression includes model, normalization, and sparse payload where needed; excludes diagnostics",
        "oracle_metric_caveat": "known clean baseline and known anomaly count; precision equals recall by construction",
    })


if __name__ == "__main__":
    main()

