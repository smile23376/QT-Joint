"""Windowed diagnostics; each window is an independent QT-Joint fit.

Normalization, when requested, is fitted once on the complete input, so its
scale is not silently changed from one window to the next. This script is not
an online factor-reuse compressor.
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
    export_and_verify, load_input, normalize_input, solver_kwargs, write_json,
)
from qulow_joint import (
    component_distribution_rows, configure_utf8_output, joint_lowrank_quantize,
    write_csv,
)


def iter_time_windows(data, time_axis, start, length, stride, drop_last=False):
    if not isinstance(time_axis, int) or not 0 <= time_axis < data.ndim:
        raise ValueError("--time-axis is outside the tensor dimensions")
    total = data.shape[time_axis]
    if not isinstance(start, int) or not 0 <= start < total:
        raise ValueError("--time-start must lie in [0, time dimension)")
    if length is not None and (not isinstance(length, int) or length <= 0):
        raise ValueError("--time-length must be positive")
    if not isinstance(stride, int) or stride <= 0:
        raise ValueError("--time-stride must be positive")
    length = total - start if length is None else length
    current = start
    while current < total:
        stop = min(total, current + length)
        if drop_last and stop - current < length:
            break
        selection = [slice(None)] * data.ndim
        selection[time_axis] = slice(current, stop)
        yield current, stop, data[tuple(selection)]
        if stop == total:
            break
        current += stride


def slice_window(array, axis, start, stop):
    selection = [slice(None)] * array.ndim
    selection[axis] = slice(start, stop)
    return array[tuple(selection)]


def slice_statistic(array, axis, start, stop):
    array = np.asarray(array)
    if array.ndim == 0 or array.shape[axis] == 1:
        return array
    return slice_window(array, axis, start, stop)


def main():
    configure_utf8_output()
    parser = argparse.ArgumentParser(description="Independent-window QT-Joint distribution diagnostics.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mat")
    source.add_argument("--matrix-sequence", nargs=3, metavar=("DIR", "PREFIX", "COUNT"))
    parser.add_argument("--mat-key")
    add_solver_arguments(parser)
    add_normalization_argument(parser)
    parser.set_defaults(max_iter=60, seed=19)
    parser.add_argument("--time-axis", type=int, default=None,
                        help="default: 0 for matrix sequences, 2 for MAT input")
    parser.add_argument("--time-start", type=int, default=0)
    parser.add_argument("--time-length", type=int)
    parser.add_argument("--time-stride", type=int)
    parser.add_argument("--drop-last", action="store_true",
                        help="skip a final window shorter than --time-length")
    parser.add_argument("--output-dir", default=os.path.join("result", "spatiotemporal_joint"))
    args = parser.parse_args()

    try:
        kwargs = solver_kwargs(args)
        raw = load_input(args)
        data, minimum, scale = normalize_input(raw, args.normalization)
        axis = args.time_axis if args.time_axis is not None else (0 if args.matrix_sequence else 2)
        if not 0 <= axis < raw.ndim:
            raise ValueError("--time-axis is outside the tensor dimensions")
        stride = args.time_stride if args.time_stride is not None else (
            args.time_length if args.time_length is not None else raw.shape[axis]
        )
        windows = list(iter_time_windows(data, axis, args.time_start, args.time_length,
                                         stride, args.drop_last))
        if not windows:
            raise ValueError("no windows remain; adjust --time-start/length/drop-last")
        if args.ranks is not None:
            for start, stop, window in windows:
                if any(rank > size for rank, size in zip(args.ranks, window.shape)):
                    raise ValueError(
                        f"ranks {args.ranks} do not fit window [{start}, {stop}) "
                        f"with shape {window.shape}; choose smaller ranks, auto, or --drop-last"
                    )
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))

    os.makedirs(args.output_dir, exist_ok=True)
    component_rows, window_rows = [], []
    for index, (start, stop, window) in enumerate(windows):
        local_kwargs = dict(kwargs, seed=args.seed + index)
        began = time.perf_counter()
        try:
            result = joint_lowrank_quantize(window, verbose=False, **local_kwargs)
        except (ValueError, TypeError) as exc:
            parser.error(str(exc))
        optimization_seconds = time.perf_counter() - began
        window_dir = os.path.join(args.output_dir, f"window_{index:04d}")
        deployed, raw_deployed, storage = export_and_verify(
            result, window_dir, slice_statistic(minimum, axis, start, stop),
            slice_statistic(scale, axis, start, stop), args.normalization,
        )
        raw_window = slice_window(raw, axis, start, stop)
        raw_errors = error_metrics(raw_window, raw_deployed)
        optimized_errors = error_metrics(window, deployed)
        for row in component_distribution_rows(result):
            row.update(window_index=index, time_start=start, time_stop=stop,
                       time_axis=axis, ranks=str(result.ranks))
            component_rows.append(row)
        row = {
            "window_index": index, "time_start": start, "time_stop": stop,
            "time_axis": axis, "tensor_values": int(window.size),
            "raw_tensor_mean": float(np.mean(raw_window)),
            "raw_tensor_std": float(np.std(raw_window)),
            "optimized_tensor_mean": float(np.mean(window)),
            "optimized_tensor_std": float(np.std(window)),
            "raw_sse": raw_errors["sse"], "raw_mse": raw_errors["mse"],
            "raw_relative_f_error": raw_errors["relative_f_error"],
            "optimized_relative_f_error": optimized_errors["relative_f_error"],
            "ideal_model_bits": result.model_bits,
            "ideal_compression_ratio": result.model_compression_ratio,
            "encoded_payload_and_metadata_bits": result.compressed_bits,
            "actual_decoder_bytes": storage["actual_decoder_bytes"],
            "actual_compression_ratio": storage["actual_compression_ratio"],
            "optimization_seconds": optimization_seconds,
            "encoding_seconds": storage["encoding_seconds"],
            "decoding_seconds": storage["decoding_seconds"],
            "ranks": str(result.ranks),
            **{f"{name}_bit": state.bit_width for name, state in result.quantizers.items()},
        }
        window_rows.append(row)
        write_json(os.path.join(window_dir, "summary.json"), {
            **row, "storage": storage, "raw_domain_metrics": raw_errors,
            "optimized_domain_metrics": optimized_errors,
            "final_metrics": result.final_metrics,
            "search_diagnostics": result.search_diagnostics,
        })
        print(row)

    config = vars(args).copy()
    config.update(resolved_time_axis=axis, resolved_time_stride=stride,
                  normalization_fit_scope="whole_input",
                  experiment_scope="independent windows; no cross-window factor reuse")
    write_json(os.path.join(args.output_dir, "run_config.json"), config)
    write_csv(os.path.join(args.output_dir, "component_distribution_joint.csv"), component_rows)
    write_csv(os.path.join(args.output_dir, "window_summary_joint.csv"), window_rows)


if __name__ == "__main__":
    main()

