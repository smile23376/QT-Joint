"""Evaluate six retained datasets using encoded and reloaded QT-Joint models.

The default source is an existing float32 canonical NPY, with a fallback to
the original files in Data. No input is generated when a file is missing.
"""
import argparse
import hashlib
import io
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import (
    add_solver_arguments, error_metrics,
    export_and_verify, normalize_input, solver_kwargs, write_json,
)
from qulow_joint import (
    configure_utf8_output, joint_lowrank_quantize, load_mat_tensor,
    load_matrix_sequence, save_history, write_csv,
)


DATASETS = ("PingData", "PlanetLab", "TaxiBJ_in", "TaxiBJ_out", "FT-AED", "UCSD-Ped2")
MAT_SOURCES = {
    "PingData": ("PingData/Pingsdata.mat", "data"),
    "TaxiBJ_in": ("Taxi/TaxiBJ_in.mat", "data"),
    "TaxiBJ_out": ("Taxi/TaxiBJ_out.mat", "data"),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw_dataset(name, data_root):
    """Return a tensor and provenance without imputing missing observations."""
    data_root = Path(data_root)
    notes = []
    if name in MAT_SOURCES:
        relative, key = MAT_SOURCES[name]
        paths = [data_root / relative]
        data = load_mat_tensor(str(paths[0]), key)
        axes = ["mode0", "mode1", "time"]
    elif name == "PlanetLab":
        paths = [data_root / "PlanetLab" / f"PlanetLabData_{i}" for i in range(1, 19)]
        data = load_matrix_sequence(str(data_root / "PlanetLab"), "PlanetLabData_", 18)
        axes = ["time / sequence index", "source node", "destination node"]
    elif name == "FT-AED":
        paths = [data_root / "FT-AED" / "FT_AED_tensor_package.zip"]
        with zipfile.ZipFile(paths[0]) as archive:
            payload = archive.read("FT_AED_tensor_package/FT_AED_tensor_ready.npz")
        with np.load(io.BytesIO(payload), allow_pickle=False) as package:
            data = np.concatenate([package[key] for key in ("speed", "volume", "occupancy")], axis=1)
        axes = ["milemarker", "speed/volume/occupancy x 4 lanes", "time"]
        notes.append("All supplied days; physical channels retain mixed units.")
    elif name == "UCSD-Ped2":
        try:
            from PIL import Image
        except ImportError as exc:
            raise ValueError("Reading UCSD-Ped2 original frames requires Pillow; use canonical NPY or install pillow") from exc
        paths = [data_root / "UCSD" / "ped2_Test.zip"]
        frames = []
        with zipfile.ZipFile(paths[0]) as archive:
            names = sorted(frame_name for frame_name in archive.namelist()
                           if frame_name.lower().endswith(".tif")
                           and "/._" not in frame_name and "_gt/" not in frame_name)
            for frame_name in names:
                with Image.open(io.BytesIO(archive.read(frame_name))) as frame:
                    resized = frame.convert("L").resize((36, 24), Image.Resampling.BILINEAR)
                    frames.append(np.asarray(resized, dtype=np.float32))
        if not frames:
            raise ValueError("UCSD-Ped2 archive contains no test frames")
        data = np.stack(frames, axis=2)
        axes = ["resized image row", "resized image column", "frame"]
        notes.append("All test frames in sorted archive order; grayscale bilinear resize to 24 x 36, matching canonical inputs.")
    else:
        raise ValueError(f"unknown dataset: {name}")
    return data, {"kind": "original", "files": [
        {"path": str(path.resolve()), "sha256": sha256(path)} for path in paths
    ], "axes": axes, "notes": notes}


def load_dataset(name, source="auto", canonical_dir=None, data_root=None):
    if name not in DATASETS:
        raise ValueError(f"unknown dataset: {name}")
    directories = ([Path(canonical_dir)] if canonical_dir is not None else [
        PROJECT_ROOT / "result" / "evaluation_H2_32" / "inputs",
        PROJECT_ROOT / "result" / "evaluation" / "inputs",
    ])
    if source in ("auto", "canonical"):
        for directory in directories:
            for filename in (f"{name}_s7.npy", f"{name}.npy"):
                path = directory / filename
                if path.is_file():
                    data = np.load(path, allow_pickle=False)
                    provenance = {"kind": "canonical", "files": [
                        {"path": str(path.resolve()), "sha256": sha256(path)}
                    ]}
                    break
            else:
                continue
            break
        else:
            if source == "canonical":
                raise ValueError(f"canonical input not found for {name} in {directories}")
            data, provenance = load_raw_dataset(name, data_root or PROJECT_ROOT / "Data")
    else:
        data, provenance = load_raw_dataset(name, data_root or PROJECT_ROOT / "Data")
    # Canonical benchmark convention: FP32 values, FP64 numerical optimization.
    if np.iscomplexobj(data):
        raise ValueError(f"{name}: input must be real-valued")
    with np.errstate(over="ignore", invalid="ignore"):
        data = np.ascontiguousarray(data, dtype=np.float32)
    normalize_input(data, "none")
    provenance.update(shape=list(data.shape), evaluation_dtype="float32",
                      tensor_values_sha256=hashlib.sha256(data.tobytes()).hexdigest())
    return data, provenance


def main():
    configure_utf8_output()
    parser = argparse.ArgumentParser(description="QT-Joint evaluation of six retained real datasets; default H={2,...,32}.")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--source", choices=("auto", "canonical", "original"), default="auto",
                        help="auto: prefer canonical NPY, otherwise read original Data files")
    parser.add_argument("--canonical-dir", type=Path, help="directory containing NAME_s7.npy or NAME.npy")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "Data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "result" / "dataset_evaluation")
    parser.add_argument("--quiet", action="store_true")
    add_solver_arguments(parser)
    parser.add_argument("--normalization", choices=("none", "scale", "global", "axis0"),
                        default="scale", help="scale=max-abs (default); none; global/axis0=min-max")
    args = parser.parse_args()
    try:
        kwargs = solver_kwargs(args)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    rows = []
    for name in dict.fromkeys(args.datasets):
        try:
            started = time.perf_counter()
            raw, provenance = load_dataset(name, args.source, args.canonical_dir, args.data_root)
            data, minimum, scale = normalize_input(raw, args.normalization)
            input_seconds = time.perf_counter() - started
            started = time.perf_counter()
            result = joint_lowrank_quantize(data, verbose=not args.quiet, **kwargs)
            optimization_seconds = time.perf_counter() - started
            directory = args.output_dir / name
            deployed, raw_deployed, storage = export_and_verify(
                result, str(directory), minimum, scale, args.normalization)
        except (ValueError, TypeError, OSError, KeyError, zipfile.BadZipFile) as exc:
            parser.error(f"{name}: {exc}")
        summary = {
            "dataset": name, "method": "QT-Joint", "source": provenance,
            "normalization": args.normalization, "candidate_bits": list(args.bits),
            "ranks": result.ranks,
            "bits": {component: state.bit_width for component, state in result.quantizers.items()},
            "raw_domain_metrics": error_metrics(raw, raw_deployed),
            "optimized_domain_metrics": error_metrics(data, deployed),
            "storage": storage, "final_metrics": result.final_metrics,
            "search_diagnostics": result.search_diagnostics,
            "input_seconds": input_seconds, "optimization_seconds": optimization_seconds,
        }
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update(dataset=name, input_source=provenance)
        write_json(directory / "run_config.json", config)
        write_json(directory / "summary.json", summary)
        save_history(str(directory / "joint_qulow_history.csv"), result)
        write_csv(str(directory / "rank_selection.csv"), [item.__dict__ for item in result.rank_diagnostics])
        row = {"dataset": name, "tensor_values": int(raw.size),
               "raw_relative_f_error": summary["raw_domain_metrics"]["relative_f_error"],
               "raw_sse": summary["raw_domain_metrics"]["sse"],
               "actual_compression_ratio": storage["actual_compression_ratio"],
               "actual_decoder_bytes": storage["actual_decoder_bytes"],
               "optimization_seconds": optimization_seconds}
        rows.append(row)
        # Persist completed datasets after every run so later failures lose no results.
        write_csv(str(args.output_dir / "dataset_summary.csv"), rows)
        write_json(args.output_dir / "dataset_summary.json", rows)
        print(row)


if __name__ == "__main__":
    main()
