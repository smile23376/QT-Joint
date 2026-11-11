"""Decode a QT-Joint model without access to the original float components."""
import argparse
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from qulow_joint import configure_utf8_output, load_quantized_model, tucker_reconstruct


def decode(model_path, normalization_path=None, sparse_path=None):
    factors, core, _, metadata = load_quantized_model(model_path)
    reconstruction = tucker_reconstruct(core, factors)
    if sparse_path is not None:
        # This file belongs to the optional external low+sparse experiment.
        from experiments.anomaly_detection_test import load_sparse_payload
        reconstruction = reconstruction + load_sparse_payload(
            sparse_path, expected_shape=reconstruction.shape
        )
    if normalization_path is not None:
        with np.load(normalization_path, allow_pickle=False) as normalization:
            minimum = np.asarray(normalization["minimum"], dtype=np.float64)
            scale = np.asarray(normalization["scale"], dtype=np.float64)
            if (not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(scale))
                    or np.any(scale <= 0)):
                raise ValueError("normalization must have finite offset and positive finite scale")
            minimum = np.broadcast_to(minimum, reconstruction.shape)
            scale = np.broadcast_to(scale, reconstruction.shape)
            reconstruction = reconstruction * scale + minimum
    if not np.all(np.isfinite(reconstruction)):
        raise ValueError("decoded tensor contains nonfinite values")
    return reconstruction, metadata


def main():
    configure_utf8_output()
    parser = argparse.ArgumentParser(description="Decode a saved QT-Joint integer model.")
    parser.add_argument("model", help="joint_qulow_model.npz")
    parser.add_argument("--normalization", help="normalization.npz; required to restore normalized data")
    parser.add_argument("--sparse", help="optional sparse_payload.npz from the external sparse experiment")
    parser.add_argument("--output", default="reconstructed.npy")
    args = parser.parse_args()
    if not args.output.lower().endswith(".npy"):
        parser.error("--output must end with .npy")
    try:
        reconstructed, metadata = decode(args.model, args.normalization, args.sparse)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.save(args.output, reconstructed, allow_pickle=False)
    print("decoded shape:", reconstructed.shape)
    print("format version:", metadata["format_version"])
    print("output domain:", "original scale" if args.normalization else "optimization domain (no sidecar supplied)")
    print("output:", os.path.abspath(args.output))


if __name__ == "__main__":
    main()
