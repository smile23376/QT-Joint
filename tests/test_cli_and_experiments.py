"""Executable regression tests for input, reported metrics, and deployment.

All files are generated numerical fixtures in temporary directories.  These
tests are not performance measurements on the paper's real datasets.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy.io import savemat

import main as runner
import qulow_joint as qt
from experiments.anomaly_detection_test import (
    load_sparse_payload, oracle_preservation_metrics, save_sparse_payload,
)
from experiments.decode_model import decode
from experiments.spatiotemporal_distribution_experiment import (
    iter_time_windows, slice_statistic,
)


PROJECT = Path(__file__).resolve().parents[1]


class InputAndMetricTests(unittest.TestCase):
    def test_four_normalization_modes_are_invertible(self):
        data = np.random.default_rng(1).normal(size=(5, 4, 3)) * 3 + 10
        for strategy in ("none", "scale", "global", "axis0"):
            with self.subTest(strategy=strategy):
                normalized, offset, scale = runner.normalize_input(data, strategy)
                np.testing.assert_allclose(normalized * scale + offset, data, atol=1e-14)

    def test_constant_and_tiny_minmax_ranges(self):
        for data in (np.full((2, 2, 2), 7.0),
                     np.linspace(1e-15, 2e-15, 8).reshape(2, 2, 2)):
            normalized, offset, scale = runner.normalize_input(data, "global")
            np.testing.assert_allclose(normalized * scale + offset, data, rtol=1e-15)
            self.assertEqual(float(normalized.min()), 0.0)
            if np.ptp(data) > 0:
                self.assertEqual(float(normalized.max()), 1.0)

    def test_real_valued_input_is_enforced_before_conversion(self):
        with self.assertRaises(ValueError):
            runner.normalize_input(np.ones((2, 2, 2)) * (1 + 2j), "none")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "complex.mat"
            savemat(path, {"data": np.ones((2, 2, 2)) * (1 + 2j)})
            with self.assertRaises(ValueError):
                qt.load_mat_tensor(str(path), "data")

    def test_metrics_match_direct_raw_calculation(self):
        raw = np.arange(24.0).reshape(2, 3, 4) + 100.0
        estimated = raw + 0.5
        metrics = runner.error_metrics(raw, estimated)
        self.assertEqual(metrics["sse"], 6.0)
        self.assertEqual(metrics["mse"], 0.25)
        self.assertEqual(metrics["rmse"], 0.5)
        self.assertAlmostEqual(metrics["relative_f_error"],
                               np.sqrt(6) / np.linalg.norm(raw))

    def test_zero_reference_metric_has_explicit_undefined_case(self):
        zeros = np.zeros((2, 2, 2))
        self.assertEqual(runner.error_metrics(zeros, zeros)["relative_f_error"], 0.0)
        self.assertIsNone(runner.error_metrics(zeros, zeros + 1)["relative_f_error"])

    def test_component_policy_parsing_and_rejection(self):
        self.assertEqual(runner.parse_clip_ratio("0.01"), 0.01)
        mapping = runner.parse_clip_ratio("G=0,A=0.1,B=0.2,C=0.3")
        self.assertEqual(mapping, dict(G=0, A=0.1, B=0.2, C=0.3))
        self.assertEqual(runner.parse_quantization_type("symmetric"), True)
        for value in ("-0.1", "1", "nan", "A=0,B=0,C=0", "G=0,A=0,B=0,C=0,C=0"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                runner.parse_clip_ratio(value)

    def test_single_row_matrix_sequence_preserves_three_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            for index in (1, 2, 3):
                np.savetxt(Path(directory) / f"scan_{index}", np.arange(4.0)[None, :] + index)
            loaded = qt.load_matrix_sequence(directory, "scan_", 3)
        self.assertEqual(loaded.shape, (3, 1, 4))
        np.testing.assert_array_equal(loaded[2], np.arange(4.0)[None, :] + 3)

    def test_no_clipping_calibration_uses_exact_extrema(self):
        # This interval is finite although upper-lower overflows float64.
        values = np.array([-1e308, 1e308])
        with np.errstate(all="raise"):
            interval = qt._calibration_interval(values, 0.0)
        self.assertEqual(interval, (-1e308, 1e308))


class ExperimentHelperTests(unittest.TestCase):
    def test_window_validation_rejects_nonprogressing_steps(self):
        data = np.zeros((3, 4, 7))
        for stride in (0, -1):
            with self.subTest(stride=stride), self.assertRaises(ValueError):
                list(iter_time_windows(data, 2, 0, 3, stride))
        for start in (-1, 7):
            with self.subTest(start=start), self.assertRaises(ValueError):
                list(iter_time_windows(data, 2, start, 3, 2))
        for length in (0, -1):
            with self.subTest(length=length), self.assertRaises(ValueError):
                list(iter_time_windows(data, 2, 0, length, 2))

    def test_windows_and_drop_last_are_explicit(self):
        data = np.zeros((3, 4, 7))
        windows = list(iter_time_windows(data, 2, 0, 3, 3))
        self.assertEqual([(a, b) for a, b, _ in windows], [(0, 3), (3, 6), (6, 7)])
        dropped = list(iter_time_windows(data, 2, 0, 3, 3, drop_last=True))
        self.assertEqual(len(dropped), 2)

    def test_windowed_axis0_normalization_inverse(self):
        raw = np.random.default_rng(2).normal(size=(3, 4, 7))
        data, offset, scale = runner.normalize_input(raw, "axis0")
        for start, stop, window in iter_time_windows(data, 2, 0, 3, 3):
            restored = window * slice_statistic(scale, 2, start, stop)
            restored += slice_statistic(offset, 2, start, stop)
            np.testing.assert_allclose(restored, raw[:, :, start:stop], atol=1e-15)

    def test_sparse_payload_evaluates_saved_precision_and_threshold(self):
        values = np.zeros((2, 3, 4))
        values[0, 0, 0] = 1.0 / 3
        values[1, 2, 3] = 1e-15
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "sparse.npz")
            decoded = save_sparse_payload(path, values, 32, 1e-12)
            repeated = load_sparse_payload(path, values.shape)
        expected = values.astype(np.float32).astype(np.float64)
        expected[np.abs(expected) <= 1e-12] = 0
        np.testing.assert_array_equal(decoded, expected)
        np.testing.assert_array_equal(repeated, expected)

    def test_oracle_metrics_are_labeled(self):
        clean = np.zeros((2, 2, 2))
        observed = clean.copy()
        observed[0, 0, 0] = 3
        mask = observed != 0
        metrics = oracle_preservation_metrics(observed, observed, mask, clean)
        self.assertIn("oracle_topk_f1", metrics)
        self.assertNotIn("anomaly_f1", metrics)
        self.assertTrue(metrics["uses_oracle_clean_baseline"])


class DeploymentAndCLITests(unittest.TestCase):
    def test_actual_storage_counts_both_decoder_files(self):
        raw = np.random.default_rng(3).normal(size=(4, 3, 2)) + 10
        data, minimum, scale = runner.normalize_input(raw, "global")
        result = qt.joint_lowrank_quantize(data, ranks=(2, 2, 2),
                                          candidate_bits=(2, 4), max_iter=1, verbose=False)
        with tempfile.TemporaryDirectory() as directory:
            deployed, raw_deployed, metrics = runner.export_and_verify(
                result, directory, minimum, scale, "global")
            actual = sum(os.path.getsize(os.path.join(directory, filename))
                         for filename in ("joint_qulow_model.npz", "normalization.npz"))
            independently_decoded, metadata = decode(
                os.path.join(directory, "joint_qulow_model.npz"),
                os.path.join(directory, "normalization.npz"))
        self.assertEqual(metrics["actual_decoder_bytes"], actual)
        self.assertEqual(metrics["actual_compression_ratio"], raw.size * 4 / actual)
        np.testing.assert_array_equal(deployed, result.reconstructed)
        np.testing.assert_array_equal(raw_deployed, independently_decoded)
        self.assertEqual(metadata["format_version"], 3)

    def _run(self, *args, expected_code=0):
        completed = subprocess.run([sys.executable, *map(str, args)], cwd=PROJECT,
                                   text=True, capture_output=True, timeout=45)
        self.assertEqual(completed.returncode, expected_code,
                         msg=completed.stdout + completed.stderr)
        return completed

    def test_cli_mat_input_metrics_and_independent_decode(self):
        raw = np.random.default_rng(4).normal(size=(5, 4, 3)) + 20
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.mat"
            output = Path(directory) / "output"
            savemat(source, {"data": raw})
            self._run("main.py", "--mat", source, "--mat-key", "data", "--ranks", "2,2,2",
                      "--bits", "2,4", "--max-iter", "2", "--delta", "0.001",
                      "--normalization", "global", "--error-budget", "0", "--quiet",
                      "--output-dir", output)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["error_budget_satisfied"])
            self._run("experiments/decode_model.py", output / "joint_qulow_model.npz",
                      "--normalization", output / "normalization.npz",
                      "--output", output / "reconstructed.npy")
            reconstructed = np.load(output / "reconstructed.npy", allow_pickle=False)
            expected = runner.error_metrics(raw, reconstructed)
            # MAT loads are column-contiguous whereas this independent fixture
            # is row-contiguous; reduction order can change the final ulp.
            for name, value in expected.items():
                np.testing.assert_allclose(summary["raw_domain_metrics"][name], value,
                                           rtol=2e-14, atol=1e-14, err_msg=name)
            self.assertAlmostEqual(
                summary["final_metrics"]["objective"],
                summary["optimized_domain_metrics"]["sse"]
                + 0.001 * summary["storage"]["ideal_model_bits"],
            )

    def test_cli_matrix_sequence_uses_time_axis_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            for index in (1, 2, 3):
                np.savetxt(Path(directory) / f"x{index}",
                           np.arange(12.0).reshape(4, 3) + index)
            output = Path(directory) / "result"
            self._run("main.py", "--matrix-sequence", directory, "x", "3",
                      "--ranks", "2,2,2", "--bits", "2,4", "--max-iter", "1",
                      "--output-dir", output, "--quiet")
            array, _ = decode(output / "joint_qulow_model.npz",
                              output / "normalization.npz")
        self.assertEqual(array.shape, (3, 4, 3))

    def test_cli_invalid_stride_fails_instead_of_hanging(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.mat"
            savemat(source, {"data": np.arange(24.0).reshape(2, 3, 4)})
            completed = self._run("experiments/spatiotemporal_distribution_experiment.py",
                                  "--mat", source, "--mat-key", "data",
                                  "--time-stride", "-1", expected_code=2)
        self.assertIn("--time-stride must be positive", completed.stderr)

    def test_cli_requires_an_explicit_input_source(self):
        for script in ("main.py", "experiments/anomaly_detection_test.py",
                       "experiments/spatiotemporal_distribution_experiment.py"):
            with self.subTest(script=script):
                completed = self._run(script, expected_code=2)
                self.assertIn("required", completed.stderr)

    def test_windowed_experiment_uses_explicit_mat_input(self):
        raw = np.arange(48.0).reshape(4, 3, 4)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.mat"
            output = Path(directory) / "windows"
            savemat(source, {"data": raw})
            self._run("experiments/spatiotemporal_distribution_experiment.py",
                      "--mat", source, "--mat-key", "data", "--ranks", "2,2,2",
                      "--bits", "2,4", "--max-iter", "1", "--time-length", "2",
                      "--output-dir", output)
            config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["resolved_time_axis"], 2)
            self.assertTrue((output / "window_0001" / "summary.json").is_file())

    def test_anomaly_experiment_uses_explicit_mat_input(self):
        raw = np.arange(24.0).reshape(3, 4, 2)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.mat"
            output = Path(directory) / "anomalies"
            savemat(source, {"data": raw})
            self._run("experiments/anomaly_detection_test.py",
                      "--mat", source, "--mat-key", "data", "--ranks", "2,2,2",
                      "--bits", "2,4", "--max-iter", "1", "--split-iter", "1",
                      "--output-dir", output)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(summary["rows"]), 2)
            self.assertEqual(summary["rows"][0]["method"], "QT-Joint")


if __name__ == "__main__":
    unittest.main()
