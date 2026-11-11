"""Independent numerical checks against QTJoint section 3--6 equations.

The reference quantizer and coordinate oracle intentionally do not call the
implementation's quantization/scoring helpers.  Finite differences below are
for smooth dequantized Tucker variables, NOT a claim that STE is a derivative
of the discontinuous, adaptively calibrated quantizer.
"""

import itertools
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

import qulow_joint as qt


NAMES = ("A", "B", "C", "G")
ORDER = ("G", "A", "B", "C")


def reference_quantize(values, bits, symmetric=False, clip_ratio=0.0):
    """Literal equations (3.3)--(3.5), linear quantiles and ties-to-even."""
    values = np.asarray(values, dtype=np.float64)
    lower = min(float(np.quantile(values, clip_ratio / 2.0)), 0.0)
    upper = max(float(np.quantile(values, 1.0 - clip_ratio / 2.0)), 0.0)
    if lower == upper == 0.0 and np.any(values != 0.0):
        lower, upper = min(float(values.min()), 0.0), max(float(values.max()), 0.0)
    qmax = 2 ** (bits - 1) - 1 if symmetric else 2**bits - 1
    qmin = -qmax if symmetric else 0
    if np.all(values == 0.0):
        scale, zero = 1.0, 0
    elif symmetric:
        scale, zero = max(abs(lower), abs(upper)) / qmax, 0
    else:
        scale = (upper - lower) / (qmax - qmin)
        zero = int(np.clip(np.rint(qmin - lower / scale), qmin, qmax))
    codes = np.clip(np.rint(values / scale) + zero, qmin, qmax).astype(np.int64)
    return codes, scale * (codes - zero), scale, zero, lower, upper


def reference_reconstruct(arrays):
    return np.einsum(
        "uvw,iu,jv,kw->ijk", arrays["G"], arrays["A"], arrays["B"], arrays["C"],
        optimize=True,
    )


def reference_objective(target, masters, bits, delta, symmetric=False, clip_ratio=0.0):
    dq = {
        name: reference_quantize(
            array, bits[name],
            symmetric[name] if isinstance(symmetric, dict) else symmetric,
            clip_ratio[name] if isinstance(clip_ratio, dict) else clip_ratio,
        )[1]
        for name, array in masters.items()
    }
    sse = float(np.sum((target - reference_reconstruct(dq)) ** 2))
    return sse + delta * sum(array.size * bits[name] for name, array in masters.items())


def reference_coordinate_scan(target, masters, initial, candidates, delta, sweeps=1,
                              symmetric=False, clip_ratio=0.0):
    bits = dict(initial)
    for _ in range(sweeps):
        changed = False
        for name in ORDER:
            best = reference_objective(target, masters, bits, delta, symmetric, clip_ratio)
            best_bit = bits[name]
            for bit in sorted(candidates):
                proposal = dict(bits, **{name: bit})
                score = reference_objective(target, masters, proposal, delta, symmetric, clip_ratio)
                if score < best:
                    best, best_bit = score, bit
            changed |= best_bit != bits[name]
            bits[name] = best_bit
        if not changed:
            break
    return bits


def random_components(seed=13):
    rng = np.random.default_rng(seed)
    shape, ranks = (5, 4, 6), (3, 2, 2)
    factors = tuple(rng.normal(size=(n, r)) for n, r in zip(shape, ranks))
    core = rng.normal(size=ranks)
    target = rng.normal(size=shape)
    return target, factors, core


def quantize_components(factors, core, bits=3, clip_ratio=0.0, symmetric=False):
    masters = dict(zip(NAMES, tuple(factors) + (core,)))
    states, arrays = {}, {}
    for name in NAMES:
        _, arrays[name], states[name] = qt._quantize_array(
            masters[name], bits, symmetric, clip_ratio=clip_ratio,
        )
    return tuple(arrays[name] for name in NAMES[:3]), arrays["G"], states


class QuantizationSpecificationTests(unittest.TestCase):
    def assert_reference(self, values, bits, symmetric=False, clip_ratio=0.0):
        expected = reference_quantize(values, bits, symmetric, clip_ratio)
        codes, dq, state = qt._quantize_array(values, bits, symmetric, clip_ratio=clip_ratio)
        np.testing.assert_array_equal(codes, expected[0])
        np.testing.assert_allclose(dq, expected[1], rtol=2e-15, atol=0)
        self.assertAlmostEqual(state.scale, expected[2], places=14)
        self.assertEqual(state.zero_point, expected[3])
        self.assertEqual(state.clip_ratio, clip_ratio)
        self.assertEqual(state.model_bits, np.asarray(values).size * bits)
        return codes, dq, state

    def test_asymmetric_positive_interval_includes_zero(self):
        _, dq, state = self.assert_reference(np.array([2.0, 4.0]), 2)
        self.assertEqual(state.zero_point, 0)
        self.assertGreater(dq[1], dq[0])

    def test_asymmetric_negative_interval_includes_zero(self):
        _, dq, state = self.assert_reference(np.array([-4.0, -2.0]), 2)
        self.assertEqual(state.zero_point, 3)
        self.assertLess(dq[0], dq[1])

    def test_rounding_precedes_zero_point_addition(self):
        codes, dq, state = self.assert_reference(np.array([-1.0, -0.5, 0.5, 2.0]), 2)
        self.assertEqual(state.scale, 1.0)
        self.assertEqual(state.zero_point, 1)
        np.testing.assert_array_equal(codes, [0, 1, 1, 3])
        np.testing.assert_array_equal(dq, [-1.0, 0.0, 0.0, 2.0])

    def test_symmetric_uses_signed_two_sided_percentiles(self):
        self.assert_reference(np.array([-10.0, -8.0, -1.0, 0.0, 1.0, 1.0, 2.0]),
                              4, symmetric=True, clip_ratio=0.5)

    def test_clipping_proportion_is_total_removed_fraction(self):
        values = np.arange(-10.0, 11.0)
        _, _, state = self.assert_reference(values, 3, clip_ratio=0.2)
        self.assertAlmostEqual(state.clip_min, -8.0)
        self.assertAlmostEqual(state.clip_max, 8.0)

    def test_nonzero_tail_survives_collapsed_quantile_interval(self):
        values = np.r_[np.zeros(1000), 1.0]
        _, dq, state = self.assert_reference(values, 4, clip_ratio=0.02)
        self.assertEqual(dq[-1], 1.0)
        self.assertEqual(np.count_nonzero(dq), 1)
        self.assertEqual(state.scale, 1.0 / 15.0)

    def test_only_exact_zero_uses_zero_special_case(self):
        values = np.array([1e-13, 2e-13, -1e-13])
        for symmetric in (False, True):
            with self.subTest(symmetric=symmetric):
                _, dq, state = self.assert_reference(values, 8, symmetric=symmetric)
                self.assertNotEqual(state.scale, 1.0)
                self.assertTrue(np.any(dq != 0.0))

    def test_all_zero_encoding_is_well_defined(self):
        for symmetric in (False, True):
            with self.subTest(symmetric=symmetric):
                codes, dq, state = self.assert_reference(np.zeros((4, 3)), 2, symmetric)
                np.testing.assert_array_equal(codes, 0)
                np.testing.assert_array_equal(dq, 0)
                self.assertEqual(state.scale, 1.0)
                self.assertEqual(state.zero_point, 0)

    def test_positive_and_negative_constants_are_not_erased(self):
        for value in (-3.75, 2.25):
            for symmetric in (False, True):
                with self.subTest(value=value, symmetric=symmetric):
                    _, dq, _ = self.assert_reference(np.full((2, 3), value), 6, symmetric, 0.8)
                    np.testing.assert_allclose(dq, value, rtol=2e-15)

    def test_quantizer_is_deterministic_and_recalibrates_with_array(self):
        values = np.array([-1.0, 0.15, 0.75, 2.0])
        c1, dq1, state1 = qt._quantize_array(values, 5, False, clip_ratio=0.1)
        c2, dq2, state2 = qt._quantize_array(values, 5, False, clip_ratio=0.1)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(dq1, dq2)
        self.assertEqual(state1.scale, state2.scale)
        _, _, state3 = qt._quantize_array(2 * values, 5, False, clip_ratio=0.1)
        self.assertAlmostEqual(state3.scale, 2 * state1.scale)


class RankAndTuckerSpecificationTests(unittest.TestCase):
    def test_joint_rank_proxy_matches_bruteforce(self):
        rng = np.random.default_rng(41)
        data = rng.normal(size=(4, 5, 3))
        spectra = [np.linalg.svd(np.moveaxis(data, m, 0).reshape(data.shape[m], -1),
                                 compute_uv=False) ** 2 for m in range(3)]
        for delta in (0.0, 0.01, 0.2, 10.0):
            with self.subTest(delta=delta):
                choices = []
                for ranks in itertools.product(range(1, 5), range(1, 6), range(1, 4)):
                    cost = 32 * (sum(n * r for n, r in zip(data.shape, ranks)) + np.prod(ranks))
                    tail = sum(float(s[r:].sum()) for s, r in zip(spectra, ranks))
                    choices.append((tail + delta * cost, cost, ranks))
                expected = min(choices)
                ranks, diagnostics = qt.select_ranks_hosvd_proxy(data, delta=delta, max_rank=None)
                self.assertEqual(ranks, expected[2])
                for state in diagnostics:
                    self.assertAlmostEqual(state.joint_proxy_objective, expected[0], places=10)

    def test_rank_cap_limits_only_search_range(self):
        data = np.random.default_rng(3).normal(size=(5, 4, 6))
        ranks, _ = qt.select_ranks_hosvd_proxy(data, delta=0.0, max_rank=2)
        self.assertEqual(ranks, (2, 2, 2))

    def test_zero_tensor_selects_minimal_structure(self):
        ranks, _ = qt.select_ranks_hosvd_proxy(np.zeros((5, 4, 3)), delta=0.1)
        self.assertEqual(ranks, (1, 1, 1))

    def test_skinny_unfolding_allows_rank_up_to_mode_size(self):
        data = np.arange(4.0).reshape(4, 1, 1)
        factors, core = qt.hosvd_initialize(data, (4, 1, 1))
        self.assertEqual(factors[0].shape, (4, 4))
        np.testing.assert_allclose(factors[0].T @ factors[0], np.eye(4), atol=2e-14)
        np.testing.assert_allclose(qt.tucker_reconstruct(core, factors), data, atol=2e-14)

    def test_hosvd_residual_respects_spectral_upper_bound(self):
        data = np.random.default_rng(9).normal(size=(5, 4, 3))
        ranks = (2, 3, 2)
        factors, core = qt.hosvd_initialize(data, ranks)
        sse = np.sum((data - qt.tucker_reconstruct(core, factors)) ** 2)
        bound = sum(np.sum(np.linalg.svd(np.moveaxis(data, m, 0).reshape(data.shape[m], -1),
                                        compute_uv=False)[r:] ** 2)
                    for m, r in enumerate(ranks))
        self.assertLessEqual(sse, bound + 1e-12)


class GradientAndIncrementSpecificationTests(unittest.TestCase):
    def test_all_half_sse_gradient_entries_match_finite_differences(self):
        rng = np.random.default_rng(77)
        target = rng.normal(size=(3, 4, 2))
        factors = (rng.normal(size=(3, 2)), rng.normal(size=(4, 2)), rng.normal(size=(2, 1)))
        core = rng.normal(size=(2, 2, 1))
        fg, cg, _ = qt.quantization_aware_gradients(target, factors, core, factors, core)
        arrays = list(factors) + [core]
        gradients = list(fg) + [cg]
        epsilon = 1e-6
        for block, array in enumerate(arrays):
            for index in np.ndindex(array.shape):
                plus, minus = [a.copy() for a in arrays], [a.copy() for a in arrays]
                plus[block][index] += epsilon
                minus[block][index] -= epsilon
                lp = 0.5 * np.sum((reference_reconstruct(dict(zip(NAMES, plus))) - target) ** 2)
                lm = 0.5 * np.sum((reference_reconstruct(dict(zip(NAMES, minus))) - target) ** 2)
                finite = (lp - lm) / (2.0 * epsilon)
                self.assertAlmostEqual(gradients[block][index], finite, delta=2e-7 * max(1.0, abs(finite)))

    def test_ste_mask_includes_endpoints_and_excludes_clipped_values(self):
        factors = (np.array([-2.0, -1.0, 0.0, 1.0, 2.0]).reshape(5, 1),
                   np.ones((2, 1)), np.ones((3, 1)))
        core = np.ones((1, 1, 1))
        dq_factors, dq_core, states = quantize_components(factors, core, 2, 0.5, True)
        fg, _, _ = qt.quantization_aware_gradients(np.zeros((5, 2, 3)), factors, core,
                                                 dq_factors, dq_core, quantizers=states)
        np.testing.assert_allclose(fg[0].ravel(), [0, -6, 0, 6, 0], atol=1e-14)

    def test_ste_uses_actual_quantizer_interval_not_percentile_endpoints(self):
        factors = (np.array([-1.0, 0.0, 1.0, 1.7]).reshape(4, 1),
                   np.ones((2, 1)), np.ones((3, 1)))
        core = np.ones((1, 1, 1))
        dq_factors, dq_core, states = quantize_components(factors, core, 2)
        masked, masked_core, _ = qt.quantization_aware_gradients(np.zeros((4, 2, 3)),
            factors, core, dq_factors, dq_core, quantizers=states)
        unmasked, unmasked_core, _ = qt.quantization_aware_gradients(np.zeros((4, 2, 3)),
            factors, core, dq_factors, dq_core)
        for name, master, raw, actual in zip(NAMES, list(factors) + [core],
                                            list(unmasked) + [unmasked_core],
                                            list(masked) + [masked_core]):
            state = states[name]
            coordinate = master / state.scale + state.zero_point
            expected_mask = (coordinate >= state.qmin) & (coordinate <= state.qmax)
            np.testing.assert_array_equal(actual, raw * expected_mask)
        self.assertEqual(masked[0][0, 0], 0.0)
        self.assertNotEqual(unmasked[0][0, 0], 0.0)

    def test_zero_component_has_unit_ste_mask(self):
        factors = (np.zeros((3, 1)), np.ones((2, 1)), np.ones((2, 1)))
        core = np.ones((1, 1, 1))
        dq_factors, dq_core, states = quantize_components(factors, core, 3)
        fg, _, _ = qt.quantization_aware_gradients(np.ones((3, 2, 2)), factors, core,
                                                 dq_factors, dq_core, quantizers=states)
        np.testing.assert_allclose(fg[0], -4.0, atol=1e-14)

    def test_loss_increments_match_full_reconstruction_for_each_component(self):
        target, factors, core = random_components()
        base = dict(zip(NAMES, list(factors) + [core]))
        residual = reference_reconstruct(base) - target
        base_sse = np.sum(residual**2)
        rng = np.random.default_rng(93)
        for name in NAMES:
            delta = rng.normal(size=base[name].shape) * 0.37
            if name == "G":
                cross = qt._core_cross_gradient_half_sse(residual, factors)
                quadratic = qt._core_delta_output_norm_sq(delta, factors)
            else:
                mode = NAMES.index(name)
                cross = qt._factor_cross_gradient_half_sse(residual, factors, core, mode)
                quadratic = qt._factor_delta_output_norm_sq(delta, factors, core, mode)
            expected = np.sum((reference_reconstruct(dict(base, **{name: base[name] + delta})) - target) ** 2) - base_sse
            predicted = 2.0 * np.sum(cross * delta) + quadratic
            self.assertAlmostEqual(predicted, expected, delta=1e-10 * max(1.0, abs(expected)))


class CoordinateSearchSpecificationTests(unittest.TestCase):
    def test_coordinate_caches_avoid_full_reconstruction_per_candidate(self):
        target, factors, core = random_components(31)
        candidates = (2, 3, 4, 5, 6, 7, 8, 10)
        diagnostics = {}
        with mock.patch.object(qt, "tucker_reconstruct", wraps=qt.tucker_reconstruct) as rebuild:
            with mock.patch.object(qt, "_calibration_interval", wraps=qt._calibration_interval) as calibrate:
                qt.select_quantizers_global(target, factors, core, candidates, delta=0.01,
                                             coordinate_sweeps=1, diagnostics=diagnostics)
        # One baseline and at most one chosen-candidate check per coordinate.
        # Thirty-two individual candidate scores must not trigger full rebuilds.
        self.assertLessEqual(rebuild.call_count, 5)
        # Four initial arrays, then one cached interval per coordinate scan.
        self.assertLessEqual(calibrate.call_count, 8)
        self.assertEqual(diagnostics["candidate_evaluations"], 4 * len(candidates))
        self.assertEqual(diagnostics["scan_order"], list(ORDER))

    def test_one_scan_agrees_with_gabc_bruteforce_oracle(self):
        target, factors, core = random_components(22)
        masters = dict(zip(NAMES, tuple(factors) + (core,)))
        candidates, delta = (2, 3, 5), 0.03
        expected = reference_coordinate_scan(target, masters, {n: 5 for n in NAMES},
                                             candidates, delta, sweeps=1)
        _, _, _, states = qt.select_quantizers_global(
            target, factors, core, candidates, delta=delta, coordinate_sweeps=1,
            symmetric=False, clip_ratio=0.0,
        )
        self.assertEqual({name: state.bit_width for name, state in states.items()}, expected)

    def test_equal_objective_preserves_incumbent_bits(self):
        factors = (np.ones((3, 1)), np.ones((2, 1)), np.ones((2, 1)))
        core = np.zeros((1, 1, 1))
        _, _, _, states = qt.select_quantizers_global(
            np.zeros((3, 2, 2)), factors, core, (2, 4, 8), delta=0.0,
            coordinate_sweeps=4, symmetric=False, clip_ratio=0.0,
        )
        self.assertEqual({state.bit_width for state in states.values()}, {8})

    def test_full_scan_result_is_coordinate_optimal_for_fixed_masters(self):
        data = np.random.default_rng(14).normal(size=(5, 4, 3))
        candidates, delta = (2, 3, 5), 0.008
        result = qt.joint_lowrank_quantize(data, ranks=(2, 2, 2), candidate_bits=candidates,
                                          delta=delta, max_iter=3, full_scan_at_end=True,
                                          use_skew_filter=True, verbose=False)
        masters = dict(zip(NAMES, tuple(result.master_factors) + (result.master_core,)))
        bits = {n: result.quantizers[n].bit_width for n in NAMES}
        baseline = reference_objective(data, masters, bits, delta)
        for name in NAMES:
            for bit in candidates:
                proposal = dict(bits, **{name: bit})
                score = reference_objective(data, masters, proposal, delta)
                self.assertGreaterEqual(score + 1e-11, baseline)


class SolverAndEncodingSpecificationTests(unittest.TestCase):
    def test_monotone_objectives_and_exact_final_units(self):
        data = np.random.default_rng(51).normal(size=(7, 6, 5))
        delta = 0.013
        result = qt.joint_lowrank_quantize(data, ranks=(3, 2, 2), delta=delta,
                                          candidate_bits=(2, 4, 6), max_iter=10,
                                          learning_rate=100.0, max_backtracks=30,
                                          clip_ratio=0.1, verbose=False)
        objectives = np.array([row["objective"] for row in result.history])
        self.assertTrue(np.all(np.isfinite(objectives)))
        self.assertTrue(np.all(np.diff(objectives) <= 0.0))
        self.assertLessEqual(result.final_metrics["objective"], objectives[-1])
        expected_bits = sum(array.size * result.quantizers[name].bit_width
                            for name, array in zip(NAMES, list(result.master_factors) + [result.master_core]))
        self.assertEqual(result.model_bits, expected_bits)
        expected = np.sum((data - result.reconstructed) ** 2) + delta * expected_bits
        self.assertAlmostEqual(result.final_metrics["objective"], expected, places=10)
        self.assertAlmostEqual(result.model_compression_ratio, 32.0 * data.size / expected_bits)

    def test_fixed_component_policy_does_not_change_during_joint_search(self):
        data = np.random.default_rng(33).normal(size=(5, 4, 3))
        clips = {"A": 0.0, "B": 0.05, "C": 0.1, "G": 0.2}
        types = {"A": True, "B": False, "C": True, "G": False}
        result = qt.joint_lowrank_quantize(data, ranks=(2, 2, 2), candidate_bits=(2, 4, 6),
                                          clip_ratio=clips, symmetric=types, max_iter=5, verbose=False)
        self.assertEqual(result.ranks, (2, 2, 2))
        self.assertEqual(tuple(a.shape[1] for a in result.master_factors), result.ranks)
        self.assertEqual(result.master_core.shape, result.ranks)
        for name in NAMES:
            self.assertEqual(result.quantizers[name].clip_ratio, clips[name])
            self.assertEqual(result.quantizers[name].symmetric, types[name])

    def test_failed_large_steps_restore_float_components(self):
        data = np.random.default_rng(10).normal(size=(4, 3, 5))
        config = dict(ranks=(2, 2, 2), candidate_bits=(4,), delta=0.001,
                      full_scan_at_end=False, verbose=False)
        before = qt.joint_lowrank_quantize(data, max_iter=0, **config)
        after = qt.joint_lowrank_quantize(data, max_iter=1, learning_rate=1e12,
                                         max_backtracks=1, gradient_clip_norm=None, **config)
        for expected, actual in zip(before.master_factors, after.master_factors):
            np.testing.assert_array_equal(expected, actual)
        np.testing.assert_array_equal(before.master_core, after.master_core)
        np.testing.assert_array_equal(before.reconstructed, after.reconstructed)

    def test_zero_input_remains_finite_and_zero(self):
        result = qt.joint_lowrank_quantize(np.zeros((4, 3, 2)), max_iter=3, verbose=False)
        self.assertEqual(result.ranks, (1, 1, 1))
        np.testing.assert_array_equal(result.reconstructed, 0.0)
        self.assertTrue(np.isfinite(result.final_metrics["objective"]))

    def test_v3_round_trip_preserves_calibrated_float64_scales(self):
        data = np.random.default_rng(90).normal(size=(5, 4, 3))
        result = qt.joint_lowrank_quantize(data, ranks=(2, 2, 2), max_iter=2, verbose=False)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.npz")
            result.save_npz(path)
            with np.load(path, allow_pickle=False) as packed:
                self.assertEqual(int(packed["format_version"]), 3)
                self.assertEqual(packed["scales"].dtype, np.dtype("float64"))
                self.assertEqual(packed["zero_points"].dtype, np.dtype("int64"))
            factors, core, states, metadata = qt.load_quantized_model(path)
            for expected, actual in zip(result.factors, factors):
                np.testing.assert_array_equal(expected, actual)
            np.testing.assert_array_equal(result.core, core)
            np.testing.assert_array_equal(result.reconstructed, qt.tucker_reconstruct(core, factors))
            self.assertEqual(metadata["model_bits"], result.model_bits)
            self.assertEqual({n: s.scale for n, s in states.items()},
                             {n: s.scale for n, s in result.quantizers.items()})

    def test_asymmetric_32bit_zero_point_serializes_without_overflow(self):
        result = qt.joint_lowrank_quantize(-np.ones((2, 2, 2)), ranks=(1, 1, 1),
                                          candidate_bits=(32,), max_iter=0,
                                          symmetric=False, verbose=False)
        self.assertTrue(any(s.zero_point > np.iinfo(np.int32).max for s in result.quantizers.values()))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model32.npz")
            result.save_npz(path)
            factors, core, _, _ = qt.load_quantized_model(path)
            np.testing.assert_array_equal(qt.tucker_reconstruct(core, factors), result.reconstructed)

    def test_legacy_v2_decoder_remains_compatible(self):
        shapes = ((2, 1), (2, 1), (2, 1), (1, 1, 1))
        payload = np.concatenate([
            qt.pack_integer_codes(np.ones(shape, dtype=np.int64), 0, 2) for shape in shapes
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "legacy_v2.npz")
            np.savez_compressed(path, format_version=np.asarray(2, dtype=np.uint8),
                                payload=payload, bit_widths=np.full(4, 2, dtype=np.uint8),
                                scales=np.array([0.25, 0.5, 1.0, 2.0], dtype=np.float32),
                                zero_points=np.zeros(4, dtype=np.int32),
                                symmetric=np.zeros(4, dtype=np.uint8),
                                ranks=np.ones(3, dtype=np.int32),
                                original_shape=np.full(3, 2, dtype=np.int32))
            factors, core, _, metadata = qt.load_quantized_model(path)
            self.assertEqual(metadata["format_version"], 2)
            np.testing.assert_array_equal(qt.tucker_reconstruct(core, factors), np.full((2, 2, 2), 0.25))


class InputContractTests(unittest.TestCase):
    def test_invalid_quantization_inputs_are_rejected(self):
        for value in ([], [np.nan], [np.inf]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                qt._quantize_array(np.asarray(value), 4, False)
        for bit in (1, 33, 2.5, True):
            with self.subTest(bit=bit), self.assertRaises(ValueError):
                qt._quantize_array(np.ones(3), bit, False)
        for c in (-0.1, 1.0, np.nan):
            with self.subTest(c=c), self.assertRaises(ValueError):
                qt._quantize_array(np.ones(3), 4, False, clip_ratio=c)

    def test_invalid_solver_shapes_bits_ranks_and_weights_are_rejected(self):
        valid = np.ones((3, 3, 3))
        cases = [
            dict(data=np.empty((0, 2, 3))), dict(data=np.ones((3, 3))),
            dict(data=np.full((2, 2, 2), np.nan)), dict(candidate_bits=()),
            dict(candidate_bits=(2, 3.5)), dict(ranks=(1, 1)),
            dict(ranks=(4, 1, 1)), dict(delta=-0.01), dict(delta=np.nan),
            dict(max_iter=-1), dict(learning_rate=0.0),
            dict(backtrack_factor=1.0), dict(backtrack_factor=0.0),
        ]
        for case in cases:
            kwargs = dict(data=valid, max_iter=0, verbose=False)
            kwargs.update(case)
            with self.subTest(case=case), self.assertRaises(ValueError):
                qt.joint_lowrank_quantize(**kwargs)

    def test_obsolete_objective_modifiers_are_not_silently_accepted(self):
        for extra in (dict(l2_weight=1e-6), dict(bit_change_penalty=0.1),
                      dict(orthogonalize_interval=3), dict(rank_storage_weight=0.01)):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                qt.joint_lowrank_quantize(np.ones((3, 3, 3)), max_iter=0,
                                          verbose=False, **extra)


if __name__ == "__main__":
    unittest.main()
