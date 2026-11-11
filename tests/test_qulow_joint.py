import os
import tempfile
import unittest

import numpy as np

from qulow_joint import (
    _core_cross_gradient_half_sse,
    _core_delta_output_norm_sq,
    _factor_cross_gradient_half_sse,
    _factor_delta_output_norm_sq,
    joint_lowrank_quantize,
    load_quantized_model,
    pack_integer_codes,
    quantization_aware_gradients,
    select_ranks_hosvd_proxy,
    tucker_reconstruct,
    unpack_integer_codes,
)


class QulowJointTests(unittest.TestCase):
    def test_pack_round_trip_for_nonstandard_bits(self):
        rng = np.random.RandomState(1)
        for bit_width in (2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 32):
            qmin = -(2 ** (bit_width - 1) - 1)
            qmax = 2 ** (bit_width - 1) - 1
            codes = rng.randint(qmin, qmax + 1, size=(17, 13))
            packed = pack_integer_codes(codes, qmin, bit_width)
            restored = unpack_integer_codes(
                packed, codes.shape, qmin, bit_width
            )
            self.assertTrue(np.array_equal(codes, restored))

    def test_analytic_tucker_gradients(self):
        rng = np.random.RandomState(3)
        shape = (4, 5, 3)
        ranks = (2, 3, 2)
        target = rng.randn(*shape)
        factors = tuple(
            rng.randn(shape[mode], ranks[mode]) for mode in range(3)
        )
        core = rng.randn(*ranks)
        factor_gradients, core_gradient, _ = quantization_aware_gradients(
            target,
            factors,
            core,
            factors,
            core,
            l2_weight=0.0,
        )

        def loss(local_factors, local_core):
            reconstruction = tucker_reconstruct(local_core, local_factors)
            # Equation (5.4): gradients are for half SSE, not normalized MSE.
            return 0.5 * np.sum((reconstruction - target) ** 2)

        epsilon = 1e-6
        for mode in range(3):
            plus = [factor.copy() for factor in factors]
            minus = [factor.copy() for factor in factors]
            plus[mode][0, 0] += epsilon
            minus[mode][0, 0] -= epsilon
            finite_difference = (
                loss(plus, core) - loss(minus, core)
            ) / (2.0 * epsilon)
            self.assertAlmostEqual(
                finite_difference, factor_gradients[mode][0, 0], places=6
            )

        plus_core = core.copy()
        minus_core = core.copy()
        plus_core[0, 0, 0] += epsilon
        minus_core[0, 0, 0] -= epsilon
        finite_difference = (
            loss(factors, plus_core) - loss(factors, minus_core)
        ) / (2.0 * epsilon)
        self.assertAlmostEqual(
            finite_difference, core_gradient[0, 0, 0], places=6
        )


    def test_global_delta_sse_formula_is_exact(self):
        rng = np.random.RandomState(13)
        target = rng.randn(5, 4, 6)
        factors = (
            rng.randn(5, 3),
            rng.randn(4, 2),
            rng.randn(6, 3),
        )
        core = rng.randn(3, 2, 3)
        reconstruction = tucker_reconstruct(core, factors)
        residual = reconstruction - target
        current_sse = float(np.sum(residual ** 2))

        for mode in range(3):
            delta = 0.01 * rng.randn(*factors[mode].shape)
            cross_gradient = _factor_cross_gradient_half_sse(
                residual, factors, core, mode
            )
            predicted = (
                current_sse
                + 2.0 * float(np.sum(cross_gradient * delta))
                + _factor_delta_output_norm_sq(delta, factors, core, mode)
            )
            candidate_factors = list(factors)
            candidate_factors[mode] = candidate_factors[mode] + delta
            exact = float(
                np.sum((tucker_reconstruct(core, candidate_factors) - target) ** 2)
            )
            self.assertAlmostEqual(predicted, exact, places=9)

        core_delta = 0.01 * rng.randn(*core.shape)
        core_cross = _core_cross_gradient_half_sse(residual, factors)
        predicted_core = (
            current_sse
            + 2.0 * float(np.sum(core_cross * core_delta))
            + _core_delta_output_norm_sq(core_delta, factors)
        )
        exact_core = float(
            np.sum((tucker_reconstruct(core + core_delta, factors) - target) ** 2)
        )
        self.assertAlmostEqual(predicted_core, exact_core, places=9)

    def test_rank_objective_recovers_low_rank_scale(self):
        rng = np.random.RandomState(5)
        a = rng.randn(14, 4)
        b = rng.randn(12, 4)
        c = rng.randn(10, 4)
        tensor = sum(
            np.einsum("i,j,k->ijk", a[:, r], b[:, r], c[:, r])
            for r in range(4)
        )
        ranks, diagnostics = select_ranks_hosvd_proxy(
            tensor, max_rank=10, delta=1e-4
        )
        self.assertEqual(len(ranks), 3)
        self.assertTrue(all(1 <= rank <= 10 for rank in ranks))
        self.assertTrue(all(item.objective >= 0 for item in diagnostics))

    def test_joint_solver_and_packed_export(self):
        rng = np.random.RandomState(7)
        tensor = rng.randn(8, 7, 6)
        result = joint_lowrank_quantize(
            tensor,
            ranks=(3, 3, 3),
            candidate_bits=(2, 4, 8),
            max_iter=4,
            quantizer_update_interval=2,
            patience=10,
            verbose=False,
        )
        self.assertEqual(result.reconstructed.shape, tensor.shape)
        self.assertGreater(result.compression_ratio, 0)
        self.assertTrue(np.isfinite(result.reconstructed).all())

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.npz")
            result.save_npz(path)
            with np.load(path, allow_pickle=False) as packed:
                self.assertTrue(
                    {
                        "format_version",
                        "payload",
                        "bit_widths",
                        "scales",
                        "zero_points",
                        "symmetric",
                        "ranks",
                        "original_shape",
                    }.issubset(set(packed.files)),
                )
                self.assertNotIn("history", packed.files)
                self.assertNotIn("master_core", packed.files)
            factors, core, _, metadata = load_quantized_model(path)
            restored = tucker_reconstruct(core, factors)
            self.assertTrue(np.allclose(restored, result.reconstructed))
            self.assertEqual(metadata["format_version"], 3)
            self.assertEqual(metadata["ranks"], result.ranks)
            self.assertEqual(metadata["compressed_bits"], result.compressed_bits)


if __name__ == "__main__":
    unittest.main()
