"""QT-Joint, aligned with the November 2026 sections 3--6 specification.

One joint HOSVD spectral-tail/storage search fixes the Tucker dimensions.
The subsequent objective is exactly SSE + delta * sum(n_Z * b_Z), in bits.
Clipping proportions and quantizer families are fixed hyperparameters; only
the four bit widths and floating Tucker components are optimization variables.
STE proposals are accepted only after recalibrated true-forward validation.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-12
COMPONENT_NAMES = ("A", "B", "C", "G")
SCAN_ORDER = ("G", "A", "B", "C")
DEFAULT_BITS = tuple(range(2, 33))
ORIGINAL_BIT_WIDTH = 32
# Version 3: float64 scale, int64 zero point, uint8 bit width/family.
# These are encoded-storage diagnostics, NOT the optimization cost.
COMPONENT_METADATA_BITS = 64 + 64 + 8 + 8
GLOBAL_METADATA_BITS = 6 * 32 + 8  # shape/ranks plus format version


@dataclass
class RankSelectionState:
    mode: int
    matrix_rows: int
    matrix_columns: int
    selected_rank: int
    relative_tail_energy: float
    relative_storage: float
    objective: float
    tail_energy: float = 0.0
    factor_storage_bits: int = 0
    core_storage_bits: int = 0
    joint_proxy_objective: float = 0.0


@dataclass
class QuantizerState:
    bit_width: int
    qmin: int
    qmax: int
    scale: float
    zero_point: int
    symmetric: bool
    clip_quantile: float
    clip_min: float
    clip_max: float
    skewness: float
    excess_kurtosis: float
    local_mse: float
    output_distortion: float
    payload_bits: int
    metadata_bits: int
    storage_bits: int
    objective: float
    clip_ratio: float = 0.0
    model_bits: int = 0


@dataclass
class JointCompressionResult:
    # Deployment representation after dequantization.
    factors: Tuple[np.ndarray, np.ndarray, np.ndarray]
    core: np.ndarray
    # Floating master variables retained only for analysis/training continuation.
    master_factors: Tuple[np.ndarray, np.ndarray, np.ndarray]
    master_core: np.ndarray
    # Integer codes before bit packing.
    codes: Dict[str, np.ndarray]
    quantizers: Dict[str, QuantizerState]
    history: List[Dict[str, float]]
    reconstructed: np.ndarray
    compressed_bits: int
    compression_ratio: float
    ranks: Tuple[int, int, int]
    rank_diagnostics: Tuple[RankSelectionState, RankSelectionState, RankSelectionState]
    original_shape: Tuple[int, int, int]
    model_bits: int = 0
    model_compression_ratio: float = 0.0
    final_metrics: Dict[str, object] = field(default_factory=dict)
    search_diagnostics: Dict[str, object] = field(default_factory=dict)

    def save_npz(self, path: str) -> None:
        """Save the deployment model as one packed low-bit payload.

        Only information required for decoding is stored in the model file:
        packed integer codes, four float64 scales, four int64 zero-points,
        bit widths, quantizer-family flags, the original shape, and Tucker
        ranks.  Floating master parameters, training history, clipping
        statistics, and diagnostic losses are deliberately excluded.  They are
        written separately by ``save_history`` and the rank-selection CSV.

        ``model_bits`` is the paper's unpadded numerical cost.  ``compressed_bits``
        additionally counts byte padding and typed metadata; neither is an
        actual file size.  Use os.path.getsize(path) after saving for that metric.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        packed_parts = []
        bit_widths = []
        scales = []
        zero_points = []
        symmetric_flags = []
        for name in COMPONENT_NAMES:
            state = self.quantizers[name]
            packed_parts.append(
                pack_integer_codes(self.codes[name], state.qmin, state.bit_width)
            )
            bit_widths.append(state.bit_width)
            scales.append(state.scale)
            zero_points.append(state.zero_point)
            symmetric_flags.append(int(state.symmetric))

        payload = (
            np.concatenate(packed_parts).astype(np.uint8, copy=False)
            if packed_parts
            else np.empty(0, dtype=np.uint8)
        )
        np.savez_compressed(
            path,
            format_version=np.asarray(3, dtype=np.uint8),
            payload=payload,
            bit_widths=np.asarray(bit_widths, dtype=np.uint8),
            scales=np.asarray(scales, dtype=np.float64),
            zero_points=np.asarray(zero_points, dtype=np.int64),
            symmetric=np.asarray(symmetric_flags, dtype=np.uint8),
            ranks=np.asarray(self.ranks, dtype=np.int32),
            original_shape=np.asarray(self.original_shape, dtype=np.int32),
        )


def configure_utf8_output() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def normalize_minmax(
    data: np.ndarray, axis: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if np.iscomplexobj(data):
        raise ValueError("normalization expects real-valued input")
    data = np.asarray(data, dtype=np.float64)
    if data.size == 0 or not np.all(np.isfinite(data)):
        raise ValueError("normalization requires finite, nonempty input")
    if axis is None:
        minimum = np.min(data)
        value_range = np.max(data) - minimum
    else:
        minimum = np.min(data, axis=axis, keepdims=True)
        value_range = np.max(data, axis=axis, keepdims=True) - minimum
    if not np.all(np.isfinite(value_range)):
        raise ValueError("normalization range overflows; rescale input explicitly")
    scale = np.where(value_range > 0.0, value_range, 1.0)
    return (data - minimum) / scale, np.asarray(minimum), np.asarray(scale)


def unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    """Paper convention: mode 1 uses (C kron B), with Fortran-order columns."""
    tensor = np.asarray(tensor)
    if mode < 0 or mode >= tensor.ndim:
        raise ValueError(f"invalid mode {mode} for a {tensor.ndim}D tensor")
    return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1, order="F")


def mode_product(tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
    tensor = np.asarray(tensor)
    matrix = np.asarray(matrix)
    moved = np.moveaxis(tensor, mode, 0)
    if matrix.ndim != 2 or matrix.shape[1] != moved.shape[0]:
        raise ValueError(
            f"matrix shape {matrix.shape} is incompatible with mode-{mode} "
            f"dimension {moved.shape[0]}"
        )
    folded = moved.reshape(moved.shape[0], -1)
    product = matrix @ folded
    new_shape = (matrix.shape[0],) + moved.shape[1:]
    return np.moveaxis(product.reshape(new_shape), 0, mode)


def tucker_reconstruct(
    core: np.ndarray, factors: Sequence[np.ndarray]
) -> np.ndarray:
    if len(factors) != 3:
        raise ValueError("this implementation expects three Tucker factors")
    out = np.asarray(core, dtype=np.float64)
    for mode, factor in enumerate(factors):
        out = mode_product(out, factor, mode)
    return out


def tucker_core(data: np.ndarray, factors: Sequence[np.ndarray]) -> np.ndarray:
    core = np.asarray(data, dtype=np.float64)
    for mode, factor in enumerate(factors):
        core = mode_product(core, np.linalg.pinv(factor), mode)
    return core


def leading_singular_vectors(matrix: np.ndarray, rank: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    rank = _integer(rank, "rank", minimum=1)
    if matrix.ndim != 2 or rank > matrix.shape[0]:
        raise ValueError("rank must be in [1, matrix.shape[0]]")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("matrix contains NaN or infinite values")
    u, _, _ = np.linalg.svd(matrix, full_matrices=False)
    return _complete_left_basis(u, rank)


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _finite_nonnegative(value, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _validate_tensor(data: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(data):
        raise ValueError("QT-Joint expects real-valued input")
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 3 or data.size == 0 or any(n == 0 for n in data.shape):
        raise ValueError("expected a nonempty three-dimensional tensor")
    if not np.all(np.isfinite(data)):
        raise ValueError("input contains NaN or infinite values")
    return data


def _validate_bits(candidate_bits: Sequence[int]) -> Tuple[int, ...]:
    bits = tuple(sorted({_integer(b, "bit width", 2) for b in candidate_bits}))
    if not bits or any(b > 32 for b in bits):
        raise ValueError("candidate bits must be nonempty integers in [2, 32]")
    return bits


def _resolve_delta(delta, rate_weight, original_bit_width: int) -> float:
    if original_bit_width != ORIGINAL_BIT_WIDTH:
        raise ValueError(
            "this paper-aligned format uses a 32-bit raw-data storage baseline; "
            "original_bit_width must be 32 (internal calculations remain float64)"
        )
    if rate_weight is not None:
        if delta is not None:
            raise ValueError("specify delta or legacy rate_weight, not both")
        rate_weight = _finite_nonnegative(rate_weight, "rate_weight")
        warnings.warn(
            "rate_weight is deprecated; using delta=rate_weight/original_bit_width. "
            "The new objective excludes padding, metadata, and L2.",
            DeprecationWarning, stacklevel=3,
        )
        delta = rate_weight / original_bit_width
    return _finite_nonnegative(1e-4 if delta is None else delta, "delta")


def _complete_left_basis(u: np.ndarray, rank: int) -> np.ndarray:
    """Add zero-singular-value directions without a full right SVD."""
    if rank <= u.shape[1]:
        return u[:, :rank].copy()
    columns = [u[:, j].copy() for j in range(u.shape[1])]
    for index in range(u.shape[0]):
        v = np.zeros(u.shape[0], dtype=np.float64)
        v[index] = 1.0
        for _ in range(2):
            if columns:
                basis = np.column_stack(columns)
                v -= basis @ (basis.T @ v)
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            columns.append(v / norm)
            if len(columns) == rank:
                return np.column_stack(columns)
    raise RuntimeError("could not complete the left singular-vector basis")


def _mode_spectra(data: np.ndarray):
    bases, tails = [], []
    for mode in range(3):
        matrix = unfold(data, mode)
        u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
        squared = np.zeros(data.shape[mode], dtype=np.float64)
        squared[:singular_values.size] = singular_values ** 2
        if not np.all(np.isfinite(squared)):
            raise ValueError("spectral energy overflows; rescale input explicitly")
        tails.append(np.r_[np.cumsum(squared[::-1])[::-1], 0.0])
        bases.append(u)
    return bases, tails


def _select_joint_ranks(data, tails, delta, max_rank, min_rank, ranks=None):
    if ranks is not None:
        selected = tuple(_integer(r, "rank", 1) for r in ranks)
        if len(selected) != 3 or any(r > n for r, n in zip(selected, data.shape)):
            raise ValueError("ranks must contain three integers in [1, mode size]")
    else:
        lower = _integer(min_rank, "min_rank", 1)
        upper_bound = max(data.shape) if max_rank is None else _integer(max_rank, "max_rank", 1)
        upper = tuple(min(upper_bound, n) for n in data.shape)
        if any(lower > bound for bound in upper):
            raise ValueError("min_rank exceeds a searched mode's upper bound")
        r3 = np.arange(lower, upper[2] + 1, dtype=np.int64)
        best_score, best_cost, selected = float("inf"), float("inf"), None
        for r1 in range(lower, upper[0] + 1):
            for r2 in range(lower, upper[1] + 1):
                bits = 32 * (
                    data.shape[0] * r1 + data.shape[1] * r2
                    + data.shape[2] * r3 + r1 * r2 * r3
                )
                scores = tails[0][r1] + tails[1][r2] + tails[2][r3] + delta * bits
                index = int(np.argmin(scores))
                score, cost = float(scores[index]), int(bits[index])
                candidate = (r1, r2, int(r3[index]))
                if (score, cost, candidate) < (best_score, best_cost, selected or candidate):
                    best_score, best_cost, selected = score, cost, candidate
        assert selected is not None
    r1, r2, r3 = selected
    factor_bits = [32 * n * r for n, r in zip(data.shape, selected)]
    core_bits = 32 * r1 * r2 * r3
    total_bits = sum(factor_bits) + core_bits
    proxy = sum(float(tails[m][r]) for m, r in enumerate(selected)) + delta * total_bits
    energy = float(np.sum(data ** 2))
    diagnostics = []
    for mode, rank in enumerate(selected):
        tail = float(tails[mode][rank])
        diagnostics.append(RankSelectionState(
            mode=mode, matrix_rows=data.shape[mode],
            matrix_columns=data.size // data.shape[mode], selected_rank=rank,
            relative_tail_energy=tail / energy if energy > 0 else 0.0,
            relative_storage=total_bits / (32 * data.size), objective=float(proxy),
            tail_energy=tail, factor_storage_bits=factor_bits[mode],
            core_storage_bits=core_bits, joint_proxy_objective=float(proxy),
        ))
    return tuple(selected), tuple(diagnostics)


def select_ranks_hosvd_proxy(
    data: np.ndarray, delta: float = 1e-4, max_rank: Optional[int] = 64,
    min_rank: int = 1, ranks: Optional[Sequence[int]] = None,
) -> Tuple[Tuple[int, int, int], Tuple[RankSelectionState, ...]]:
    """Exactly search eq. (5.2) on the requested Cartesian rank range.

    The spectral bound is a surrogate, not actual Tucker loss.  A max_rank
    cap limits the searched set and makes no claim about omitted ranks.
    """
    data = _validate_tensor(data)
    delta = _finite_nonnegative(delta, "delta")
    _, tails = _mode_spectra(data)
    return _select_joint_ranks(data, tails, delta, max_rank, min_rank, ranks)


def select_ranks_original_objective(
    data: np.ndarray,
    max_rank: int = 64,
    rank_error_weight: Optional[float] = None,
    rank_storage_weight: Optional[float] = None,
    min_rank: int = 1,
    *, delta: float = 1e-4,
) -> Tuple[Tuple[int, int, int], Tuple[RankSelectionState, ...]]:
    """Deprecated name delegating to the new coupled HOSVD proxy."""
    if rank_error_weight is not None or rank_storage_weight is not None:
        raise ValueError("old independent rank weights are unsupported; use shared delta")
    warnings.warn("use select_ranks_hosvd_proxy", DeprecationWarning, stacklevel=2)
    return select_ranks_hosvd_proxy(data, delta, max_rank, min_rank)


def auto_ranks(
    data: np.ndarray,
    max_rank: int = 64,
    energy_threshold: Optional[float] = None,
    storage_weight: Optional[float] = None,
    *, delta: float = 1e-4,
) -> Tuple[int, int, int]:
    """Convenience wrapper for the canonical coupled HOSVD proxy."""
    if energy_threshold is not None or storage_weight is not None:
        raise ValueError("use delta; old rank heuristics are not the QT-Joint proxy")
    ranks, _ = select_ranks_hosvd_proxy(data, delta=delta, max_rank=max_rank)
    return ranks


def hosvd_initialize(
    data: np.ndarray, ranks: Sequence[int]
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    data = _validate_tensor(data)
    ranks = tuple(_integer(r, "rank", 1) for r in ranks)
    if len(ranks) != 3:
        raise ValueError("this implementation expects exactly three Tucker ranks")
    for mode, rank in enumerate(ranks):
        if rank < 1 or rank > data.shape[mode]:
            raise ValueError(
                f"rank {rank} is invalid for mode {mode} with size {data.shape[mode]}"
            )
    factors = tuple(
        leading_singular_vectors(unfold(data, mode), ranks[mode])
        for mode in range(3)
    )
    # HOSVD factors are orthonormal, so transposes are exact pseudoinverses.
    core = np.asarray(data, dtype=np.float64)
    for mode, factor in enumerate(factors):
        core = mode_product(core, factor.T, mode)
    return factors, core


def orthogonalize_tucker(
    factors: Sequence[np.ndarray], core: np.ndarray
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Standalone float-model utility; NOT used in QT-Joint optimization.

    QR can change the quantized reconstruction even when the floating
    reconstruction is unchanged.
    """
    normalized: List[np.ndarray] = []
    adjusted_core = np.asarray(core, dtype=np.float64)
    for mode, factor in enumerate(factors):
        q, r = np.linalg.qr(np.asarray(factor, dtype=np.float64), mode="reduced")
        normalized.append(q)
        adjusted_core = mode_product(adjusted_core, r, mode)
    return (normalized[0], normalized[1], normalized[2]), adjusted_core


def _shape_stats(values: np.ndarray) -> Tuple[float, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0 or not np.all(np.isfinite(flat)):
        raise ValueError("distribution statistics require finite, nonempty input")
    amplitude = float(np.max(np.abs(flat)))
    if amplitude == 0:
        return 0.0, 0.0
    scaled = flat / amplitude
    centered = scaled - np.mean(scaled)
    variance = float(np.mean(centered ** 2))
    if variance == 0:
        return 0.0, 0.0
    standardized = centered / math.sqrt(variance)
    skew = float(np.mean(standardized**3))
    excess_kurtosis = float(np.mean(standardized**4) - 3.0)
    return skew, excess_kurtosis


def _calibration_interval(values: np.ndarray, clip_ratio: float):
    """Eq. (3.4): c is the TOTAL removed proportion, not retained mass."""
    c = float(clip_ratio)
    if not np.isfinite(c) or not 0 <= c < 1:
        raise ValueError("clip_ratio must lie in [0, 1)")
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0 or not np.all(np.isfinite(flat)):
        raise ValueError("quantization requires finite, nonempty input")
    if c == 0.0:
        # Exact endpoints avoid needless interpolation, including overflow in
        # an intermediate subtraction for very large opposite-sign values.
        lower, upper = np.min(flat), np.max(flat)
    else:
        lower, upper = np.quantile(flat, [c / 2, 1 - c / 2], method="linear")
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("quantile calibration overflows; rescale input explicitly")
    lower, upper = min(float(lower), 0.0), max(float(upper), 0.0)
    if lower == upper == 0.0 and np.any(flat != 0.0):
        lower, upper = min(float(np.min(flat)), 0.0), max(float(np.max(flat)), 0.0)
    return lower, upper


def _payload_storage_bits(num_values: int, bit_width: int) -> int:
    return ((int(num_values) * int(bit_width) + 7) // 8) * 8


def _quantize_array(
    values: np.ndarray,
    bit_width: int,
    symmetric: bool,
    clip_quantile: Optional[float] = None,
    *,
    clip_ratio: float = 0.0,
    _interval=None,
    _stats=None,
) -> Tuple[np.ndarray, np.ndarray, QuantizerState]:
    """Deterministic linear-quantile, nearest-even affine quantization.

    clip_quantile is an explicit deprecated retained-proportion alias (1-c).
    Only the bit width varies in QT-Joint's discrete optimization.
    """
    if np.iscomplexobj(values):
        raise ValueError("quantization expects real-valued input")
    values = np.asarray(values, dtype=np.float64)
    bit_width = _integer(bit_width, "bit_width", 2)
    if bit_width > 32:
        raise ValueError("bit_width must lie in [2, 32]")
    if not isinstance(symmetric, (bool, np.bool_)):
        raise ValueError("symmetric must be a boolean")
    if clip_quantile is not None:
        if clip_ratio != 0.0:
            raise ValueError("use clip_ratio or legacy clip_quantile, not both")
        warnings.warn("clip_quantile is retained mass; use clip_ratio=1-clip_quantile",
                      DeprecationWarning, stacklevel=2)
        clip_ratio = 1.0 - float(clip_quantile)
    c = float(clip_ratio)
    if not np.isfinite(c) or not 0 <= c < 1:
        raise ValueError("clip_ratio must lie in [0, 1)")
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("quantization requires finite, nonempty input")
    clip_min, clip_max = (
        _calibration_interval(values, c) if _interval is None else _interval
    )
    skew, excess_kurtosis = _shape_stats(values) if _stats is None else _stats
    if symmetric:
        qmax = 2 ** (bit_width - 1) - 1
        qmin = -qmax
    else:
        qmin, qmax = 0, 2 ** bit_width - 1
    if clip_min == clip_max == 0.0:
        scale, zero_point = 1.0, 0
    elif symmetric:
        scale, zero_point = max(abs(clip_min), abs(clip_max)) / qmax, 0
    else:
        # Dividing before subtracting avoids an overflowing u-l for large data.
        scale = clip_max / qmax - clip_min / qmax
        scale = max(scale, float(np.nextafter(0.0, 1.0)))
        zero_point = int(np.clip(np.rint(qmin - clip_min / scale), qmin, qmax))
    scale = max(float(scale), float(np.nextafter(0.0, 1.0)))
    if not np.isfinite(scale):
        raise ValueError("quantizer scale is not finite; rescale input")
    with np.errstate(over="ignore", invalid="ignore"):
        # Round BEFORE adding zeta.  Nearest-even ties are not translation-invariant.
        codes = np.clip(np.rint(values / scale) + zero_point, qmin, qmax).astype(np.int64)
        dequantized = scale * (codes.astype(np.float64) - zero_point)
    local_mse = float(np.mean((values - dequantized) ** 2))
    payload_bits = _payload_storage_bits(values.size, bit_width)
    storage_bits = payload_bits + COMPONENT_METADATA_BITS
    state = QuantizerState(
        bit_width=int(bit_width),
        qmin=int(qmin),
        qmax=int(qmax),
        scale=float(scale),
        zero_point=int(zero_point),
        symmetric=bool(symmetric),
        clip_quantile=1.0 - c,
        clip_min=float(clip_min),
        clip_max=float(clip_max),
        skewness=float(skew),
        excess_kurtosis=float(excess_kurtosis),
        local_mse=local_mse,
        output_distortion=0.0,
        payload_bits=payload_bits,
        metadata_bits=COMPONENT_METADATA_BITS,
        storage_bits=storage_bits,
        objective=float("inf"),
        clip_ratio=c,
        model_bits=int(values.size * bit_width),
    )
    return codes, dequantized, state


def quantize_dequantize(
    values: np.ndarray,
    bit_width: int,
    symmetric: bool,
    clip_quantile: Optional[float] = None,
    *, clip_ratio: float = 0.0,
) -> Tuple[np.ndarray, QuantizerState]:
    """Compatibility wrapper returning dequantized values and quantizer state."""
    _, dequantized, state = _quantize_array(
        values, bit_width, symmetric, clip_quantile, clip_ratio=clip_ratio
    )
    return dequantized, state


def _state_configuration(state: QuantizerState) -> Tuple[int, bool, float]:
    return state.bit_width, state.symmetric, state.clip_ratio


def _apply_quantizer_configuration(
    values: np.ndarray, state: QuantizerState
) -> Tuple[np.ndarray, np.ndarray, QuantizerState]:
    return _quantize_array(
        values,
        bit_width=state.bit_width,
        symmetric=state.symmetric,
        clip_ratio=state.clip_ratio,
    )


def _candidate_quantizers(
    values: np.ndarray, candidate_bits: Sequence[int],
    symmetric: bool = False, clip_ratio: float = 0.0,
) -> List[Tuple[np.ndarray, np.ndarray, QuantizerState]]:
    bits = _validate_bits(candidate_bits)
    interval = _calibration_interval(values, clip_ratio)
    stats = _shape_stats(values)
    return [_quantize_array(values, bit, symmetric, clip_ratio=clip_ratio,
                            _interval=interval, _stats=stats) for bit in bits]


def _initial_quantizer(
    values: np.ndarray, candidate_bits: Sequence[int],
    symmetric: bool = False, clip_ratio: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, QuantizerState]:
    return _quantize_array(
        values,
        bit_width=max(_validate_bits(candidate_bits)),
        symmetric=symmetric,
        clip_ratio=clip_ratio,
    )


def model_storage_bits(quantizers: Mapping[str, QuantizerState]) -> int:
    """Encoded numerical payload + typed metadata; not the paper objective cost."""
    return int(
        GLOBAL_METADATA_BITS
        + sum(state.storage_bits for state in quantizers.values())
    )


def model_cost_bits(quantizers: Mapping[str, QuantizerState]) -> int:
    """The exact, unpadded Cost(r,b)=sum_Z n_Z b_Z in eqs. (4.4)/(5.24)."""
    return int(sum(state.model_bits for state in quantizers.values()))


def ste_mask(values: np.ndarray, state: QuantizerState) -> np.ndarray:
    """Eq. (5.10), inclusive ACTUAL representable endpoints, no calibration AD."""
    lower = state.scale * (state.qmin - state.zero_point)
    upper = state.scale * (state.qmax - state.zero_point)
    return ((np.asarray(values) >= lower) & (np.asarray(values) <= upper)).astype(np.float64)


def _factor_cross_gradient_half_sse(
    residual: np.ndarray,
    factors: Sequence[np.ndarray],
    core: np.ndarray,
    mode: int,
) -> np.ndarray:
    projected = residual
    for other_mode in range(3):
        if other_mode != mode:
            projected = mode_product(
                projected, factors[other_mode].T, other_mode
            )
    return unfold(projected, mode) @ unfold(core, mode).T


def _factor_delta_output_norm_sq(
    delta: np.ndarray,
    factors: Sequence[np.ndarray],
    core: np.ndarray,
    mode: int,
) -> float:
    sensitivity = _factor_sensitivity_gram(factors, core, mode)
    value = np.einsum("ir,rs,is->", delta, sensitivity, delta, optimize=True)
    return float(max(value, 0.0))


def _factor_sensitivity_gram(factors, core, mode):
    """N_Z N_Z.T by core-sized contractions, no explicit Kronecker array."""
    if mode not in (0, 1, 2):
        raise ValueError("mode must be 0, 1, or 2")
    weighted = core
    for other in range(3):
        if other != mode:
            weighted = mode_product(weighted, factors[other].T @ factors[other], other)
    gram = unfold(weighted, mode) @ unfold(core, mode).T
    return 0.5 * (gram + gram.T)


def _core_cross_gradient_half_sse(
    residual: np.ndarray, factors: Sequence[np.ndarray]
) -> np.ndarray:
    projected = residual
    for mode, factor in enumerate(factors):
        projected = mode_product(projected, factor.T, mode)
    return projected


def _core_delta_output_norm_sq(
    delta: np.ndarray, factors: Sequence[np.ndarray]
) -> float:
    weighted = np.asarray(delta, dtype=np.float64)
    for mode, factor in enumerate(factors):
        weighted = mode_product(weighted, factor.T @ factor, mode)
    value = float(np.sum(delta * weighted))
    return max(value, 0.0)


def _candidate_is_better(
    candidate_objective: float,
    candidate_state: QuantizerState,
    best_objective: float,
    best_state: QuantizerState,
    preferred_symmetric: bool,
) -> bool:
    # Compatibility helper.  Equal scores retain the incumbent, eq. (5.25).
    del candidate_state, best_state, preferred_symmetric
    return bool(np.isfinite(candidate_objective) and candidate_objective < best_objective)


def _component_settings(value, name: str):
    if isinstance(value, Mapping):
        if set(value) != set(COMPONENT_NAMES):
            raise ValueError(f"{name} mapping must contain exactly A, B, C, G")
        settings = dict(value)
    else:
        settings = {key: value for key in COMPONENT_NAMES}
    for key, item in settings.items():
        if name == "symmetric":
            if not isinstance(item, (bool, np.bool_)):
                raise ValueError("symmetric entries must be booleans")
            settings[key] = bool(item)
        else:
            item = float(item)
            if not np.isfinite(item) or not 0 <= item < 1:
                raise ValueError("clip_ratio entries must lie in [0, 1)")
            settings[key] = item
    return settings


def _skew_candidates(values, bits, current_bit, enabled, thresholds):
    if not enabled:
        return bits
    g1, g2, g3 = thresholds
    skew = abs(_shape_stats(values)[0])
    groups = ((2, 3, 4, 5, 6), (4, 5, 6, 7, 8),
              (6, 7, 8, 10, 12), (8, 10, 12, 16))
    group = groups[0 if skew < g1 else 1 if skew < g2 else 2 if skew < g3 else 3]
    return tuple(sorted((set(group) & set(bits)) | {current_bit}))


def select_quantizers_global(
    target: np.ndarray,
    master_factors: Sequence[np.ndarray],
    master_core: np.ndarray,
    candidate_bits: Sequence[int] = DEFAULT_BITS,
    rate_weight: Optional[float] = None,
    previous_quantizers: Optional[Mapping[str, QuantizerState]] = None,
    coordinate_sweeps: Optional[int] = 2,
    original_bit_width: int = ORIGINAL_BIT_WIDTH,
    bit_change_penalty: float = 0.0,
    *,
    delta: Optional[float] = None,
    clip_ratio=0.0,
    symmetric=False,
    use_skew_filter: bool = False,
    skew_thresholds=(0.5, 1.0, 2.0),
    diagnostics: Optional[Dict[str, object]] = None,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
    Dict[str, np.ndarray],
    Dict[str, QuantizerState],
]:
    """Eqs. (5.18)--(5.25): bit-only coordinate descent in G,A,B,C order.

    A coordinate's interval, residual projection, and Gram tensors are cached.
    Each chosen improvement is verified by a complete true forward evaluation.
    None sweeps means run to a no-change sweep, with the finite |H|**4 bound.
    No claim of joint global optimality or exact stationarity under roundoff.
    """
    target = _validate_tensor(target)
    bits = _validate_bits(candidate_bits)
    original_bit_width = _integer(original_bit_width, "original_bit_width", 1)
    delta = _resolve_delta(delta, rate_weight, original_bit_width)
    if bit_change_penalty != 0:
        raise ValueError("bit_change_penalty is not part of the QT-Joint objective")
    if coordinate_sweeps is not None:
        coordinate_sweeps = _integer(coordinate_sweeps, "coordinate_sweeps", 1)
    thresholds = tuple(float(x) for x in skew_thresholds)
    if (len(thresholds) != 3 or not all(np.isfinite(x) for x in thresholds)
            or not 0 <= thresholds[0] < thresholds[1] < thresholds[2]):
        raise ValueError("skew_thresholds must satisfy 0 <= gamma1 < gamma2 < gamma3")
    clipping = _component_settings(clip_ratio, "clip_ratio")
    families = _component_settings(symmetric, "symmetric")
    if len(master_factors) != 3:
        raise ValueError("exactly three factors are required")
    arrays = {
        "A": np.asarray(master_factors[0], dtype=np.float64),
        "B": np.asarray(master_factors[1], dtype=np.float64),
        "C": np.asarray(master_factors[2], dtype=np.float64),
        "G": np.asarray(master_core, dtype=np.float64),
    }
    if arrays["G"].ndim != 3 or any(
        arrays[name].shape != (target.shape[m], arrays["G"].shape[m])
        for m, name in enumerate(("A", "B", "C"))
    ):
        raise ValueError("factor/core dimensions do not match the target")
    if any(a.size == 0 or not np.all(np.isfinite(a)) for a in arrays.values()):
        raise ValueError("Tucker components must be finite and nonempty")
    if previous_quantizers is not None and set(previous_quantizers) != set(COMPONENT_NAMES):
        raise ValueError("previous_quantizers must contain exactly A, B, C, G")
    codes: Dict[str, np.ndarray] = {}
    dequantized: Dict[str, np.ndarray] = {}
    quantizers: Dict[str, QuantizerState] = {}
    for name in COMPONENT_NAMES:
        if previous_quantizers is not None:
            if previous_quantizers[name].bit_width not in bits:
                raise ValueError("every incumbent bit width must belong to candidate_bits")
            code, dq, state = _apply_quantizer_configuration(
                arrays[name], previous_quantizers[name]
            )
        else:
            code, dq, state = _initial_quantizer(
                arrays[name], bits, families[name], clipping[name]
            )
        codes[name] = code
        dequantized[name] = dq
        quantizers[name] = state

    dq_factors = (dequantized["A"], dequantized["B"], dequantized["C"])
    dq_core = dequantized["G"]
    reconstruction = tucker_reconstruct(dq_core, dq_factors)
    residual = reconstruction - target
    current_sse = float(np.sum(residual**2))
    current_bits = model_cost_bits(quantizers)
    current_objective = current_sse + delta * current_bits
    if not np.isfinite(current_objective):
        raise ValueError("initial objective is not finite; rescale input")
    diagnostics = {} if diagnostics is None else diagnostics
    for key in ("candidate_evaluations", "accepted_bit_updates", "full_objective_checks",
                "roundoff_rejections", "coordinate_passes"):
        diagnostics.setdefault(key, 0)
    diagnostics.setdefault("coordinate_trace", [])
    diagnostics["scan_order"] = list(SCAN_ORDER)
    diagnostics["used_skew_filter"] = bool(use_skew_filter)
    diagnostics["full_scan_stationary"] = False
    diagnostics["scan_limit_reached"] = False
    max_sweeps = len(bits) ** 4 if coordinate_sweeps is None else coordinate_sweeps
    terminated_without_changes = False
    for _ in range(max_sweeps):
        diagnostics["coordinate_passes"] += 1
        configuration_changed = False
        for name in SCAN_ORDER:
            current_state = quantizers[name]
            current_dq = dequantized[name]
            allowed = _skew_candidates(
                arrays[name], bits, current_state.bit_width, use_skew_filter, thresholds
            )
            # Scope of these caches is exactly this coordinate scan.
            interval = _calibration_interval(arrays[name], current_state.clip_ratio)
            stats = _shape_stats(arrays[name])
            if name != "G":
                mode = COMPONENT_NAMES.index(name)
                cross_gradient = _factor_cross_gradient_half_sse(
                    residual, dq_factors, dq_core, mode
                )
                gram = _factor_sensitivity_gram(dq_factors, dq_core, mode)
            else:
                cross_gradient = _core_cross_gradient_half_sse(
                    residual, dq_factors
                )
                grams = tuple(factor.T @ factor for factor in dq_factors)
            best_code = codes[name]
            best_dq = current_dq
            best_state = current_state
            best_increment = 0.0  # exact no-op incumbent; ties retain it
            for bit in allowed:
                diagnostics["candidate_evaluations"] += 1
                if bit == current_state.bit_width:
                    continue
                candidate_code, candidate_dq, candidate_state = _quantize_array(
                    arrays[name], bit, current_state.symmetric,
                    clip_ratio=current_state.clip_ratio, _interval=interval, _stats=stats,
                )
                change = candidate_dq - current_dq
                cross = float(np.sum(cross_gradient * change))
                if name != "G":
                    norm_sq = float(np.sum((change @ gram) * change))
                else:
                    weighted = change
                    for m, factor_gram in enumerate(grams):
                        weighted = mode_product(weighted, factor_gram, m)
                    norm_sq = float(np.sum(change * weighted))
                increment = 2.0 * cross + norm_sq + delta * arrays[name].size * (
                    bit - current_state.bit_width
                )
                if np.isfinite(increment) and increment < best_increment:
                    best_code = candidate_code
                    best_dq = candidate_dq
                    best_state = candidate_state
                    best_increment = float(increment)
            before_objective = current_objective
            if best_state.bit_width != current_state.bit_width:
                trial = dict(dequantized)
                trial[name] = best_dq
                trial_factors = (trial["A"], trial["B"], trial["C"])
                trial_reconstruction = tucker_reconstruct(trial["G"], trial_factors)
                trial_residual = trial_reconstruction - target
                trial_sse = float(np.sum(trial_residual ** 2))
                trial_bits = current_bits + arrays[name].size * (
                    best_state.bit_width - current_state.bit_width
                )
                trial_objective = trial_sse + delta * trial_bits
                diagnostics["full_objective_checks"] += 1
                reliability = 64 * np.finfo(np.float64).eps * max(
                    1.0, abs(current_objective), abs(trial_objective)
                )
                if (np.isfinite(trial_objective)
                        and trial_objective < current_objective - reliability):
                    codes[name], dequantized[name], quantizers[name] = (
                        best_code, best_dq, best_state
                    )
                    dq_factors, dq_core = trial_factors, trial["G"]
                    reconstruction, residual = trial_reconstruction, trial_residual
                    current_sse, current_bits = trial_sse, int(trial_bits)
                    current_objective = float(trial_objective)
                    configuration_changed = True
                    diagnostics["accepted_bit_updates"] += 1
                else:
                    diagnostics["roundoff_rejections"] += 1
            diagnostics["coordinate_trace"].append({
                "component": name, "old_bit": current_state.bit_width,
                "new_bit": quantizers[name].bit_width,
                "objective_before": before_objective,
                "objective_after": current_objective,
                "candidates": list(allowed),
            })
        if not configuration_changed:
            terminated_without_changes = True
            break
    diagnostics["terminated_without_changes"] = terminated_without_changes
    diagnostics["scan_limit_reached"] = not terminated_without_changes
    diagnostics["full_scan_stationary"] = bool(
        terminated_without_changes and not use_skew_filter
    )
    for state in quantizers.values():
        state.objective = float(current_objective)
    return dq_factors, dq_core, codes, quantizers


def _apply_quantizer_states(
    master_factors: Sequence[np.ndarray],
    master_core: np.ndarray,
    quantizers: Mapping[str, QuantizerState],
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
    Dict[str, np.ndarray],
    Dict[str, QuantizerState],
]:
    arrays = list(master_factors) + [master_core]
    outputs: List[np.ndarray] = []
    codes: Dict[str, np.ndarray] = {}
    refreshed: Dict[str, QuantizerState] = {}
    for name, array in zip(COMPONENT_NAMES, arrays):
        code, dq, state = _apply_quantizer_configuration(
            np.asarray(array, dtype=np.float64), quantizers[name]
        )
        codes[name] = code
        outputs.append(dq)
        refreshed[name] = state
    return (outputs[0], outputs[1], outputs[2]), outputs[3], codes, refreshed


def choose_quantizer(
    values: np.ndarray,
    candidate_bits: Sequence[int],
    rate_weight: Optional[float] = None,
    *, delta: Optional[float] = None, clip_ratio: float = 0.0, symmetric: bool = False,
) -> Tuple[np.ndarray, QuantizerState]:
    """Local compatibility helper.

    The joint solver does not use this helper because it selects quantizers from
    global tensor distortion.  It is retained for small standalone diagnostics.
    """
    values = np.asarray(values, dtype=np.float64)
    delta = _resolve_delta(delta, rate_weight, ORIGINAL_BIT_WIDTH)
    _, best_dq, best_state = _initial_quantizer(values, candidate_bits, symmetric, clip_ratio)
    best_state.objective = float(np.sum((values - best_dq) ** 2)) + delta * best_state.model_bits
    for _, dq, state in _candidate_quantizers(values, candidate_bits, symmetric, clip_ratio):
        objective = state.local_mse * values.size + delta * state.model_bits
        state.objective = objective
        if objective < best_state.objective:
            best_dq, best_state = dq, state
    assert best_dq is not None and best_state is not None
    return best_dq, best_state


def quantized_components(
    factors: Sequence[np.ndarray],
    core: np.ndarray,
    candidate_bits: Sequence[int],
    rate_weight: float,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
    Dict[str, QuantizerState],
]:
    outputs = []
    quantizers = {}
    for name, array in zip(COMPONENT_NAMES, list(factors) + [core]):
        dq, state = choose_quantizer(array, candidate_bits, rate_weight)
        outputs.append(dq)
        quantizers[name] = state
    return (outputs[0], outputs[1], outputs[2]), outputs[3], quantizers


def _l2_loss_and_gradients(
    factors: Sequence[np.ndarray], core: np.ndarray, l2_weight: float
) -> Tuple[float, Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    arrays = list(factors) + [core]
    if l2_weight <= 0:
        zeros = tuple(np.zeros_like(factor) for factor in factors)
        return 0.0, zeros, np.zeros_like(core)
    loss = 0.0
    gradients = []
    for array in arrays:
        loss += 0.5 * l2_weight * float(np.mean(array**2))
        gradients.append(l2_weight * array / max(array.size, 1))
    return loss, (gradients[0], gradients[1], gradients[2]), gradients[3]


def quantization_aware_gradients(
    target: np.ndarray,
    master_factors: Sequence[np.ndarray],
    master_core: np.ndarray,
    dq_factors: Sequence[np.ndarray],
    dq_core: np.ndarray,
    l2_weight: float = 0.0,
    *,
    quantizers: Optional[Mapping[str, QuantizerState]] = None,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, Dict[str, float]
]:
    """Eqs. (5.12)--(5.15), gradients of F=one-half SSE, not MSE.

    Quantizer scales/zero points are constants for this derivative only.
    If states are supplied, multiply by the interval-inclusion STE masks.
    Without states, return ordinary gradients with respect to supplied DQ
    components; this is useful for independent finite-difference verification.
    """
    if l2_weight != 0:
        raise ValueError("L2 regularization is not part of the QT-Joint objective")
    reconstruction = tucker_reconstruct(dq_core, dq_factors)
    residual = reconstruction - target
    num_values = max(target.size, 1)
    mse = float(np.sum(residual**2) / num_values)

    factor_gradients = []
    for mode in range(3):
        projected = residual
        for other_mode in range(3):
            if other_mode != mode:
                projected = mode_product(
                    projected, dq_factors[other_mode].T, other_mode
                )
        gradient = unfold(projected, mode) @ unfold(dq_core, mode).T
        if quantizers is not None:
            gradient = gradient * ste_mask(master_factors[mode], quantizers[COMPONENT_NAMES[mode]])
        factor_gradients.append(gradient)

    core_gradient = residual
    for mode, factor in enumerate(dq_factors):
        core_gradient = mode_product(core_gradient, factor.T, mode)
    if quantizers is not None:
        core_gradient = core_gradient * ste_mask(master_core, quantizers["G"])
    return (
        factor_gradients[0],
        factor_gradients[1],
        factor_gradients[2],
    ), core_gradient, {
        "reconstruction_mse": mse,
        "reconstruction_sse": float(mse * num_values),
        "half_sse": float(0.5 * mse * num_values),
        "l2_loss": 0.0,
    }


# Backward-compatible name retained for older imports.
def finite_difference_gradients(
    target: np.ndarray,
    factors: Sequence[np.ndarray],
    core: np.ndarray,
    candidate_bits: Sequence[int],
    rate_weight: float,
    l2_weight: float,
):
    dq_factors, dq_core, _, quantizers = select_quantizers_global(
        target,
        factors,
        core,
        candidate_bits,
        rate_weight,
        coordinate_sweeps=1,
    )
    warnings.warn(
        "finite_difference_gradients is a legacy name for analytic STE gradients, "
        "not a numerical derivative of the quantized objective.",
        DeprecationWarning, stacklevel=2,
    )
    factor_grads, core_grad, metrics = quantization_aware_gradients(
        target, factors, core, dq_factors, dq_core, l2_weight, quantizers=quantizers
    )
    compressed_bits = model_storage_bits(quantizers)
    metrics.update(
        {
            "quantization_mse": float(
                sum(state.local_mse for state in quantizers.values())
            ),
            "compressed_bits": float(compressed_bits),
            "compression_ratio": float(
                target.size * ORIGINAL_BIT_WIDTH / max(compressed_bits, 1)
            ),
            "objective": metrics["reconstruction_sse"]
            + (rate_weight / ORIGINAL_BIT_WIDTH) * model_cost_bits(quantizers),
        }
    )
    return factor_grads, core_grad, quantizers, metrics


def _clip_gradients(
    factor_gradients: Sequence[np.ndarray],
    core_gradient: np.ndarray,
    max_norm: Optional[float],
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, float]:
    squared_norm = sum(float(np.sum(gradient**2)) for gradient in factor_gradients)
    squared_norm += float(np.sum(core_gradient**2))
    total_norm = math.sqrt(max(squared_norm, 0.0))
    if max_norm is None or max_norm <= 0 or total_norm <= max_norm:
        return (
            factor_gradients[0],
            factor_gradients[1],
            factor_gradients[2],
        ), core_gradient, total_norm
    scale = max_norm / (total_norm + EPS)
    clipped_factors = tuple(gradient * scale for gradient in factor_gradients)
    return clipped_factors, core_gradient * scale, total_norm


def _evaluate_model(
    target: np.ndarray,
    master_factors: Sequence[np.ndarray],
    master_core: np.ndarray,
    quantizers: Mapping[str, QuantizerState],
    delta: float,
    original_bit_width: int = ORIGINAL_BIT_WIDTH,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
    Dict[str, np.ndarray],
    Dict[str, QuantizerState],
    np.ndarray,
    Dict[str, float],
]:
    dq_factors, dq_core, codes, refreshed = _apply_quantizer_states(
        master_factors, master_core, quantizers
    )
    reconstructed = tucker_reconstruct(dq_core, dq_factors)
    float_reconstructed = tucker_reconstruct(master_core, master_factors)
    quantized_mse = float(np.mean((reconstructed - target) ** 2))
    quantized_sse = float(np.sum((reconstructed - target) ** 2))
    float_mse = float(np.mean((float_reconstructed - target) ** 2))
    # Report the isolated output-space distortion of every deployed component.
    # This quantity is substantially more meaningful than local parameter MSE
    # because Tucker components have different sensitivities and scaling roles.
    for mode, name in enumerate(("A", "B", "C")):
        component_change = dq_factors[mode] - master_factors[mode]
        refreshed[name].output_distortion = (
            _factor_delta_output_norm_sq(component_change, dq_factors, dq_core, mode)
            / max(target.size, 1)
        )
    core_delta = dq_core - master_core
    refreshed["G"].output_distortion = (
        _core_delta_output_norm_sq(core_delta, dq_factors)
        / max(target.size, 1)
    )

    compressed_bits = model_storage_bits(refreshed)
    model_bits = model_cost_bits(refreshed)
    rate_ratio = compressed_bits / max(target.size * original_bit_width, 1)
    objective = quantized_sse + delta * model_bits
    for state in refreshed.values():
        state.objective = float(objective)
    metrics = {
        "float_reconstruction_mse": float_mse,
        "reconstruction_mse": quantized_mse,
        "reconstruction_sse": quantized_sse,
        "half_sse": 0.5 * quantized_sse,
        "relative_error": relative_error(target, reconstructed),
        "quantization_gap": quantized_mse - float_mse,
        "component_local_quantization_mse": float(
            sum(state.local_mse for state in refreshed.values())
        ),
        "compressed_bits": float(compressed_bits),
        "model_bits": int(model_bits),
        "model_compression_ratio": float(target.size * original_bit_width / model_bits),
        "storage_penalty": float(delta * model_bits),
        "delta": float(delta),
        "rate_ratio": float(rate_ratio),
        "compression_ratio": float(1.0 / max(rate_ratio, EPS)),
        "l2_loss": 0.0,
        "objective": float(objective),
    }
    for name, state in refreshed.items():
        metrics[f"{name}_bit"] = float(state.bit_width)
        metrics[f"{name}_symmetric"] = float(int(state.symmetric))
        metrics[f"{name}_clip_quantile"] = float(state.clip_quantile)
        metrics[f"{name}_clip_ratio"] = float(state.clip_ratio)
    return dq_factors, dq_core, codes, refreshed, reconstructed, metrics


def joint_lowrank_quantize(
    data: np.ndarray,
    ranks: Optional[Sequence[int]] = None,
    candidate_bits: Sequence[int] = DEFAULT_BITS,
    max_iter: int = 80,
    learning_rate: float = 0.05,
    rate_weight: Optional[float] = None,
    l2_weight: float = 0.0,
    patience: int = 15,
    min_relative_improvement: float = 1e-6,
    rank_max: int = 64,
    rank_error_weight: Optional[float] = None,
    rank_storage_weight: Optional[float] = None,
    quantizer_update_interval: int = 1,
    quantizer_coordinate_sweeps: Optional[int] = 2,
    orthogonalize_interval: Optional[int] = None,
    gradient_clip_norm: Optional[float] = None,
    bit_change_penalty: float = 0.0,
    original_bit_width: int = ORIGINAL_BIT_WIDTH,
    seed: int = 7,
    verbose: bool = True,
    *,
    delta: Optional[float] = None,
    clip_ratio=0.0,
    symmetric=False,
    use_skew_filter: bool = False,
    skew_thresholds=(0.5, 1.0, 2.0),
    full_scan_at_end: bool = True,
    parameter_steps: int = 1,
    max_backtracks: int = 20,
    backtrack_factor: float = 0.5,
    min_absolute_improvement: float = 1e-10,
) -> JointCompressionResult:
    """Algorithm 1 with recalibrated true-loss backtracking and fixed ranks.

    The guaranteed sequence is the accepted OBJECTIVE, not convergence of
    parameters, differentiable stationarity, or a joint global optimum.
    seed is retained for API compatibility; the canonical solver adds no jitter.
    max_backtracks counts reductions after the initial trial.
    """
    data = _validate_tensor(data)
    candidate_bits = _validate_bits(candidate_bits)
    original_bit_width = _integer(original_bit_width, "original_bit_width", 1)
    delta = _resolve_delta(delta, rate_weight, original_bit_width)
    max_iter = _integer(max_iter, "max_iter", 0)
    patience = _integer(patience, "patience", 1)
    parameter_steps = _integer(parameter_steps, "parameter_steps", 0)
    max_backtracks = _integer(max_backtracks, "max_backtracks", 0)
    quantizer_update_interval = _integer(quantizer_update_interval, "quantizer_update_interval", 1)
    if quantizer_coordinate_sweeps is not None:
        quantizer_coordinate_sweeps = _integer(quantizer_coordinate_sweeps, "quantizer_coordinate_sweeps", 1)
    learning_rate = float(learning_rate)
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not np.isfinite(backtrack_factor) or not 0 < backtrack_factor < 1:
        raise ValueError("backtrack_factor must lie in (0, 1)")
    if l2_weight != 0 or bit_change_penalty != 0:
        raise ValueError("nonzero L2 or bit-change penalties change the specified objective")
    if orthogonalize_interval is not None:
        raise ValueError("periodic QR is not part of QT-Joint; use orthogonalize_interval=None")
    if rank_error_weight is not None or rank_storage_weight is not None:
        raise ValueError("old independent rank weights are unsupported; use shared delta")
    if gradient_clip_norm is not None:
        gradient_clip_norm = float(gradient_clip_norm)
        if not np.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive or None")
    min_relative_improvement = _finite_nonnegative(min_relative_improvement, "min_relative_improvement")
    min_absolute_improvement = _finite_nonnegative(min_absolute_improvement, "min_absolute_improvement")
    thresholds = tuple(float(x) for x in skew_thresholds)
    if (len(thresholds) != 3 or not all(np.isfinite(x) for x in thresholds)
            or not 0 <= thresholds[0] < thresholds[1] < thresholds[2]):
        raise ValueError("skew_thresholds must satisfy 0 <= gamma1 < gamma2 < gamma3")
    clipping = _component_settings(clip_ratio, "clip_ratio")
    families = _component_settings(symmetric, "symmetric")

    # Reuse one set of mode SVDs for both rank search and HOSVD initialization.
    bases, tails = _mode_spectra(data)
    selected_ranks, rank_diagnostics = _select_joint_ranks(
        data, tails, delta, rank_max, 1, ranks
    )
    master_factors = tuple(_complete_left_basis(u, r) for u, r in zip(bases, selected_ranks))
    master_core = data
    for mode, factor in enumerate(master_factors):
        master_core = mode_product(master_core, factor.T, mode)
    quantizers = {}
    for name, array in zip(COMPONENT_NAMES, list(master_factors) + [master_core]):
        _, _, quantizers[name] = _initial_quantizer(
            array, candidate_bits, families[name], clipping[name]
        )
    dq_factors, dq_core, codes, quantizers, reconstructed, metrics = _evaluate_model(
        data, master_factors, master_core, quantizers, delta, original_bit_width
    )
    if not np.isfinite(metrics["objective"]):
        raise ValueError("initial objective is not finite; rescale input explicitly")
    metrics.update(iteration=0.0, phase="initial", learning_rate=0.0,
                   gradient_norm=0.0, quantizer_updated=0.0, parameter_accepted=0.0)
    history = [dict(metrics)]
    search = {
        "parameter_trials": 0, "accepted_parameter_steps": 0,
        "rejected_parameter_steps": 0, "backtracking_reductions": 0,
        "parameter_trace": [], "candidate_evaluations": 0,
        "accepted_bit_updates": 0, "full_objective_checks": 0,
        "roundoff_rejections": 0, "coordinate_passes": 0,
        "coordinate_trace": [], "scan_order": list(SCAN_ORDER),
        "full_scan_stationary": False, "final_full_scan_requested": bool(full_scan_at_end),
        "skew_filter_enabled": bool(use_skew_filter),
        "rank_proxy_objective": float(rank_diagnostics[0].joint_proxy_objective),
        "rank_search_capped": ranks is None and rank_max is not None and any(n > rank_max for n in data.shape),
        "fixed_ranks": list(selected_ranks),
    }
    best_objective = float(metrics["objective"])
    best_snapshot = (tuple(x.copy() for x in master_factors), master_core.copy(), copy.deepcopy(quantizers))
    stalled, iteration, stop_reason = 0, 0, "max_iter"
    for iteration in range(1, max_iter + 1):
        previous_objective = float(metrics["objective"])
        accepted_steps, gradient_norm, accepted_step_size = 0, 0.0, 0.0
        current_sse = float(metrics["reconstruction_sse"])
        for _ in range(parameter_steps):
            dq_factors, dq_core, codes, quantizers = _apply_quantizer_states(
                master_factors, master_core, quantizers
            )
            factor_gradients, core_gradient, _ = quantization_aware_gradients(
                data, master_factors, master_core, dq_factors, dq_core,
                quantizers=quantizers,
            )
            factor_gradients, core_gradient, gradient_norm = _clip_gradients(
                factor_gradients, core_gradient, gradient_clip_norm
            )
            step_size = learning_rate / math.sqrt(iteration)
            before_sse, accepted = current_sse, False
            for attempt in range(max_backtracks + 1):
                search["parameter_trials"] += 1
                with np.errstate(over="ignore", invalid="ignore"):
                    trial_factors = tuple(x - step_size * g for x, g in zip(master_factors, factor_gradients))
                    trial_core = master_core - step_size * core_gradient
                try:
                    with np.errstate(over="ignore", invalid="ignore"):
                        trial_dq_factors, trial_dq_core, trial_codes, trial_states = _apply_quantizer_states(
                            trial_factors, trial_core, quantizers
                        )
                        trial_reconstruction = tucker_reconstruct(trial_dq_core, trial_dq_factors)
                        trial_sse = float(np.sum((trial_reconstruction - data) ** 2))
                except (ValueError, FloatingPointError, OverflowError):
                    trial_sse = float("inf")
                # No positive tolerance: accepting even a tiny rise breaks theorem 6.1.
                if np.isfinite(trial_sse) and trial_sse <= current_sse:
                    master_factors, master_core = trial_factors, trial_core
                    dq_factors, dq_core, codes, quantizers = (
                        trial_dq_factors, trial_dq_core, trial_codes, trial_states
                    )
                    reconstructed, current_sse = trial_reconstruction, trial_sse
                    accepted, accepted_step_size = True, float(step_size)
                    accepted_steps += 1
                    search["accepted_parameter_steps"] += 1
                    search["full_scan_stationary"] = False
                    break
                if attempt < max_backtracks:
                    step_size *= backtrack_factor
                    search["backtracking_reductions"] += 1
            if not accepted:
                search["rejected_parameter_steps"] += 1
            search["parameter_trace"].append({
                "iteration": iteration, "sse_before": before_sse,
                "sse_after": current_sse, "accepted": accepted,
                "step_size": float(step_size) if accepted else 0.0,
            })
        quantizer_updated = iteration % quantizer_update_interval == 0
        if quantizer_updated:
            dq_factors, dq_core, codes, quantizers = select_quantizers_global(
                data, master_factors, master_core, candidate_bits,
                previous_quantizers=quantizers,
                coordinate_sweeps=quantizer_coordinate_sweeps,
                original_bit_width=original_bit_width, delta=delta,
                use_skew_filter=use_skew_filter, skew_thresholds=thresholds,
                diagnostics=search,
            )
        dq_factors, dq_core, codes, quantizers, reconstructed, metrics = _evaluate_model(
            data, master_factors, master_core, quantizers, delta, original_bit_width
        )
        if metrics["objective"] > previous_objective:
            raise RuntimeError("internal error: accepted QT-Joint objective increased")
        metrics.update(
            iteration=float(iteration), phase="alternating",
            learning_rate=accepted_step_size, gradient_norm=float(gradient_norm),
            quantizer_updated=float(quantizer_updated), parameter_accepted=float(accepted_steps),
        )
        history.append(dict(metrics))
        if metrics["objective"] <= best_objective:
            best_objective = float(metrics["objective"])
            best_snapshot = (tuple(x.copy() for x in master_factors), master_core.copy(), copy.deepcopy(quantizers))
        if verbose:
            summary = ",".join(f"{name}:{quantizers[name].bit_width}" for name in SCAN_ORDER)
            print(f"iter={iteration:03d} sse={metrics['reconstruction_sse']:.6e} "
                  f"obj={metrics['objective']:.6e} model_ratio={metrics['model_compression_ratio']:.3f}x "
                  f"bits=[{summary}] accepted_steps={accepted_steps}")
        improvement = previous_objective - metrics["objective"]
        threshold = max(min_absolute_improvement,
                        min_relative_improvement * max(abs(previous_objective), 1e-12))
        stalled = stalled + 1 if improvement <= threshold else 0
        if stalled >= patience:
            stop_reason = "small_improvement_patience"
            break
    master_factors, master_core, quantizers = best_snapshot
    if full_scan_at_end:
        _, _, _, quantizers = select_quantizers_global(
            data, master_factors, master_core, candidate_bits,
            previous_quantizers=quantizers, coordinate_sweeps=None,
            original_bit_width=original_bit_width, delta=delta,
            use_skew_filter=False, skew_thresholds=thresholds, diagnostics=search,
        )
    final_factors, final_core, final_codes, final_quantizers, final_reconstructed, final_metrics = _evaluate_model(
        data, master_factors, master_core, quantizers, delta, original_bit_width
    )
    final_metrics["stop_reason"] = stop_reason
    final_metrics["outer_iterations"] = int(iteration)
    final_metrics["full_scan_stationary"] = bool(search["full_scan_stationary"])
    if full_scan_at_end:
        final_row = dict(final_metrics)
        final_row.update(iteration=float(iteration), phase="final_full_scan",
                         learning_rate=0.0, gradient_norm=0.0,
                         quantizer_updated=1.0, parameter_accepted=0.0)
        history.append(final_row)
    search["stop_reason"] = stop_reason
    search["stationarity_note"] = (
        "A no-change complete bit sweep is numerical coordinate stationarity only; "
        "no floating-parameter stationarity or joint global optimality is claimed."
    )
    return JointCompressionResult(
        factors=final_factors, core=final_core,
        master_factors=master_factors, master_core=master_core,
        codes=final_codes, quantizers=final_quantizers, history=history,
        reconstructed=final_reconstructed, compressed_bits=int(final_metrics["compressed_bits"]),
        compression_ratio=float(final_metrics["compression_ratio"]), ranks=selected_ranks,
        rank_diagnostics=rank_diagnostics, original_shape=tuple(data.shape),
        model_bits=int(final_metrics["model_bits"]),
        model_compression_ratio=float(final_metrics["model_compression_ratio"]),
        final_metrics=final_metrics, search_diagnostics=search,
    )


def relative_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Relative Frobenius error, without a scale-dependent epsilon bias."""
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have the same shape")
    magnitude = float(np.max(np.abs(reference))) if reference.size else 0.0
    if magnitude == 0.0:
        return 0.0 if np.array_equal(reference, estimate) else float("inf")
    return float(np.linalg.norm(((reference - estimate) / magnitude).ravel())
                 / np.linalg.norm((reference / magnitude).ravel()))


def pack_integer_codes(
    codes: np.ndarray, qmin: int, bit_width: int
) -> np.ndarray:
    """Pack signed/unsigned integer codes into a byte stream."""
    bit_width = _integer(bit_width, "bit_width", 2)
    if bit_width > 32:
        raise ValueError("bit_width must lie in [2, 32]")
    original = np.asarray(codes)
    if not np.issubdtype(original.dtype, np.integer):
        raise ValueError("codes must have integer dtype")
    flat = np.asarray(codes, dtype=np.int64).reshape(-1)
    shifted = flat - int(qmin)
    max_code = 2**bit_width - 1
    if np.any(shifted < 0) or np.any(shifted > max_code):
        raise ValueError("codes lie outside the range represented by bit_width")
    if flat.size == 0:
        return np.empty(0, dtype=np.uint8)

    # Chunk length is aligned so every non-final chunk ends on a byte boundary.
    alignment = 8 // math.gcd(bit_width, 8)
    values_per_chunk = max(alignment, (1_000_000 // alignment) * alignment)
    positions = np.arange(bit_width - 1, -1, -1, dtype=np.uint64)
    chunks = []
    for start in range(0, shifted.size, values_per_chunk):
        part = shifted[start : start + values_per_chunk].astype(np.uint64)
        bits = ((part[:, None] >> positions[None, :]) & 1).astype(np.uint8)
        chunks.append(np.packbits(bits.reshape(-1), bitorder="big"))
    return np.concatenate(chunks).astype(np.uint8, copy=False)


def unpack_integer_codes(
    packed: np.ndarray,
    shape: Sequence[int],
    qmin: int,
    bit_width: int,
) -> np.ndarray:
    bit_width = _integer(bit_width, "bit_width", 2)
    if bit_width > 32:
        raise ValueError("bit_width must lie in [2, 32]")
    shape = tuple(_integer(n, "shape dimension", 0) for n in shape)
    count = math.prod(shape)
    if count == 0:
        return np.empty(tuple(shape), dtype=np.int64)
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8), bitorder="big")
    required = count * bit_width
    if bits.size < required:
        raise ValueError("packed payload is shorter than expected")
    bits = bits[:required].reshape(count, bit_width)
    positions = np.arange(bit_width - 1, -1, -1, dtype=np.uint64)
    shifted = np.sum(bits.astype(np.uint64) << positions[None, :], axis=1)
    return (shifted.astype(np.int64) + int(qmin)).reshape(tuple(shape))


def load_quantized_model(
    path: str,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
    Dict[str, QuantizerState],
    Dict[str, object],
]:
    """Load a packed deployment model saved by ``JointCompressionResult.save_npz``.

    Version 3 preserves float64 scales and int64 zero points exactly.
    Versions 1 and 2 remain decodable.  Deployment files intentionally omit
    training master arrays and clipping hyperparameters; loading is not a
    training-checkpoint restore.
    """
    with np.load(path, allow_pickle=False) as loaded:
        if "payload" in loaded.files:
            format_version = int(loaded["format_version"]) if "format_version" in loaded.files else 2
            if format_version not in (2, 3):
                raise ValueError(f"unsupported model format version {format_version}")
            ranks = tuple(int(value) for value in loaded["ranks"])
            original_shape = tuple(int(value) for value in loaded["original_shape"])
            if (len(ranks) != 3 or len(original_shape) != 3
                    or any(n < 1 for n in original_shape)
                    or any(r < 1 or r > n for r, n in zip(ranks, original_shape))):
                raise ValueError("invalid stored tensor shape or Tucker ranks")
            bit_widths = [int(value) for value in loaded["bit_widths"]]
            scales = [float(value) for value in loaded["scales"]]
            zero_points = [int(value) for value in loaded["zero_points"]]
            symmetric_flags = [bool(value) for value in loaded["symmetric"]]
            if not all(len(items) == 4 for items in (
                bit_widths, scales, zero_points, symmetric_flags
            )):
                raise ValueError("packed model must contain four quantizer records")
            if any(b < 2 or b > 32 for b in bit_widths):
                raise ValueError("stored bit widths must lie in [2, 32]")
            if any(not np.isfinite(s) or s <= 0 for s in scales):
                raise ValueError("stored quantizer scales must be finite and positive")
            component_metadata_bits = 144 if format_version == 3 else 80
            component_shapes = (
                (original_shape[0], ranks[0]),
                (original_shape[1], ranks[1]),
                (original_shape[2], ranks[2]),
                ranks,
            )
            payload = np.asarray(loaded["payload"], dtype=np.uint8)
            arrays: Dict[str, np.ndarray] = {}
            quantizers: Dict[str, QuantizerState] = {}
            offset = 0
            for index, (name, shape) in enumerate(zip(COMPONENT_NAMES, component_shapes)):
                bit_width = bit_widths[index]
                symmetric = symmetric_flags[index]
                qmin = -(2 ** (bit_width - 1) - 1) if symmetric else 0
                qmax = 2 ** (bit_width - 1) - 1 if symmetric else 2**bit_width - 1
                if not qmin <= zero_points[index] <= qmax or (symmetric and zero_points[index] != 0):
                    raise ValueError("stored zero point is outside the integer range")
                payload_bits = _payload_storage_bits(int(np.prod(shape)), bit_width)
                payload_bytes = payload_bits // 8
                part = payload[offset : offset + payload_bytes]
                if part.size != payload_bytes:
                    raise ValueError(f"payload for component {name} is truncated")
                offset += payload_bytes
                codes = unpack_integer_codes(part, shape, qmin, bit_width)
                if np.any(codes > qmax):
                    raise ValueError("payload contains an unused symmetric integer code")
                scale = scales[index]
                zero_point = zero_points[index]
                arrays[name] = (codes.astype(np.float64) - zero_point) * scale
                quantizers[name] = QuantizerState(
                    bit_width=bit_width,
                    qmin=qmin,
                    qmax=qmax,
                    scale=scale,
                    zero_point=zero_point,
                    symmetric=symmetric,
                    clip_quantile=float("nan"),
                    clip_min=float("nan"),
                    clip_max=float("nan"),
                    skewness=float("nan"),
                    excess_kurtosis=float("nan"),
                    local_mse=float("nan"),
                    output_distortion=float("nan"),
                    payload_bits=payload_bits,
                    metadata_bits=component_metadata_bits,
                    storage_bits=payload_bits + component_metadata_bits,
                    objective=float("nan"),
                    clip_ratio=float("nan"),
                    model_bits=math.prod(shape) * bit_width,
                )
            if offset != payload.size:
                raise ValueError("packed model contains unexpected trailing payload bytes")
            compressed_bits = sum(s.storage_bits for s in quantizers.values()) + 6 * 32
            if "format_version" in loaded.files:
                compressed_bits += 8
            compression_ratio = (
                np.prod(original_shape) * ORIGINAL_BIT_WIDTH / max(compressed_bits, 1)
            )
            metadata = {
                "format_version": format_version,
                "ranks": ranks,
                "original_shape": original_shape,
                "compressed_bits": int(compressed_bits),
                "compression_ratio": float(compression_ratio),
            }
        else:
            # Compatibility with the first revised package, which saved four
            # packed arrays and diagnostic JSON in the same NPZ file.
            shapes = json.loads(str(loaded["component_shapes"]))
            quantizer_data = json.loads(str(loaded["quantizers"]))
            quantizers = {
                name: QuantizerState(**metadata)
                for name, metadata in quantizer_data.items()
            }
            arrays = {}
            for name in COMPONENT_NAMES:
                state = quantizers[name]
                state.model_bits = math.prod(shapes[name]) * state.bit_width
                state.clip_ratio = 1.0 - state.clip_quantile
                codes = unpack_integer_codes(
                    loaded[f"packed_{name}"],
                    shapes[name],
                    state.qmin,
                    state.bit_width,
                )
                arrays[name] = (
                    codes.astype(np.float64) - state.zero_point
                ) * state.scale
            metadata = {
                "format_version": 1,
                "ranks": tuple(int(value) for value in loaded["ranks"]),
                "original_shape": tuple(int(value) for value in loaded["original_shape"]),
                "compressed_bits": int(loaded["compressed_bits"]),
                "compression_ratio": float(loaded["compression_ratio"]),
            }
    model_bits = model_cost_bits(quantizers)
    raw_bits = math.prod(metadata["original_shape"]) * ORIGINAL_BIT_WIDTH
    file_bytes = os.path.getsize(path)
    metadata.update({
        "model_bits": model_bits,
        "model_compression_ratio": float(raw_bits / model_bits),
        "actual_file_bytes": file_bytes,
        "actual_file_bits": file_bytes * 8,
        "actual_compression_ratio": float(raw_bits / (file_bytes * 8)),
        "training_hyperparameters_present": metadata["format_version"] == 1,
        "original_bit_width": ORIGINAL_BIT_WIDTH,
    })
    return (
        arrays["A"],
        arrays["B"],
        arrays["C"],
    ), arrays["G"], quantizers, metadata

def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def robust_low_sparse_split(
    data: np.ndarray,
    ranks: Sequence[int],
    lambda_value: Optional[float] = None,
    max_iter: int = 50,
    tol: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    """Simple low-rank plus sparse split for controlled anomaly experiments."""
    data = np.asarray(data, dtype=np.float64)
    lam = (
        float(lambda_value)
        if lambda_value is not None
        else 1.0 / math.sqrt(max(data.shape))
    )
    sparse = np.zeros_like(data)
    history = []
    low = np.zeros_like(data)
    for iteration in range(1, max_iter + 1):
        factors, core = hosvd_initialize(data - sparse, ranks)
        low = tucker_reconstruct(core, factors)
        residual = data - low
        threshold = lam * (np.std(residual) + EPS)
        sparse_new = soft_threshold(residual, threshold)
        change = relative_error(sparse_new, sparse)
        sparse = sparse_new
        rel = relative_error(data, low + sparse)
        history.append(
            {
                "iteration": float(iteration),
                "relative_residual": rel,
                "relative_sparse_change": change,
                "sparse_density": float(np.mean(np.abs(sparse) > EPS)),
            }
        )
        if max(rel, change) < tol:
            break
    return low, sparse, history


def sparse_coo_storage_bits(
    sparse: np.ndarray, value_bit_width: int = 32, zero_tolerance: float = EPS
) -> int:
    """Theoretical COO storage including values, indices, and shape metadata."""
    sparse = np.asarray(sparse)
    nnz = int(np.sum(np.abs(sparse) > zero_tolerance))
    index_bits = sum(max(1, int(math.ceil(math.log2(max(size, 2))))) for size in sparse.shape)
    shape_metadata = sparse.ndim * 32 + 32  # shape plus nnz
    return int(nnz * (value_bit_width + index_bits) + shape_metadata)


def anomaly_metrics(
    raw: np.ndarray,
    reconstructed: np.ndarray,
    anomaly_mask: Optional[np.ndarray] = None,
    top_fraction: float = 0.01,
    baseline: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Measure anomaly preservation rather than anomaly residual detectability.

    When a clean baseline is available, injected anomaly amplitude is defined as
    ``raw - baseline`` and the reconstructed anomaly score as
    ``|reconstructed - baseline|``.  This avoids the old contradiction in which
    a perfectly preserved anomaly produced zero reconstruction error and was
    therefore counted as undetected.
    """
    raw = np.asarray(raw, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    if anomaly_mask is None:
        threshold = np.quantile(np.abs(raw), 1.0 - top_fraction)
        anomaly_mask = np.abs(raw) >= threshold
    anomaly_mask = np.asarray(anomaly_mask, dtype=bool)
    normal_mask = ~anomaly_mask

    overall_relative = relative_error(raw, reconstructed)
    anomaly_relative = (
        relative_error(raw[anomaly_mask], reconstructed[anomaly_mask])
        if np.any(anomaly_mask)
        else 0.0
    )
    normal_relative = (
        relative_error(raw[normal_mask], reconstructed[normal_mask])
        if np.any(normal_mask)
        else 0.0
    )

    if baseline is not None:
        baseline = np.asarray(baseline, dtype=np.float64)
        true_amplitude = raw - baseline
        reconstructed_amplitude = reconstructed - baseline
        scores = np.abs(reconstructed_amplitude)
        denominator = float(np.mean(np.abs(true_amplitude[anomaly_mask]))) + EPS
        amplitude_retention = float(
            np.mean(np.abs(reconstructed_amplitude[anomaly_mask])) / denominator
        ) if np.any(anomaly_mask) else 0.0
        amplitude_error = float(
            np.mean(
                np.abs(
                    reconstructed_amplitude[anomaly_mask]
                    - true_amplitude[anomaly_mask]
                )
            )
            / denominator
        ) if np.any(anomaly_mask) else 0.0
    else:
        scores = np.abs(reconstructed)
        denominator = float(np.mean(np.abs(raw[anomaly_mask]))) + EPS
        amplitude_retention = float(
            np.mean(np.abs(reconstructed[anomaly_mask])) / denominator
        ) if np.any(anomaly_mask) else 0.0
        amplitude_error = float(
            np.mean(np.abs(reconstructed[anomaly_mask] - raw[anomaly_mask]))
            / denominator
        ) if np.any(anomaly_mask) else 0.0

    anomaly_count = int(np.sum(anomaly_mask))
    predicted = np.zeros_like(anomaly_mask, dtype=bool)
    if anomaly_count > 0:
        flat_scores = scores.reshape(-1)
        top_indices = np.argpartition(flat_scores, -anomaly_count)[-anomaly_count:]
        predicted.reshape(-1)[top_indices] = True
    tp = float(np.sum(predicted & anomaly_mask))
    fp = float(np.sum(predicted & ~anomaly_mask))
    fn = float(np.sum(~predicted & anomaly_mask))
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2.0 * precision * recall / (precision + recall + EPS)
    iou = tp / (tp + fp + fn + EPS)

    return {
        "overall_relative_error": overall_relative,
        "normal_relative_error": normal_relative,
        "anomaly_relative_error": anomaly_relative,
        "anomaly_precision": float(precision),
        "anomaly_recall": float(recall),
        "anomaly_f1": float(f1),
        "anomaly_iou": float(iou),
        "anomaly_amplitude_retention": amplitude_retention,
        "anomaly_amplitude_error": amplitude_error,
    }


def component_distribution_rows(
    result: JointCompressionResult,
) -> List[Dict[str, float]]:
    rows = []
    master_arrays = list(result.master_factors) + [result.master_core]
    deployment_arrays = list(result.factors) + [result.core]
    for name, master, deployment in zip(
        COMPONENT_NAMES, master_arrays, deployment_arrays
    ):
        skew, kurtosis = _shape_stats(master)
        quantizer = result.quantizers[name]
        rows.append(
            {
                "component": name,
                "values": int(master.size),
                "master_mean": float(np.mean(master)),
                "master_std": float(np.std(master)),
                "deployment_mean": float(np.mean(deployment)),
                "deployment_std": float(np.std(deployment)),
                "skewness": skew,
                "excess_kurtosis": kurtosis,
                "bit_width": quantizer.bit_width,
                "symmetric": int(quantizer.symmetric),
                "clip_quantile": quantizer.clip_quantile,
                "clip_ratio": quantizer.clip_ratio,
                "local_quantization_mse": quantizer.local_mse,
                "output_distortion": quantizer.output_distortion,
                "storage_bits": quantizer.storage_bits,
                "model_bits": quantizer.model_bits,
            }
        )
    return rows


def write_csv(path: str, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_matrix_sequence(directory: str, prefix: str, count: int) -> np.ndarray:
    if count < 1:
        raise ValueError("count must be positive")
    matrices = []
    for index in range(1, count + 1):
        path = os.path.join(directory, f"{prefix}{index}")
        matrices.append(np.loadtxt(path, dtype=np.float64, ndmin=2))
    first_shape = matrices[0].shape
    if any(matrix.shape != first_shape for matrix in matrices):
        raise ValueError("all matrices in the sequence must have the same shape")
    return np.stack(matrices, axis=0)


def load_mat_tensor(path: str, key: Optional[str] = None) -> np.ndarray:
    from scipy.io import loadmat

    loaded = loadmat(path)
    if key is None:
        keys = [name for name in loaded.keys() if not name.startswith("__")]
        if len(keys) != 1:
            raise ValueError(f"please specify --mat-key; candidates: {keys}")
        key = keys[0]
    if np.iscomplexobj(loaded[key]):
        raise ValueError(f"variable {key!r} must contain real-valued data")
    tensor = np.asarray(loaded[key], dtype=np.float64)
    if tensor.ndim != 3:
        raise ValueError(f"variable {key!r} is not a 3D tensor: shape={tensor.shape}")
    return tensor


def save_history(path: str, result: JointCompressionResult) -> None:
    rows = []
    for row in result.history:
        output = dict(row)
        output["final_compression_ratio"] = result.compression_ratio
        output["rank_A"] = result.ranks[0]
        output["rank_B"] = result.ranks[1]
        output["rank_C"] = result.ranks[2]
        rows.append(output)
    write_csv(path, rows)
