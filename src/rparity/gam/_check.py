"""Numerical GAM checks and reproducible residual randomization diagnostics.

The neighbor difference variance diagnostic follows Wood (2017), section 5.9.
The public ``mgcv::k.check`` API describes the test and its permutation null.
The random stream and geometric conventions were validated by public numeric
black-box queries through the repository's R oracle. No R source is used here.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ._formula import evaluate_expression


class _RRandom:
    """MT19937 with the R seed convention and unbiased integer rejection.

    This uses NumPy's standard MT19937 implementation. Only initialization and
    integer selection differ from NumPy's Generator API. The 32-bit linear
    congruential initialization and 16-bit rejection stream are checked against
    the public ``set.seed``, ``runif`` and ``sample.int`` oracle observations.
    """

    def __init__(self, seed: int | None) -> None:
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
        if not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer or None")
        value = int(seed) & 0xFFFFFFFF
        for _ in range(50):
            value = (69069 * value + 1) & 0xFFFFFFFF
        state = []
        for _ in range(625):
            value = (69069 * value + 1) & 0xFFFFFFFF
            state.append(value)
        self._bitgen = np.random.MT19937()
        self._bitgen.state = {
            "bit_generator": "MT19937",
            "state": {"key": np.asarray(state[1:], dtype=np.uint32), "pos": 624},
        }
        self._buffer: np.ndarray = np.empty(0, dtype=np.uint64)
        self._position = 0

    def _raw(self) -> int:
        if self._position == len(self._buffer):
            self._buffer = self._bitgen.random_raw(4096)
            self._position = 0
        value = int(self._buffer[self._position])
        self._position += 1
        return value

    def uniform(self, size: int) -> np.ndarray:
        """Standard 32-bit MT uniforms, excluding the two endpoints."""
        values = np.fromiter((self._raw() for _ in range(size)), dtype=np.uint64)
        result = values.astype(float) / 2**32
        return np.clip(result, 0.5 / 2**32, 1 - 0.5 / 2**32)

    def sample(self, population: int, size: int | None = None) -> np.ndarray:
        """A zero-based sample without replacement with R's modern sampler."""
        size = population if size is None else size
        if population < 0 or size < 0 or size > population:
            raise ValueError("Invalid sample size")
        available = list(range(population))
        result = np.empty(size, dtype=np.int64)
        for i in range(size):
            remaining = population - i
            bits = (remaining - 1).bit_length()
            mask = (1 << bits) - 1
            while True:
                value = 0
                # Even the final singleton consumes a uniform in sample.int.
                for _ in range(bits // 16 + 1):
                    value = (value << 16) | (self._raw() >> 16)
                choice = value & mask
                if choice < remaining:
                    break
            result[i] = available[choice]
            available[choice] = available[remaining - 1]
        return result


def _coordinates(smooth: Any, data: pd.DataFrame) -> np.ndarray | None:
    columns = []
    for variable in smooth.variables:
        values = evaluate_expression(variable, data)
        if not pd.api.types.is_numeric_dtype(values):
            return None
        columns.append(np.asarray(values, dtype=float))
    return np.column_stack(columns) if columns else None


def _neighbor_indices(
    design: Any, smooth: Any, data: pd.DataFrame, coordinates: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Three nearest neighbors in the fitted smooth's local variation metric."""
    ranges = np.ptp(coordinates, axis=0)
    indices = np.asarray(smooth.indices, dtype=int)
    baseline = np.asarray(design.predict(data), dtype=float)[:, indices] @ coefficients[indices]
    scaling = np.zeros(coordinates.shape[1])
    for j, variable in enumerate(smooth.variables):
        if ranges[j] == 0:
            continue
        shifted = data.copy()
        if variable not in shifted:
            raise ValueError("Multivariate k checking requires named numeric covariates")
        shifted[variable] = np.asarray(shifted[variable], dtype=float) + ranges[j] / 1000
        fitted = np.asarray(design.predict(shifted), dtype=float)[:, indices] @ coefficients[indices]
        scaling[j] = np.mean(np.abs(fitted - baseline)) / ranges[j]**2
    # A flat smooth carries no preferred local variation direction. Use the
    # covariate ranges rather than an arbitrary zero-dimensional distance tie.
    if not np.any(scaling > 0):
        scaling = np.divide(1, ranges, out=np.zeros_like(ranges), where=ranges > 0)
    locations = coordinates * scaling
    count = min(3, len(data) - 1)
    raw = cKDTree(locations).query(locations, k=count + 1)[1]
    # Remove each row's own index explicitly. This also handles duplicated
    # coordinates, for which the tree may put another zero-distance row first.
    return np.asarray([
        [int(j) for j in row if j != i][:count] for i, row in enumerate(raw)
    ], dtype=int)


def k_check(
    design: Any, data: pd.DataFrame, residuals: np.ndarray, edf: np.ndarray, *,
    coefficients: np.ndarray | None = None, seed: int | None = 1,
    n_rep: int = 400, subsample: int = 5000,
) -> pd.DataFrame:
    """Basis dimension, EDF, neighbor variance ratio and permutation p-value.

    Univariate diagnostics use adjacent observations in covariate order.
    Multivariate diagnostics use three neighbors after scaling by the fitted
    smooth's finite differences. A factor covariate has an unavailable test.
    Factor-by smooths still test their numeric covariates using all residuals.
    A supplied seed reproduces R's default Mersenne-Twister/Rejection stream;
    it does not modify NumPy's global random state.
    """
    if not isinstance(n_rep, (int, np.integer)) or n_rep < 1:
        raise ValueError("n_rep must be a positive integer")
    if not isinstance(subsample, (int, np.integer)) or subsample < 2:
        raise ValueError("subsample must be an integer of at least two")
    residuals = np.asarray(residuals, dtype=float).reshape(-1)
    if len(data) != len(residuals) or len(data) < 2:
        raise ValueError("k checking needs matching data and at least two residuals")
    if not np.all(np.isfinite(residuals)):
        raise ValueError("k checking needs finite residuals")
    rng = _RRandom(seed)
    if len(data) > subsample:
        sample = rng.sample(len(data), subsample)
        data = data.iloc[sample].reset_index(drop=True)
        residuals = residuals[sample]
    variance = float(np.mean(residuals**2))
    rows = []
    labels = []
    for smooth in design.smooths:
        coordinates = _coordinates(smooth, data)
        ratio, pvalue = np.nan, np.nan
        if coordinates is not None and variance > 0:
            if coordinates.shape[1] == 1:
                order = np.argsort(coordinates[:, 0], kind="stable")
                local = float(np.mean(np.diff(residuals[order])**2) / 2)

                def randomized(values: np.ndarray, neighbor_rows: np.ndarray | None = None) -> float:
                    return float(np.mean(np.diff(values)**2) / 2)

            else:
                if coefficients is None:
                    raise ValueError("Multivariate k checking requires fitted coefficients")
                neighbors = _neighbor_indices(design, smooth, data, coordinates, coefficients)

                def randomized(values: np.ndarray, neighbor_rows: np.ndarray | None = neighbors) -> float:
                    assert neighbor_rows is not None
                    return float(np.mean((values[:, None] - values[neighbor_rows])**2) / 2)

                local = randomized(residuals)
            ratio = local / variance
            count = sum(randomized(residuals[rng.sample(len(data))]) < local for _ in range(n_rep))
            pvalue = count / n_rep
        rows.append([float(len(smooth.indices)), float(np.sum(edf[smooth.indices])), ratio, pvalue])
        labels.append(smooth.label)
    return pd.DataFrame(rows, index=labels, columns=["k'", "edf", "k-index", "p-value"])


def gam_check(
    model: Any, *, k_rep: int = 200, k_sample: int = 5000, seed: int | None = 0,
) -> dict[str, Any]:
    """Return convergence and residual diagnostics without creating plots."""
    residuals = model.residuals(type="deviance")
    table = k_check(model.design, model.data, residuals.to_numpy(), model.edf,
                    coefficients=model.beta, n_rep=k_rep, subsample=k_sample, seed=seed)
    outer = model._outer_result
    gradient = getattr(model, "_outer_gradient", getattr(outer, "jac", None))
    hessian = getattr(model, "_outer_hessian", getattr(outer, "hess", None))
    gradient = np.asarray(gradient, dtype=float) if gradient is not None else np.empty(0)
    hessian = np.asarray(hessian, dtype=float) if hessian is not None else np.empty((0, 0))
    eigenvalues = np.linalg.eigvalsh((hessian + hessian.T) / 2) if hessian.size else np.empty(0)
    # Random-effect dummy columns overlap the intercept in X, but their
    # positive penalty identifies that overlap. Report the penalized rank.
    rank_blocks = [model.X / max(float(np.linalg.norm(model.X, 2)), 1.0)]
    for penalty in getattr(model.design, "S", []):
        values, vectors = np.linalg.eigh((penalty + penalty.T) / 2)
        positive = values > max(float(np.max(values)), 1.0) * 1e-12
        root = np.sqrt(values[positive])[:, None] * vectors[:, positive].T
        if root.size:
            rank_blocks.append(root / max(float(np.linalg.norm(root, 2)), 1.0))
    penalized_rank = int(np.linalg.matrix_rank(np.vstack(rank_blocks), tol=1e-10))
    return {
        "converged": model.converged,
        "iterations": model.iterations,
        "rank": penalized_rank,
        "model_rank": len(model.beta),
        "coefficients": len(model.beta),
        "gradient": gradient,
        "hessian": hessian,
        "hessian_eigenvalues": eigenvalues,
        "hessian_positive_definite": bool(np.all(eigenvalues > 0)) if eigenvalues.size else None,
        "scale": model.scale,
        "score": model.objective,
        "deviance": model.deviance(),
        "residual_df": model.df_resid,
        "k.check": table,
        "residuals": residuals,
        "fitted": model.fitted(),
    }
