"""Small-sample fixed-effect inference for Gaussian mixed models.

Algorithms are derived from Satterthwaite (1946), Fai and Cornelius (1996),
and Kenward and Roger (1997), Sections 2 and 3. No R is used at runtime.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import linalg, stats

Array = NDArray[np.float64]


@dataclass(frozen=True)
class ContrastTest:
    """A scaled Wald F test and its reference degrees of freedom."""

    statistic: float
    num_df: int
    den_df: float
    pvalue: float
    covariance: Array
    scale: float = 1.0


def _differentiate(fun: Callable[[Array], Any], x: Array) -> tuple[list[Array], list[list[Array]]]:
    steps = 1e-4 * np.maximum(np.abs(x), 1.0)
    center = np.asarray(fun(x), dtype=float)
    first = []
    second: list[list[Array]] = []
    for i, hi in enumerate(steps):
        ei = np.zeros_like(x)
        ei[i] = hi
        plus, minus = np.asarray(fun(x + ei)), np.asarray(fun(x - ei))
        first.append((plus - minus) / (2 * hi))
        row = []
        for j, hj in enumerate(steps):
            if i == j:
                row.append((plus - 2 * center + minus) / hi**2)
            else:
                ej = np.zeros_like(x)
                ej[j] = hj
                row.append((fun(x + ei + ej) - fun(x + ei - ej)
                            - fun(x - ei + ej) + fun(x - ei - ej)) / (4 * hi * hj))
        second.append(row)
    return first, second


def _satterthwaite_state(model: Any) -> tuple[Array, list[Array]]:
    cache = getattr(model, "_satterthwaite_cache", None)
    if cache is not None:
        return cache  # type: ignore[no-any-return]
    params = np.asarray(model.variance_params, dtype=float)
    _, second = _differentiate(model.variance_objective, params)
    hessian = np.asarray(second, dtype=float).reshape(len(params), len(params))
    eig, vec = linalg.eigh(hessian)
    keep = eig > max(1e-8, np.max(np.abs(eig)) * 1e-8)
    covariance = 2 * (vec[:, keep] / eig[keep]) @ vec[:, keep].T
    derivatives, _ = _differentiate(model.covariance_at, params)
    cache = covariance, derivatives
    model._satterthwaite_cache = cache
    return cache


def _kr_state(model: Any) -> tuple[Array, Array, list[Array], Array]:
    cache = getattr(model, "_kenward_roger_cache", None)
    if cache is not None:
        return cache  # type: ignore[no-any-return]
    if not model.reml:
        raise ValueError("Kenward-Roger inference requires a REML fit")
    x = np.asarray(model.X, dtype=float)
    v = np.asarray(model.marginal_covariance_at(model.variance_params), dtype=float)
    vi = linalg.solve(v, np.eye(len(v)), assume_a="pos")
    c = np.asarray(model.cov_beta, dtype=float)
    h = vi @ x
    projection = vi - h @ c @ h.T
    if hasattr(model, "covariance_components"):
        derivatives = model.covariance_components()
    else:
        derivatives, _ = _differentiate(model.marginal_covariance_at,
                                        np.asarray(model.variance_params, dtype=float))
    # LMM covariance is linear in residual variance and the entries of each G.
    # A local derivative basis is equivalent at every interior full-rank estimate.
    bases = [np.asarray(g, dtype=float) for g in derivatives]
    projected = [projection @ g for g in bases]
    info = np.array([[0.5 * np.trace(a @ b) for b in projected] for a in projected])
    w = linalg.pinvh(info)
    p = [-h.T @ g @ h for g in bases]
    gh = [g @ h for g in bases]
    correction = np.zeros_like(c)
    for i in range(len(bases)):
        for j in range(len(bases)):
            q = gh[i].T @ vi @ gh[j]
            correction += w[i, j] * (q - p[i] @ c @ p[j])
    adjusted = c + 2 * c @ correction @ c
    adjusted = (adjusted + adjusted.T) / 2
    cache = adjusted, w, p, c
    model._kenward_roger_cache = cache
    return cache


def adjusted_covariance(model: Any) -> Array:
    """Return the Kenward-Roger bias-adjusted fixed-effect covariance.

    Uses the expected REML information and the linear-covariance expression
    in Kenward and Roger (1997), equations (4) and (5).
    """
    return _kr_state(model)[0].copy()


def _kr_moments(model: Any, contrast: Array) -> tuple[float, float]:
    _, w, p, c = _kr_state(model)
    rank = int(np.linalg.matrix_rank(contrast @ c @ contrast.T))
    theta = contrast.T @ linalg.pinvh(contrast @ c @ contrast.T) @ contrast
    matrices = [theta @ c @ pi @ c for pi in p]
    traces = np.array([np.trace(a) for a in matrices])
    a1 = float(traces @ w @ traces)
    a2 = float(sum(w[i, j] * np.trace(a @ b)
                   for i, a in enumerate(matrices) for j, b in enumerate(matrices)))
    if abs(a2) < 1e-12:
        return float("inf"), 1.0
    b = (a1 + 6 * a2) / (2 * rank)
    g = ((rank + 1) * a1 - (rank + 4) * a2) / ((rank + 2) * a2)
    divisor = 3 * rank + 2 * (1 - g)
    c1, c2, c3 = g / divisor, (rank - g) / divisor, (rank + 2 - g) / divisor
    expectation = 1 / (1 - a2 / rank)
    variance = 2 / rank * (1 + c1 * b) / ((1 - c2 * b)**2 * (1 - c3 * b))
    rho = variance / (2 * expectation**2)
    df = 4 + (rank + 2) / (rank * rho - 1)
    scale = df / (expectation * (df - 2))
    return float(df), float(scale)


def contrast_df(model: Any, L: Any, method: str = "satterthwaite") -> float:
    """Approximate denominator df for a scalar or joint fixed-effect contrast.

    Satterthwaite uses the observed variance-parameter Hessian; joint contrasts
    use the Fai-Cornelius eigencomponent combination. KR uses moment matching.
    """
    contrast = np.atleast_2d(np.asarray(L, dtype=float))
    method = method.lower().replace("_", "-")
    if method in {"asymptotic", "z", "none"}:
        return float("inf")
    if method in {"kenward-roger", "kenwardroger", "kr"}:
        return _kr_moments(model, contrast)[0]
    if method != "satterthwaite":
        raise ValueError(f"Unknown degrees-of-freedom method: {method}")
    covariance, derivatives = _satterthwaite_state(model)
    c = np.asarray(model.cov_beta, dtype=float)
    eigenvalues, eigenvectors = linalg.eigh(contrast @ c @ contrast.T)
    keep = eigenvalues > max(1e-12, np.max(np.abs(eigenvalues)) * 1e-10)
    dfs = []
    for value, vector in zip(eigenvalues[keep], eigenvectors[:, keep].T, strict=True):
        row = vector @ contrast
        gradient = np.array([row @ d @ row for d in derivatives])
        variance = float(gradient @ covariance @ gradient)
        dfs.append(2 * value**2 / variance if variance > 1e-20 else np.inf)
    if len(dfs) == 0:
        raise ValueError("Contrast has zero estimable rank")
    if len(dfs) == 1:
        return float(dfs[0])
    if min(dfs) <= 2:
        return 2.0
    if all(np.isinf(dfs)):
        return float("inf")
    expectation = sum(d / (d - 2) if np.isfinite(d) else 1.0 for d in dfs)
    return float(2 * expectation / (expectation - len(dfs)))


def contrast_test(model: Any, L: Any, rhs: Any = 0,
                  method: str = "satterthwaite") -> ContrastTest:
    """Test ``L beta = rhs`` with Satterthwaite, KR or asymptotic inference."""
    contrast = np.atleast_2d(np.asarray(L, dtype=float))
    is_kr = method.lower().replace("_", "-") in {"kr", "kenward-roger", "kenwardroger"}
    covariance = adjusted_covariance(model) if is_kr else np.asarray(model.cov_beta)
    cov_l = contrast @ covariance @ contrast.T
    rank = int(np.linalg.matrix_rank(cov_l))
    if rank == 0:
        raise ValueError("Contrast has zero estimable rank")
    difference = contrast @ np.asarray(model.beta) - np.asarray(rhs)
    df = contrast_df(model, contrast, method)
    scale = _kr_moments(model, contrast)[1] if is_kr else 1.0
    f = float(difference @ linalg.pinvh(cov_l) @ difference / rank) * scale
    pvalue = float(stats.chi2.sf(f * rank, rank) if np.isinf(df) else stats.f.sf(f, rank, df))
    return ContrastTest(f, rank, df, pvalue, covariance, scale)


def coefficient_tests(model: Any, ddf: str = "satterthwaite") -> pd.DataFrame:
    """Return fixed-effect t estimates, SEs, denominator df and two-sided p-values."""
    is_kr = ddf.lower().replace("_", "-") in {"kr", "kenward-roger", "kenwardroger"}
    covariance = adjusted_covariance(model) if is_kr else np.asarray(model.cov_beta)
    se = np.sqrt(np.diag(covariance))
    values = np.asarray(model.beta) / se
    dfs = np.array([contrast_df(model, row, ddf) for row in np.eye(len(values))])
    pvalues = np.array([2 * stats.norm.sf(abs(t)) if np.isinf(df)
                       else 2 * stats.t.sf(abs(t), df)
                       for t, df in zip(values, dfs, strict=True)])
    return pd.DataFrame({"Estimate": model.beta, "Std. Error": se, "df": dfs,
                         "t value": values, "Pr(>|t|)": pvalues}, index=model.coef_names)


__all__ = ["ContrastTest", "adjusted_covariance", "coefficient_tests", "contrast_df", "contrast_test"]
