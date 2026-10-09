"""GAM component inference from published quadratic-form tests.

The smooth test is the fractional-rank Wald construction in Wood (2013),
Biometrika 100, 221--228, with the IRLS-weighted function space described in
the public ``summary.gam`` documentation. Fully penalized terms instead use
Wood (2013), Biometrika 100, 1005--1010. All calculations use the fitted
Python model; no target-package implementation or R runtime is used.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import integrate, linalg, special, stats

Array = NDArray[np.float64]


@lru_cache(maxsize=1)
def _quadrature() -> tuple[Array, Array]:
    nodes, weights = np.polynomial.legendre.leggauss(100)
    return (nodes + 1) / 2, weights / 2


@lru_cache(maxsize=64)
def _beta_quadrature(base: int) -> tuple[Array, Array]:
    # Jacobi weights integrate the Beta density directly. Transforming a
    # uniform quadrature to Beta introduces an avoidable endpoint singularity.
    nodes, weights = special.roots_jacobi(80, 0, base / 2 - 1)
    return (nodes + 1) / 2, weights / np.sum(weights)


def _fractional_sf(statistic: float, rank: float, residual_df: float | None) -> float:
    """Tail of the fractional-rank quadratic form, optionally divided by scale.

    The two non-unit eigenvalues act on independent one-degree chi squares.
    Their squared-normal directions have a Beta(1/2, 1/2) distribution. The
    unit-eigenvalue directions have an independent Beta((floor(rank)-1)/2, 1)
    share of the total radius. This gives bounded quadrature rather than a
    slowly converging characteristic-function integral near integer rank.
    """
    if statistic <= 0:
        return 1.0
    nearest = round(rank)
    if abs(rank - nearest) < 1e-9:
        return float(stats.chi2.sf(statistic, nearest) if residual_df is None
                     else stats.f.sf(statistic / nearest, nearest, residual_df))
    integer = int(np.floor(rank))
    fraction = rank - integer
    if integer < 1:
        # A nonzero-dimensional null space normally gives rank >= 1. This
        # continuous extension covers numerically deficient function spaces.
        return float(stats.chi2.sf(statistic, max(rank, 1e-12)) if residual_df is None
                     else stats.f.sf(statistic / max(rank, 1e-12), max(rank, 1e-12), residual_df))
    upper = (1 + fraction + np.sqrt(1 - fraction**2)) / 2
    lower = 1 + fraction - upper
    t, weights = _quadrature()
    share = lower + (upper - lower) * np.sin(np.pi * t / 2)**2
    base = integer - 1
    total_df = base + 2
    if base:
        unit_share, beta_weight = _beta_quadrature(base)
        share = unit_share[:, None] + (1 - unit_share[:, None]) * share[None, :]
        weight = beta_weight[:, None] * weights[None, :]
    else:
        weight = weights
    tail = (stats.chi2.sf(statistic / share, total_df) if residual_df is None
            else stats.f.sf(statistic / (total_df * share), total_df, residual_df))
    return float(np.sum(weight * tail))


def _weighted_chisq_sf(
    statistic: float, eigenvalues: Array, residual_df: float | None = None,
) -> float:
    """Invert the characteristic function of a central quadratic form.

    This is the Imhof/Gil-Pelaez identity. An estimated scale is represented
    by an additional negative chi-square weight, so its tail requires no
    nested integration. Positive equal weights recover chi-square/F tails.
    """
    values = np.asarray(eigenvalues, dtype=float)
    values = values[values > max(float(np.max(values, initial=0)), 1e-300) * 1e-10]
    if not len(values):
        return float(statistic <= 0)
    if statistic <= 0:
        return 1.0
    maximum = float(np.max(values))
    values = values / maximum
    statistic /= maximum
    if np.ptp(values) < 1e-10:
        return float(stats.chi2.sf(statistic / values[0], len(values)) if residual_df is None
                     else stats.f.sf(statistic / (values[0] * len(values)), len(values), residual_df))
    degrees = np.ones(len(values))
    threshold = statistic
    if residual_df is not None:
        values = np.r_[values, -statistic / residual_df]
        degrees = np.r_[degrees, residual_df]
        threshold = 0.0

    def phase_radius(t: float) -> tuple[float, float]:
        twice = 2 * t * values
        phase = float(np.sum(degrees * np.arctan(twice)) / 2)
        reciprocal = float(np.exp(-np.sum(degrees * np.log1p(twice**2)) / 4))
        return phase, reciprocal

    def integrand(t: float) -> float:
        if t == 0:
            return float(np.sum(degrees * values) - threshold)
        phase, reciprocal = phase_radius(t)
        return float(np.sin(phase - t * threshold) * reciprocal / t)

    if threshold == 0:
        integral = integrate.quad(integrand, 0, np.inf, epsabs=2e-8, limit=250)[0]
    else:
        # Fourier quadrature handles the otherwise oscillatory infinite tail.
        integral = integrate.quad(integrand, 0, 1, epsabs=2e-8, limit=150)[0]

        def cosine_amplitude(t: float) -> float:
            phase, reciprocal = phase_radius(t)
            return float(np.sin(phase) * reciprocal / t)

        def sine_amplitude(t: float) -> float:
            phase, reciprocal = phase_radius(t)
            return float(np.cos(phase) * reciprocal / t)

        integral += integrate.quad(cosine_amplitude, 1, np.inf, weight="cos", wvar=threshold,
                                   epsabs=2e-8, limlst=150)[0]
        integral -= integrate.quad(sine_amplitude, 1, np.inf, weight="sin", wvar=threshold,
                                   epsabs=2e-8, limlst=150)[0]
    return float(np.clip(0.5 + integral / np.pi, 0, 1))


def _working_weights(model: Any) -> Array:
    from ._fit import _Family

    family = _Family(model.family, model.link)
    derivative = family.derivatives(model.linear_predictor, model.fitted_values)[0]
    return np.asarray(model.weights * derivative**2 / family.variance(model.fitted_values)[0], dtype=float)


def _smooth_wald(
    coefficients: Array, covariance: Array, design: Array, rank: float,
    residual_df: float | None,
) -> tuple[float, float, float]:
    _, triangular = linalg.qr(design, mode="economic")
    function_covariance = triangular @ covariance @ triangular.T
    values, vectors = linalg.eigh(function_covariance)
    values, vectors = values[::-1], vectors[:, ::-1]
    positive = values > max(float(values[0]), 1e-300) * np.finfo(float).eps * len(values)
    values, vectors = values[positive], vectors[:, positive]
    rank = float(np.clip(rank, 1e-12, len(values)))
    # A fixed sign convention makes the displayed branch reproducible.
    vectors *= np.where(vectors[0] < 0, -1.0, 1.0)
    standardized = vectors.T @ triangular @ coefficients / np.sqrt(values)
    integer = min(int(np.floor(rank)), len(values))
    fraction = rank - integer
    statistic = float(np.sum(standardized[:integer]**2))
    alternate = statistic
    if fraction > 1e-9 and integer < len(values):
        statistic += fraction * standardized[integer]**2
        cross = (2 * np.sqrt(fraction * (1 - fraction) / 2)
                 * standardized[integer - 1] * standardized[integer]) if integer else 0.0
        alternate = float(statistic - cross)
        statistic = float(statistic + cross)
    # Averaging the two signs removes arbitrary eigenvector orientation from
    # inference. The displayed statistic uses the positive-first-entry branch.
    pvalue = (_fractional_sf(statistic, rank, residual_df)
              + _fractional_sf(alternate, rank, residual_df)) / 2
    return statistic, rank, pvalue


def _random_test(model: Any, indices: Array) -> tuple[float, float, float]:
    covariance = model.cov_beta[np.ix_(indices, indices)]
    coefficients = model.beta[indices]
    values, vectors = linalg.eigh(covariance / model.scale)
    positive = values > max(float(np.max(values)), 1e-300) * 1e-12
    values, vectors = values[positive], vectors[:, positive]
    inverse_root = vectors / np.sqrt(values)
    whole_covariance = model.cov_beta / model.scale
    inner = getattr(model, "_inner", None)
    information = (inner.information if inner is not None else
                   model.X.T @ (_working_weights(model)[:, None] * model.X))
    # Ordinary smooths are conditioned on for this variance-component test.
    # Other fully penalized random components retain their fitted prior
    # variance in the null distribution, as in Wood's mixed-model argument.
    null_covariance = whole_covariance[indices] @ information @ whole_covariance[:, indices]
    for other in model.design.smooths:
        other_indices = np.asarray(other.indices, dtype=int)
        if other.null_space_dim != 0 or np.array_equal(other_indices, indices):
            continue
        cross = whole_covariance[np.ix_(indices, other_indices)]
        null_covariance += cross @ model._penalty[np.ix_(other_indices, other_indices)] @ cross.T
    eigenvalues = np.maximum(linalg.eigvalsh(inverse_root.T @ null_covariance @ inverse_root), 0)
    reference_df = float(np.sum(eigenvalues > max(float(np.max(eigenvalues)), 1e-300) * 1e-8))
    projections = vectors.T @ coefficients
    statistic = float(np.sum(projections**2 / values) / model.scale)
    pvalue = _weighted_chisq_sf(statistic, eigenvalues, None if model._scale_known else model.df_resid)
    return statistic, reference_df, pvalue


def smooth_table(model: Any) -> pd.DataFrame:
    """Return Bayesian smooth tests and boundary-aware fully penalized tests."""
    weighted = model.X * np.sqrt(_working_weights(model))[:, None]
    residual_df = None if model._scale_known else model.df_resid
    rows: list[list[float]] = []
    labels: list[str] = []
    for smooth in model.design.smooths:
        indices = np.asarray(smooth.indices, dtype=int)
        edf = float(np.sum(model.edf[indices]))
        if smooth.null_space_dim == 0:
            statistic, rank, pvalue = _random_test(model, indices)
        else:
            statistic, rank, pvalue = _smooth_wald(
                model.beta[indices], model.cov_beta[np.ix_(indices, indices)],
                weighted[:, indices], float(np.sum(model.edf1[indices])), residual_df,
            )
        displayed = statistic if model._scale_known else statistic / max(rank, 1e-300)
        rows.append([edf, rank, displayed, pvalue])
        labels.append(smooth.label)
    return pd.DataFrame(rows, index=labels,
                        columns=["edf", "Ref.df", "Chi.sq" if model._scale_known else "F", "p-value"])


def summary_statistics(model: Any) -> dict[str, Any]:
    """Numerical metadata accompanying the coefficient and smooth tables."""
    from ._fit import _Family, _irls

    family = _Family(model.family, model.link)
    mean = float(np.average(model.y, weights=model.weights))
    null_fit = _irls(np.ones((model.nobs, 1)), model.y, model.weights, model.offset,
                     family, np.zeros((1, 1)), 1000, 1e-12)
    null_deviance = null_fit.deviance
    weighted_residuals = np.sqrt(model.weights) * (model.y - model.fitted_values)
    weighted_response = np.sqrt(model.weights) * (model.y - mean)
    residual_variance = float(np.var(weighted_residuals)) * model.nobs / model.df_resid
    original_variance = float(np.var(weighted_response)) * model.nobs / max(1, model.nobs - 1)
    penalty_values, penalty_vectors = linalg.eigh(model._penalty)
    penalty_root = np.sqrt(np.maximum(penalty_values, 0))[:, None] * penalty_vectors.T
    penalized_rank = int(np.linalg.matrix_rank(np.vstack((model.X, penalty_root))))
    return {"scale": model.scale,
            "r.sq": 1 - residual_variance / original_variance if original_variance else np.nan,
            "dev.expl": 1 - model.deviance() / null_deviance if null_deviance else np.nan,
            "residual.df": model.df_resid, "sp.criterion": model.objective,
            "method": model.method, "rank": penalized_rank, "n": model.nobs}


def summary_tables(model: Any) -> dict[str, Any]:
    indices = np.asarray(model.design.parametric_indices, dtype=int)
    standard_errors = np.sqrt(np.maximum(np.diag(model.cov_beta), 0))
    statistic = model.beta[indices] / standard_errors[indices]
    known = model._scale_known
    pvalues = 2 * (stats.norm.sf(np.abs(statistic)) if known
                   else stats.t.sf(np.abs(statistic), model.df_resid))
    parametric = pd.DataFrame(
        {"Estimate": model.beta[indices], "Std. Error": standard_errors[indices],
         "z value" if known else "t value": statistic,
         "Pr(>|z|)" if known else "Pr(>|t|)": pvalues},
        index=[model.coef_names[i] for i in indices],
    )
    return {"parametric": parametric, "smooth": smooth_table(model), **summary_statistics(model)}
