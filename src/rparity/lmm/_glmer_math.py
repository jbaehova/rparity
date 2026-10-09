"""Numerical primitives for Laplace GLMMs, following Bates' PIRLS formulation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, special

Array = NDArray[np.float64]


def inverse_link(eta: Array, link: str) -> tuple[Array, Array]:
    """Return the inverse link and its derivative."""
    if link == "log":
        mu = np.exp(np.clip(eta, -700, 40))
        return mu, mu
    if link == "logit":
        mu = np.asarray(special.expit(eta), dtype=float)
        return mu, mu * (1 - mu)
    if link == "probit":
        return np.asarray(special.ndtr(eta), dtype=float), np.exp(-(eta**2) / 2) / np.sqrt(
            2 * np.pi
        )
    if link == "cloglog":
        e = np.exp(np.clip(eta, -700, 40))
        return -np.expm1(-e), np.exp(np.clip(eta - e, -745, 40))
    raise ValueError(f"Unsupported link: {link}")


def conditional_terms(
    eta: Array, y: Array, trials: Array, weights: Array, family: str, link: str
) -> tuple[float, Array, Array]:
    """Conditional negative log likelihood, score, and PIRLS information."""
    mu, derivative = inverse_link(eta, link)
    if family == "poisson":
        nll = np.sum(weights * (mu - y * eta + special.gammaln(y + 1)))
        return float(nll), weights * (mu - y), weights * mu
    p = np.clip(mu, 1e-15, 1 - 1e-15)
    k = y * trials
    if link == "logit":
        logs = k * (-np.logaddexp(0, -eta)) + (trials - k) * (-np.logaddexp(0, eta))
    elif link == "probit":
        logs = k * special.log_ndtr(eta) + (trials - k) * special.log_ndtr(-eta)
    else:
        e = np.exp(np.clip(eta, -700, 40))
        logs = k * np.log(np.maximum(-np.expm1(-e), 1e-300)) - (trials - k) * e
    constant = special.gammaln(trials + 1) - special.gammaln(k + 1)
    constant -= special.gammaln(trials - k + 1)
    nll = -np.sum(weights * (constant + logs))
    if link == "cloglog":
        # Avoid subtracting a probability rounded to one. The score and weights
        # admit formulas in exp(eta) and exp(-exp(eta)) directly.
        e = np.exp(np.clip(eta, -700, 40))
        ratio_success = e * np.exp(-e) / np.maximum(-np.expm1(-e), 1e-300)
        score = weights * ((trials - k) * e - k * ratio_success)
        information = weights * trials * e * ratio_success
        return float(nll), score, np.maximum(information, 1e-15)
    ratio = derivative / (p * (1 - p))
    grad = weights * trials * (p - y) * ratio
    information = np.maximum(weights * trials * derivative * ratio, 1e-15)
    return float(nll), grad, information


def observed_information(
    eta: Array, y: Array, trials: Array, weights: Array, family: str, link: str
) -> Array:
    """Observed information for Newton updates, Ly et al. appendix 6.1."""
    if link in {"log", "logit"}:
        return conditional_terms(eta, y, trials, weights, family, link)[2]
    if link == "cloglog":
        e = np.exp(np.clip(eta, -700, 40))
        p = np.maximum(-np.expm1(-e), 1e-300)
        ratio = e * np.exp(-e) / p
        value = weights * trials * ((1 - y) * e - y * ratio * (1 - e / p))
        return np.maximum(value, 1e-15)
    p, derivative = inverse_link(eta, link)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    variance = p * (1 - p)
    second_derivative = -eta * derivative
    ratio = derivative / variance
    ratio_derivative = second_derivative / variance - derivative**2 * (1 - 2 * p) / variance**2
    return np.maximum(weights * trials * (derivative * ratio + (p - y) * ratio_derivative), 1e-15)


def information_derivative(
    eta: Array, trials: Array, weights: Array, family: str, link: str
) -> Array:
    """Differentiate the PIRLS information with respect to the predictor."""
    mu, derivative = inverse_link(eta, link)
    if family == "poisson":
        return weights * mu
    if link == "cloglog":
        e = np.exp(np.clip(eta, -700, 40))
        ratio = e * np.exp(-e) / np.maximum(-np.expm1(-e), 1e-300)
        information = weights * trials * e * ratio
        return information * (2 - e - ratio)
    p = np.clip(mu, 1e-15, 1 - 1e-15)
    variance = p * (1 - p)
    information = weights * trials * derivative**2 / variance
    if link == "logit":
        return information * (1 - 2 * p)
    return information * (-2 * eta - derivative * (1 - 2 * p) / variance)


def conditional_mode(
    base: Array,
    A: Array,
    y: Array,
    trials: Array,
    weights: Array,
    family: str,
    link: str,
) -> tuple[float, Array, Array, Array]:
    """Solve the penalized likelihood with line-searched Newton scoring.

    Ly et al. (2026), sections 2.5 and 2.6, gives the
    standardized normal random-effect representation and Laplace determinant.
    """
    q = A.shape[1]
    u = np.zeros(q)
    identity = np.eye(q)
    converged = False
    for _ in range(100):
        eta = base + A @ u
        nll, score, info = conditional_terms(eta, y, trials, weights, family, link)
        criterion = nll + u @ u / 2
        gradient = A.T @ score + u
        curvature = (
            info
            if link in {"log", "logit"}
            else observed_information(eta, y, trials, weights, family, link)
        )
        hessian = (A.T * curvature) @ A + identity
        if not np.all(np.isfinite(hessian)) or not np.all(np.isfinite(gradient)):
            return float("inf"), u, eta, hessian
        try:
            factor = linalg.cho_factor(hessian, lower=True, check_finite=False)
            step = linalg.cho_solve(factor, gradient, check_finite=False)
        except linalg.LinAlgError:
            # Extreme outer proposals can lose the identity penalty to rounding.
            # Reject this likelihood evaluation, allowing the optimizer to recover.
            return float("inf"), u, eta, hessian
        if np.max(np.abs(gradient), initial=0) < 1e-9:
            converged = True
            break
        scale = 1.0
        while scale > 2**-25:
            candidate = u - scale * step
            trial_value = (
                conditional_terms(base + A @ candidate, y, trials, weights, family, link)[0]
                + float(candidate @ candidate) / 2
            )
            if trial_value <= criterion + 1e-12:
                break
            scale /= 2
        u = candidate
        if np.max(np.abs(scale * step), initial=0) < 1e-10:
            converged = True
            break
    eta = base + A @ u
    nll, _, info = conditional_terms(eta, y, trials, weights, family, link)
    hessian = (A.T * info) @ A + identity
    try:
        chol = linalg.cholesky(hessian, lower=True, check_finite=False)
    except linalg.LinAlgError:
        return float("inf"), u, eta, hessian
    objective_value = nll + float(u @ u) / 2 + float(np.log(np.diag(chol)).sum())
    if not converged or not np.isfinite(objective_value):
        # Failed inner iterations must never masquerade as a likelihood optimum.
        return float("inf"), u, eta, hessian
    return objective_value, u, eta, hessian


def numerical_hessian(function: Callable[[Array], float], x: Array) -> Array:
    """Centered finite differences with scale-adjusted steps."""
    steps = 1e-4 * np.maximum(1, np.abs(x))
    hessian = np.empty((len(x), len(x)))
    center = function(x)
    for i in range(len(x)):
        ei = np.zeros(len(x))
        ei[i] = steps[i]
        hessian[i, i] = (function(x + ei) - 2 * center + function(x - ei)) / steps[i] ** 2
        for j in range(i):
            ej = np.zeros(len(x))
            ej[j] = steps[j]
            value = function(x + ei + ej) - function(x + ei - ej)
            value -= function(x - ei + ej) - function(x - ei - ej)
            hessian[i, j] = hessian[j, i] = value / (4 * steps[i] * steps[j])
    return hessian


def score_hessian(gradient: Callable[[Array], Array], x: Array) -> Array:
    """Differentiate a score with Richardson cancellation of quadratic error.

    A score derivative avoids subtracting nearly equal likelihood values.
    Extrapolation permits a larger step, suppressing inner-mode roundoff
    without retaining the corresponding second-order truncation error.
    """
    steps = 1e-3 * np.maximum(1, np.abs(x))
    hessian = np.empty((len(x), len(x)))
    for i, step in enumerate(steps):
        direction = np.zeros(len(x))
        direction[i] = step
        coarse = (gradient(x + direction) - gradient(x - direction)) / (2 * step)
        direction[i] /= 2
        fine = (gradient(x + direction) - gradient(x - direction)) / step
        hessian[:, i] = (4 * fine - coarse) / 3
    return (hessian + hessian.T) / 2
