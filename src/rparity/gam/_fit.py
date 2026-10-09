"""Penalized GAM likelihoods and smoothing selection, implemented from mathematics.

The inner fit uses penalized IRLS. The outer criteria are the GCV/UBRE
definitions and the Laplace marginal likelihood of Wood (2011, JRSS-B 73,
3-36); covariance and effective degrees of freedom follow Wood (2017),
chapters 6 and 7. The implementation has no R runtime dependency.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import linalg, optimize, special

from rparity.formula import as_dataframe

from ._formula import build_design

Array = NDArray[np.float64]


def _vector(value: Any, data: pd.DataFrame, default: float) -> Array:
    if value is None:
        return np.full(len(data), default, dtype=float)
    if isinstance(value, str):
        return np.asarray(data[value], dtype=float).copy()
    result = np.asarray(value, dtype=float)
    if result.ndim == 0:
        return np.full(len(data), float(result), dtype=float)
    if result.shape != (len(data),):
        raise ValueError("weights and offset must have one value per observation")
    return result.copy()


@dataclass(frozen=True)
class _Family:
    name: str
    link: str

    def inverse(self, eta: Array) -> Array:
        if self.link == "identity":
            return eta.copy()
        if self.link == "log":
            return np.exp(np.clip(eta, -700, 700))
        if self.link == "inverse":
            return 1 / eta
        if self.link == "logit":
            return np.asarray(special.expit(eta), dtype=float)
        if self.link == "probit":
            return np.asarray(special.ndtr(eta), dtype=float)
        return -np.expm1(-np.exp(np.clip(eta, -700, 7)))

    def forward(self, mu: Array) -> Array:
        if self.link == "identity":
            return mu.copy()
        if self.link == "log":
            return np.log(mu)
        if self.link == "inverse":
            return 1 / mu
        if self.link == "logit":
            return np.asarray(special.logit(mu), dtype=float)
        if self.link == "probit":
            return np.asarray(special.ndtri(mu), dtype=float)
        return np.log(-np.log1p(-mu))

    def derivatives(self, eta: Array, mu: Array) -> tuple[Array, Array]:
        if self.link == "identity":
            return np.ones_like(eta), np.zeros_like(eta)
        if self.link == "log":
            return mu, mu
        if self.link == "inverse":
            return -mu**2, 2 * mu**3
        if self.link == "logit":
            first = mu * (1 - mu)
            return first, first * (1 - 2 * mu)
        if self.link == "probit":
            first = np.exp(-eta**2 / 2) / np.sqrt(2 * np.pi)
            return first, -eta * first
        value = np.exp(np.clip(eta, -700, 7))
        first = value * np.exp(-value)
        return first, first * (1 - value)

    def variance(self, mu: Array) -> tuple[Array, Array]:
        if self.name == "gaussian":
            return np.ones_like(mu), np.zeros_like(mu)
        if self.name == "binomial":
            return mu * (1 - mu), 1 - 2 * mu
        if self.name == "poisson":
            return mu, np.ones_like(mu)
        return mu**2, 2 * mu

    def valid(self, mu: Array) -> bool:
        if not np.isfinite(mu).all():
            return False
        if self.name == "binomial":
            return bool(np.all((mu > 0) & (mu < 1)))
        if self.name in {"Gamma", "poisson"}:
            return bool(np.all(mu > 0))
        return True

    def deviance_parts(self, y: Array, mu: Array, weights: Array) -> Array:
        if self.name == "gaussian":
            return weights * (y - mu) ** 2
        if self.name == "poisson":
            return np.asarray(2 * weights * (special.xlogy(y, y / mu) - (y - mu)), dtype=float)
        if self.name == "binomial":
            return np.asarray(2 * weights * (
                special.xlogy(y, y / mu)
                + special.xlogy(1 - y, (1 - y) / (1 - mu))
            ), dtype=float)
        return 2 * weights * ((y - mu) / mu - np.log(y / mu))

    def loglik(self, y: Array, mu: Array, weights: Array, scale: float) -> float:
        if self.name == "gaussian":
            value = weights * (y - mu) ** 2 / scale + np.log(2 * np.pi * scale / weights)
            return -0.5 * float(np.sum(value))
        if self.name == "poisson":
            return float(np.sum(weights * (special.xlogy(y, mu) - mu - special.gammaln(y + 1))))
        if self.name == "binomial":
            successes = y * weights
            constants = (
                special.gammaln(weights + 1)
                - special.gammaln(successes + 1)
                - special.gammaln(weights - successes + 1)
            )
            binomial_values = np.asarray(constants + special.xlogy(successes, mu), dtype=float)
            binomial_values += np.asarray(special.xlogy(weights - successes, 1 - mu), dtype=float)
            return float(np.sum(binomial_values))
        shape = 1 / scale
        value = (shape - 1) * np.log(y) - y / (mu * scale)
        value -= special.gammaln(shape) + shape * np.log(mu * scale)
        return float(np.sum(weights * value))

    def selection_loglik(self, y: Array, mu: Array, weights: Array, scale: float) -> float:
        """Exponential-dispersion likelihood for Laplace smoothing selection.

        Gamma prior weights specify precision, giving shape weight/scale.
        The conditional family likelihood reported by logLik retains the
        frequency-weight convention of the ordinary Gamma family.
        """
        if self.name != "Gamma":
            return self.loglik(y, mu, weights, scale)
        shape = weights/scale
        value = (shape-1)*np.log(y)-shape*y/mu
        value -= special.gammaln(shape)+shape*np.log(mu/shape)
        return float(np.sum(value))


def _family(value: Any, link: str | None) -> _Family:
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(gaussian|binomial|poisson|Gamma|gamma)(?:\((\w+)\))?\s*", value)
        if match is None:
            raise ValueError("Supported GAM families are gaussian, binomial, poisson and Gamma")
        name, declared = match.group(1), match.group(2)
        name = "Gamma" if name.lower() == "gamma" else name
        selected = link or declared or {
            "gaussian": "identity", "binomial": "logit", "poisson": "log", "Gamma": "inverse"
        }[name]
    else:
        name = type(value).__name__.lower()
        name = "Gamma" if name == "gamma" else name
        selected = link or type(value.link).__name__.lower()
        selected = {"inversepower": "inverse", "loglog": "cloglog"}.get(selected, selected)
    allowed = {
        "gaussian": {"identity", "log", "inverse"},
        "binomial": {"logit", "probit", "cloglog"},
        "poisson": {"log", "identity"},
        "Gamma": {"inverse", "log", "identity"},
    }
    if name not in allowed or selected not in allowed[name]:
        raise ValueError(f"Unsupported GAM family/link: {name}/{selected}")
    return _Family(name, selected)


def _inverse(matrix: Array) -> Array:
    """Diagonal equilibration preserves the unpenalized directions at large lambda."""
    diagonal = np.sqrt(np.maximum(np.diag(matrix), np.finfo(float).tiny))
    scaled = matrix / diagonal[:, None] / diagonal[None, :]
    try:
        factor = linalg.cho_factor(scaled, lower=True, check_finite=False)
        result = linalg.cho_solve(factor, np.eye(len(matrix)), check_finite=False)
    except linalg.LinAlgError:
        result = linalg.pinvh(scaled, rtol=1e-12)
    return np.asarray(result / diagonal[:, None] / diagonal[None, :], dtype=float)


def _logdet(matrix: Array) -> float:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sign, value = np.linalg.slogdet(matrix)
    if sign <= 0 or not np.isfinite(value):
        raise linalg.LinAlgError("Penalized model information is not positive definite")
    return float(value)


@dataclass
class _Inner:
    beta: Array
    mu: Array
    eta: Array
    information: Array
    observed: Array
    inverse: Array
    edf: Array
    edf1: Array
    deviance: float
    penalty: float
    iterations: int
    converged: bool
    logdet_hessian: float


def _penalized_qr(X: Array, weights: Array, root: Array, target: Array) -> tuple[Array, Array, float]:
    weighted = np.sqrt(weights)
    augmented = np.vstack([X*weighted[:, None], root])
    response = np.r_[target*weighted, np.zeros(root.shape[0])]
    Q, R = linalg.qr(augmented, mode="economic", check_finite=False)
    beta = linalg.solve_triangular(R, Q.T@response, check_finite=False)
    inverse_factor = linalg.solve_triangular(R, np.eye(X.shape[1]), check_finite=False)
    covariance = inverse_factor@inverse_factor.T
    determinant = 2*float(np.sum(np.log(np.abs(np.diag(R)))))
    return np.asarray(beta, dtype=float), np.asarray(covariance, dtype=float), determinant


def _root_logdet(root: Array) -> float:
    if root.shape[1] == 0:
        return 0.0
    R = linalg.qr(root, mode="r", check_finite=False)[0]
    diagonal = np.abs(np.diag(R))
    if np.any(diagonal <= 0):
        raise linalg.LinAlgError("Penalty root is rank deficient")
    return 2*float(np.sum(np.log(diagonal)))


def _component_penalty_root(matrix: Array, space: Array, null_dimension: int) -> Array:
    """Factor a smooth on its original support before rotating coefficients.

    Factoring an already rotated full matrix can leak rounding errors between
    independent smooth blocks. A large smoothing multiplier amplifies those
    errors. This order retains exact structural zeros in each component.
    """
    support = np.flatnonzero(np.max(np.abs(matrix), axis=0) > 0)
    if not len(support):
        return np.zeros((0, space.shape[1]))
    values, vectors = linalg.eigh(matrix[np.ix_(support, support)])
    positive = values > max(float(np.max(values)), 1.0) * 1e-12
    native_root = np.zeros((int(np.sum(positive)), len(matrix)))
    native_root[:, support] = np.sqrt(values[positive])[:, None] * vectors[:, positive].T
    root = native_root @ space
    root[:, :null_dimension] = 0
    return root


def _inverse_root(X: Array, weights: Array, root: Array) -> Array:
    augmented = np.vstack([X*np.sqrt(weights)[:, None], root])
    R = linalg.qr(augmented, mode="r", check_finite=False)[0][:X.shape[1], :]
    return np.asarray(linalg.solve_triangular(R, np.eye(X.shape[1]), check_finite=False), dtype=float)


def _criterion_derivatives(value: Array, objective: Any, analytic_gradient: Any) -> tuple[Array, Array]:
    """Differentiate the profiled criterion without interpreting optimizer flags."""
    count = len(value)
    gradient = np.zeros(count)
    hessian = np.zeros((count, count))
    step = 1e-3
    if analytic_gradient is not None:
        gradient = np.asarray(analytic_gradient(value), dtype=float)
        for i in range(count):
            delta = np.zeros(count)
            delta[i] = step
            coarse = (analytic_gradient(value+delta)-analytic_gradient(value-delta))/(2*step)
            fine = (analytic_gradient(value+delta/2)-analytic_gradient(value-delta/2))/step
            hessian[:, i] = (4*fine-coarse)/3
        return gradient, (hessian+hessian.T)/2
    baseline = objective(value)
    for i in range(count):
        delta = np.zeros(count)
        delta[i] = step
        plus, minus = objective(value+delta), objective(value-delta)
        half_plus, half_minus = objective(value+delta/2), objective(value-delta/2)
        gradient[i] = (4*(half_plus-half_minus)/step-(plus-minus)/(2*step))/3
        hessian[i, i] = (plus+minus-2*baseline)/step**2
        for j in range(i):
            other = np.zeros(count)
            other[j] = step
            hessian[i, j] = hessian[j, i] = (
                objective(value+delta+other)-objective(value+delta-other)
                - objective(value-delta+other)+objective(value-delta-other)
            )/(4*step**2)
    return gradient, hessian


def _penalized_stationary(
    X: Array, y: Array, weights: Array, family: _Family, inner: _Inner, root: Array,
    *, beta: Array | None = None,
) -> bool:
    """Certify an inner optimum at floating-point objective resolution.

    The score is whitened with the augmented QR information factor. Its
    squared Newton decrement estimates the attainable decrease in deviance
    plus penalty. Comparing it with machine precision avoids a false
    failure from subtracting large normal-equation penalty entries or from an
    IRLS coefficient step that has stopped changing the fitted likelihood.
    This check does not change the coefficients or an iteration tolerance.
    """
    first, second = family.derivatives(inner.eta, inner.mu)
    variance, variance_prime = family.variance(inner.mu)
    working = weights*first**2/variance
    observed_weights = weights*(first**2/variance-(y-inner.mu)*(
        second/variance-first**2*variance_prime/variance**2))
    inverse_root = _inverse_root(X, working, root)
    predictor_root = X@inverse_root
    penalty_root = root@inverse_root
    curvature = (predictor_root.T@(observed_weights[:, None]*predictor_root)
                 + penalty_root.T@penalty_root)
    curvature = (curvature+curvature.T)/2
    coefficient = inner.beta if beta is None else beta
    score = root.T@(root@coefficient)-X.T@(weights*(y-inner.mu)*first/variance)
    whitened_score = inverse_root.T@score
    if not np.isfinite(curvature).all() or not np.isfinite(whitened_score).all():
        return False
    try:
        factor = linalg.cho_factor(curvature, lower=True, check_finite=False)
        decrement = float(whitened_score@linalg.cho_solve(factor, whitened_score, check_finite=False))
    except linalg.LinAlgError:
        return False
    resolution = 8*np.finfo(float).eps*(1+abs(inner.deviance)+abs(inner.penalty))
    return bool(decrement >= 0 and decrement <= resolution)


def _criterion_stationary(gradient: Array, hessian: Array, criterion: float) -> bool:
    """Require the actual outer score and nonnegative local curvature.

    A square-root machine-precision score is the usual precision limit when
    function values round before the next optimizer line search can improve
    them. Curvature has the same relative numerical margin. An optimizer's
    success or failure message never substitutes for either check.
    """
    if not np.isfinite(criterion) or not np.isfinite(gradient).all() or not np.isfinite(hessian).all():
        return False
    precision = np.sqrt(np.finfo(float).eps)
    if np.max(np.abs(gradient), initial=0) > precision*max(1, abs(criterion)):
        return False
    eigenvalues = linalg.eigvalsh(hessian)
    return bool(np.min(eigenvalues, initial=0) >= -precision*max(1, np.max(np.abs(eigenvalues), initial=0)))


def _irls(
    X: Array, y: Array, weights: Array, offset: Array, family: _Family, penalty: Array,
    max_iter: int, tol: float, penalty_root: Array | None = None,
) -> _Inner:
    if penalty_root is None:
        values, vectors = linalg.eigh(penalty)
        positive = values > max(float(np.max(values)), 1.0)*1e-12
        penalty_root = np.sqrt(values[positive])[:, None]*vectors[:, positive].T

    def penalty_value(beta: Array) -> float:
        return float(np.sum((penalty_root @ beta)**2))

    if family.name == "gaussian" and family.link == "identity":
        information = X.T @ (weights[:, None] * X)
        beta, inverse, logdet_hessian = _penalized_qr(X, weights, penalty_root, y-offset)
        eta = X @ beta + offset
        mu = eta.copy()
        observed = information.copy()
        converged, iteration = True, 1
    else:
        if family.name == "binomial":
            starting = (weights * y + 0.5) / (weights + 1)
        elif family.name == "poisson":
            starting = y + 0.1
        else:
            starting = np.maximum(y, np.finfo(float).eps)
        beta = np.asarray(linalg.lstsq(X, family.forward(starting) - offset)[0], dtype=float)
        eta = X @ beta + offset
        mu = family.inverse(eta)
        if not family.valid(mu):
            beta = np.zeros(X.shape[1])
            intercept = np.flatnonzero(np.all(np.isclose(X, 1), axis=0))
            if len(intercept):
                beta[intercept[0]] = float(family.forward(np.array([np.average(starting, weights=weights)]))[0])
            else:
                beta = linalg.lstsq(X, family.forward(starting) - offset)[0]
            eta = X @ beta + offset
            mu = family.inverse(eta)
        objective = float(np.sum(family.deviance_parts(y, mu, weights))) + penalty_value(beta)
        converged = False
        for iteration in range(1, max_iter + 1):
            first, second = family.derivatives(eta, mu)
            variance, variance_prime = family.variance(mu)
            variance = np.maximum(variance, np.finfo(float).tiny)
            working = weights * first**2 / variance
            adjusted = eta - offset + (y - mu) / first
            information = X.T @ (working[:, None] * X)
            candidate, inverse, _ = _penalized_qr(X, working, penalty_root, adjusted)
            step = candidate - beta
            step_size = 1.0
            next_beta, next_eta, next_mu = beta, eta, mu
            next_objective = objective
            for _ in range(60):
                proposed = beta + step_size * step
                proposed_eta = X @ proposed + offset
                proposed_mu = family.inverse(proposed_eta)
                if family.valid(proposed_mu):
                    value = float(np.sum(family.deviance_parts(y, proposed_mu, weights))) + penalty_value(proposed)
                    if value <= objective + 1e-10 * (1 + abs(objective)):
                        next_beta, next_eta, next_mu, next_objective = proposed, proposed_eta, proposed_mu, value
                        break
                step_size *= 0.5
            improvement = abs(objective - next_objective)
            beta, eta, mu = next_beta, next_eta, next_mu
            objective = next_objective
            if improvement < tol * (1 + abs(objective)) and np.max(np.abs(step)) < tol * (1 + np.max(np.abs(beta))):
                converged = True
                break
        first, second = family.derivatives(eta, mu)
        variance, variance_prime = family.variance(mu)
        working = weights * first**2 / variance
        information = X.T @ (working[:, None] * X)
        _, inverse, _ = _penalized_qr(X, working, penalty_root, np.zeros(len(y)))
        observed_weights = weights * (
            first**2 / variance
            - (y - mu) * (second / variance - first**2 * variance_prime / variance**2)
        )
        observed = X.T @ (observed_weights[:, None] * X)
        if np.all(observed_weights > 0):
            _, _, logdet_hessian = _penalized_qr(X, observed_weights, penalty_root, np.zeros(len(y)))
        else:
            logdet_hessian = _logdet(observed+penalty)
    influence = inverse @ information
    edf = np.asarray(np.diag(influence), dtype=float)
    edf1 = np.asarray(np.diag(2 * influence - influence @ influence), dtype=float)
    return _Inner(beta, mu, eta, information, observed, inverse, edf, edf1,
                  float(np.sum(family.deviance_parts(y, mu, weights))),
                  penalty_value(beta), iteration, converged, logdet_hessian)


@dataclass
class GamResult:
    """A penalized GAM with fitting-time basis information for safe prediction."""

    formula: str
    data: pd.DataFrame
    design: Any
    X: Array
    y: Array
    beta: Array
    cov_beta: Array
    coef_names: list[str]
    family: str
    link: str
    method: str
    sp: Array
    scale: float
    edf: Array
    edf1: Array
    weights: Array
    offset: Array
    linear_predictor: Array
    fitted_values: Array
    objective: float
    converged: bool
    iterations: int
    _inner: _Inner
    _penalty: Array
    _outer_result: Any
    _scale_known: bool
    _likelihood_scale: float
    _criterion_function: Any
    _smoothing_solution: Array
    _gamma: float
    _penalty_root: Array
    _observed_root: Array | None
    _criterion_gradient: Any

    @property
    def nobs(self) -> int:
        """Number of observations retained by the fitted formula."""
        return len(self.y)

    @property
    def df_resid(self) -> float:
        """Residual degrees of freedom, n minus the influence-matrix trace."""
        return float(self.nobs - np.sum(self.edf))

    @property
    def fixed_spec(self) -> Any:
        """Reusable fitted design for prediction and common model adapters."""
        # The reusable design implements get_model_matrix for common adapters.
        return self.design

    @property
    def fixed_formula(self) -> str:
        """Original formula used by the fitted design."""
        return self.formula

    @property
    def coefficients(self) -> pd.Series:
        """Named coefficients in the fitted spline basis coordinates."""
        return self.coef()

    @property
    def smooth_edf(self) -> pd.Series:
        """Influence-matrix degrees of freedom summed within each smooth."""
        return pd.Series({s.label: float(np.sum(self.edf[s.indices])) for s in self.design.smooths})

    def coef(self) -> pd.Series:
        """Return coefficients in this model's spline basis coordinates."""
        return pd.Series(self.beta.copy(), index=self.coef_names, name="Estimate")

    def vcov(self, *, freq: bool = False) -> pd.DataFrame:
        """Bayesian covariance, or the frequentist sandwich when requested."""
        covariance = self.cov_beta
        if freq:
            covariance = self.scale * self._inner.inverse @ self._inner.information @ self._inner.inverse
        return pd.DataFrame(covariance, index=self.coef_names, columns=self.coef_names)

    def fitted(self) -> pd.Series:
        """Fitted response means on the retained observations."""
        return pd.Series(self.fitted_values.copy(), index=self.data.index, name="fitted")

    def residuals(self, type: str = "deviance") -> pd.Series:
        """Response, Pearson, working, or signed deviance residuals."""
        family = _Family(self.family, self.link)
        value = self.y - self.fitted_values
        if type == "deviance":
            value = np.sign(value) * np.sqrt(np.maximum(0, family.deviance_parts(self.y, self.fitted_values, self.weights)))
        elif type == "pearson":
            value = value * np.sqrt(self.weights / family.variance(self.fitted_values)[0])
        elif type == "working":
            value = value / family.derivatives(self.linear_predictor, self.fitted_values)[0]
        elif type != "response":
            raise ValueError("Residual type must be response, deviance, pearson or working")
        return pd.Series(value, index=self.data.index, name="residual")

    def deviance(self) -> float:
        """Weighted family deviance relative to the saturated mean model."""
        return self._inner.deviance

    def logLik(self) -> float:
        """Conditional data log likelihood at the fitted mean and scale."""
        likelihood_scale = self._likelihood_scale
        if self.family == "gaussian":
            likelihood_scale = self.deviance() / self.nobs
        return _Family(self.family, self.link).loglik(self.y, self.fitted_values, self.weights, likelihood_scale)

    def AIC(self) -> float:
        """Conditional AIC using the influence trace and estimated-scale count.

        This conditions on selected smoothing parameters, following the
        effective-parameter convention of Wood (2017), chapter 6.
        """
        return -2 * self.logLik() + 2 * (float(np.sum(self.edf)) + int(not self._scale_known))

    def criterion_in_basis(self, transform: Any, *, constraint: Any = None) -> float:
        """Evaluate the likelihood criterion in explicitly supplied coefficient coordinates.

        ``transform @ beta`` defines the new coefficient vector. REML's
        improper prior on unpenalized coefficients contributes a coordinate
        dependent additive constant. This method changes that measure without
        changing the fitted likelihood, penalty, or smoothing parameters.
        For a jointly unidentifiable design, ``constraint`` supplies column
        normals imposing constraint.T @ beta_new = 0. The likelihood integral
        then uses the orthonormal measure on that specified coefficient slice.
        """
        if self.method == "GCV.Cp":
            return self.objective
        mapping = np.asarray(transform, dtype=float)
        if mapping.shape != (len(self.beta), len(self.beta)):
            raise ValueError("The coefficient transform must be square and nonsingular")
        inverse_mapping = linalg.inv(mapping)
        penalty = inverse_mapping.T @ self._penalty @ inverse_mapping
        observed = inverse_mapping.T @ self._inner.observed @ inverse_mapping
        # The rank is fixed by the active penalty spaces, not by lambda's size.
        base = sum((np.asarray(s) for value, s in zip(self.sp, self.design.S) if value > 0), np.zeros_like(penalty))
        base = inverse_mapping.T @ base @ inverse_mapping
        eigenvalues, eigenvectors = linalg.eigh(base)
        active = eigenvalues > max(float(np.max(np.abs(eigenvalues))), 1.0) * 1e-10
        rank = int(np.sum(active))
        null_count = len(self.beta)-rank
        rotated_penalty = eigenvectors.T @ penalty @ eigenvectors
        rotated_penalty[:null_count, :] = 0
        rotated_penalty[:, :null_count] = 0
        rotated_information = eigenvectors.T @ observed @ eigenvectors+rotated_penalty
        prior_root = self._penalty_root @ inverse_mapping
        prior_det = _root_logdet(prior_root@eigenvectors[:, active]) if rank else 0.0
        information_root = None
        if self._observed_root is not None:
            information_root = np.vstack([self._observed_root@inverse_mapping, prior_root])
        if self.method == "ML":
            determinant = (_root_logdet(information_root@eigenvectors[:, active]) if information_root is not None else _logdet(rotated_information[null_count:, null_count:])) if rank else 0.0
            null_correction = 0.0
        elif constraint is not None:
            normals = np.asarray(constraint, dtype=float)
            if normals.ndim == 1:
                normals = normals[:, None]
            if normals.ndim != 2 or normals.shape[0] != len(self.beta):
                raise ValueError("Constraint normals must have one row per transformed coefficient")
            target_X = self.X@inverse_mapping
            joint_null = linalg.null_space(np.vstack([target_X, base]), rcond=1e-12)
            constraint_rank = np.linalg.matrix_rank(normals.T@joint_null, tol=1e-10*max(float(np.linalg.norm(normals)), 1e-100)) if joint_null.shape[1] else 0
            if normals.shape[1] != joint_null.shape[1] or constraint_rank != joint_null.shape[1]:
                raise ValueError("Constraint normals must remove exactly the joint design/penalty nullspace")
            coefficient_slice = linalg.null_space(normals.T, rcond=1e-12)
            family = _Family(self.family, self.link)
            first, second = family.derivatives(self.linear_predictor, self.fitted_values)
            variance, variance_prime = family.variance(self.fitted_values)
            full_weights = self.weights*(first**2/variance-(self.y-self.fitted_values)*(second/variance-first**2*variance_prime/variance**2))
            if np.all(full_weights > 0):
                full_root = np.vstack([target_X*np.sqrt(full_weights)[:, None], prior_root])
                determinant = _root_logdet(full_root@coefficient_slice)
            else:
                full_information = target_X.T@(full_weights[:, None]*target_X)+penalty
                determinant = _logdet(coefficient_slice.T@full_information@coefficient_slice)
            null_correction = (len(self.beta)-rank)*np.log(2*np.pi*self._likelihood_scale)
        elif np.linalg.matrix_rank(self.X) == len(self.beta):
            determinant = _root_logdet(information_root) if information_root is not None else _logdet(rotated_information)
            null_correction = (len(self.beta) - rank) * np.log(2*np.pi*self._likelihood_scale)
        else:
            if information_root is not None:
                target_observed_root = self._observed_root@inverse_mapping
                values, vectors = linalg.eigh(target_observed_root.T@target_observed_root+base)
                positive = values > max(float(np.max(values)), 1.0)*1e-12
                determinant = _root_logdet(information_root@vectors[:, positive])
            else:
                values = linalg.eigvalsh(rotated_information)
                positive = values > max(float(np.max(values)), 1.0) * 1e-12
                determinant = float(np.sum(np.log(values[positive])))
            null_correction = (len(self.beta) - rank) * np.log(2*np.pi*self._likelihood_scale)
        return (-_Family(self.family, self.link).selection_loglik(self.y, self.fitted_values, self.weights, self._likelihood_scale)/self._gamma
                + self._inner.penalty/(2*self._likelihood_scale)
                + 0.5*(determinant-prior_det-null_correction))

    def smoothing_derivatives(self) -> tuple[Array, Array]:
        """Gradient and Hessian of the profiled outer selection criterion.

        Analytic score derivatives follow Wood (2011). Curvature uses
        Richardson differentiation of that score; unsupported score formulas
        use Richardson differentiation of the objective instead.
        """
        return _criterion_derivatives(self._smoothing_solution, self._criterion_function,
                                      self._criterion_gradient)

    def predict(
        self, newdata: Any = None, *, type: str = "link", se_fit: bool = False,
        terms: list[str] | None = None, exclude: list[str] | None = None,
        offset: Any = None,
    ) -> Any:
        """Predict means, link values, individual terms, or the linear-predictor matrix.

        Standard errors condition on the estimated smoothing parameters, as in
        the usual Bayesian GAM covariance. Term predictions exclude the intercept.
        """
        frame = self.data if newdata is None else as_dataframe(newdata)
        X = self.X.copy() if newdata is None else np.asarray(self.design.predict(frame), dtype=float)
        if type == "lpmatrix":
            return X
        if type not in {"link", "response", "terms", "iterms"}:
            raise ValueError("Prediction type must be link, response, terms, iterms or lpmatrix")
        if type in {"terms", "iterms"}:
            columns: dict[str, Array] = {}
            errors: dict[str, Array] = {}
            for label, indices in self.design.term_slices.items():
                intercept_term = all(self.coef_names[index] in {"Intercept", "(Intercept)"} for index in indices)
                if intercept_term or (terms and label not in terms):
                    continue
                block = X[:, indices]
                beta = self.beta[indices]
                covariance = self.cov_beta[np.ix_(np.arange(len(self.beta))[indices], np.arange(len(self.beta))[indices])]
                columns[label] = block @ beta
                errors[label] = np.sqrt(np.maximum(0, np.einsum("ij,jk,ik->i", block, covariance, block)))
                if exclude and label in exclude:
                    columns[label] = np.zeros(len(frame))
                    errors[label] = np.zeros(len(frame))
            fit = pd.DataFrame(columns, index=frame.index)
            fit.attrs["constant"] = float(np.sum(self.beta[[name in {"Intercept", "(Intercept)"} for name in self.coef_names]]))
            return {"fit": fit, "se.fit": pd.DataFrame(errors, index=frame.index)} if se_fit else fit
        if exclude:
            for label in exclude:
                if label in self.design.term_slices:
                    X[:, self.design.term_slices[label]] = 0
        value = X @ self.beta
        prediction_offset = self.offset if newdata is None else np.zeros(len(frame))
        if offset is not None:
            prediction_offset = _vector(offset, frame, 0)
        elif newdata is not None and hasattr(self.design, "prediction_offset"):
            prediction_offset = np.asarray(self.design.prediction_offset(frame), dtype=float)
        value += prediction_offset
        standard_error = np.sqrt(np.maximum(0, np.einsum("ij,jk,ik->i", X, self.cov_beta, X)))
        if type == "response":
            family = _Family(self.family, self.link)
            mu = family.inverse(value)
            standard_error *= np.abs(family.derivatives(value, mu)[0])
            value = mu
        fitted_series = pd.Series(value, index=frame.index, name="fit")
        return {"fit": fitted_series, "se.fit": pd.Series(standard_error, index=frame.index)} if se_fit else fitted_series

    def summary_tables(self) -> dict[str, Any]:
        """Numerical coefficient and approximate smooth significance tables.

        Smooth Wald tests use Bayesian covariance and the alternative influence
        degrees of freedom 2 tr(A)-tr(A²). P-values condition on smoothing
        parameters; they do not include smoothing-selection uncertainty.
        """
        from ._summary import summary_tables

        return summary_tables(self)

    def summary(self) -> str:
        """Format coefficient and Wood (2013) approximate smooth-test tables."""
        tables = self.summary_tables()
        output = [f"Family: {self.family}", f"Link function: {self.link}", "", "Formula:", self.formula,
                  "", "Parametric coefficients:", tables["parametric"].to_string(float_format=lambda v: f"{v:.5g}")]
        if len(tables["smooth"]):
            output += ["", "Approximate significance of smooth terms:", tables["smooth"].to_string(float_format=lambda v: f"{v:.5g}")]
        output += ["", f"R-sq.(adj) = {tables['r.sq']:.4g}   Deviance explained = {100*tables['dev.expl']:.4g}%",
                   f"{self.method} = {self.objective:.6g}  Scale est. = {self.scale:.6g}  n = {self.nobs}"]
        return "\n".join(output)

    def partial_effects(self, term: str | None = None, *, n: int = 100, newdata: Any = None) -> Any:
        """Return smooth effect and standard-error data for plotting.

        Supplied newdata supports arbitrary multidimensional grids. Otherwise a
        one-dimensional grid is generated for each smooth's first covariate, with
        remaining covariates fixed at their median or first factor level.
        """
        labels = [s.label for s in self.design.smooths]
        if term is not None and term not in labels:
            raise ValueError(f"Unknown smooth term {term!r}")
        selected = [term] if term is not None else labels
        result: dict[str, pd.DataFrame] = {}
        for label in selected:
            smooth = next(s for s in self.design.smooths if s.label == label)
            if newdata is None:
                baseline = {name: (float(column.median()) if pd.api.types.is_numeric_dtype(column) else column.iloc[0]) for name, column in self.data.items()}
                if getattr(smooth, "level", None) is not None:
                    baseline[smooth.by] = smooth.level
                variables = list(getattr(smooth, "variables", ()))
                if not variables:
                    variables = [part.strip() for part in label.split("(", 1)[1].split(")", 1)[0].split(",") if part.strip() in self.data]
                variable = variables[0]
                column = self.data[variable]
                grid = np.linspace(float(column.min()), float(column.max()), n) if pd.api.types.is_numeric_dtype(column) else np.asarray(pd.unique(column))
                frame = pd.DataFrame({name: np.repeat(value, len(grid)) for name, value in baseline.items()})
                frame[variable] = grid
            else:
                frame = as_dataframe(newdata).copy()
            predicted = self.predict(frame, type="terms", se_fit=True, terms=[label])
            frame["fit"] = predicted["fit"][label]
            frame["se"] = predicted["se.fit"][label]
            result[label] = frame
        return result[term] if term is not None else result

    def check(self, *, k_rep: int = 200, k_sample: int = 5000, seed: int | None = 0) -> dict[str, Any]:
        """Return numerical convergence and residual basis-dimension diagnostics.

        Basis adequacy uses the nearest-neighbor residual statistic described
        by Wood (2017), section 5.9, with seeded residual permutations.
        """
        return gam_check(self, k_rep=k_rep, k_sample=k_sample, seed=seed)


def gam(
    formula: str, data: Any, *, family: Any = "gaussian", link: str | None = None,
    method: str = "GCV.Cp", weights: Any = None, offset: Any = None,
    sp: Any = None, scale: float = 0, gamma: float = 1,
    control: dict[str, Any] | None = None,
) -> GamResult:
    """Fit a Gaussian, binomial, Poisson or Gamma penalized additive model.

    Smooths are specified with s(), te(), or ti(). Negative entries in ``sp``
    request estimation; nonnegative entries fix the corresponding penalty.
    The default ``GCV.Cp`` uses GCV for unknown scale and UBRE for known scale.
    ``ML`` and ``REML`` optimize the Laplace marginal likelihood.
    """
    if method not in {"GCV.Cp", "ML", "REML"}:
        raise ValueError("GAM method must be GCV.Cp, ML or REML")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    frame = as_dataframe(data)
    design = build_design(formula, frame)
    frame = design.data
    X, response = np.asarray(design.X, dtype=float), np.asarray(design.y, dtype=float)
    trials = np.ones(len(frame))
    if response.ndim == 2 and response.shape[1] == 2:
        trials = np.sum(response, axis=1)
        if np.any(response < 0) or np.any(trials <= 0):
            raise ValueError("Binomial success/failure responses need nonnegative counts and positive totals")
        y = response[:, 0] / trials
    else:
        y = response.ravel()
    prior_weights = _vector(weights, frame, 1)
    offset_values = _vector(offset, frame, 0)
    if hasattr(design, "offset"):
        offset_values += np.asarray(design.offset, dtype=float)
    if not np.isfinite(prior_weights).all() or np.any(prior_weights <= 0):
        raise ValueError("GAM weights must be positive and finite")
    if not np.isfinite(offset_values).all():
        raise ValueError("GAM offsets must be finite")
    family_info = _family(family, link)
    if response.ndim == 2 and response.shape[1] == 2 and family_info.name != "binomial":
        raise ValueError("Two-column GAM responses require binomial family")
    prior_weights *= trials
    if family_info.name == "binomial" and np.any((y < 0) | (y > 1)):
        raise ValueError("Binomial GAM responses must be in [0, 1]")
    if family_info.name == "poisson" and np.any(y < 0):
        raise ValueError("Poisson GAM responses must be nonnegative")
    if family_info.name == "Gamma" and np.any(y <= 0):
        raise ValueError("Gamma GAM responses must be positive")
    if X.shape[1] > len(y):
        raise ValueError("GAM has more coefficients than observations; reduce k")
    penalties = [np.asarray(matrix, dtype=float) for matrix in design.S]
    count = len(penalties)
    requested = np.full(count, -1.0) if sp is None else np.atleast_1d(np.asarray(sp, dtype=float))
    if len(requested) != count or not np.isfinite(requested).all():
        raise ValueError(f"sp must have one finite value per penalty ({count})")
    free = requested < 0
    controls = control or {}
    max_iter = int(controls.get("max_iter", controls.get("maxit", 150)))
    tol = float(controls.get("tol", controls.get("epsilon", 1e-11)))
    outer_max = int(controls.get("outer_max_iter", 150))
    scale_known = scale > 0 or (scale == 0 and family_info.name in {"binomial", "poisson"})
    known_scale = float(scale) if scale > 0 else 1.0
    if method != "GCV.Cp" and family_info.name in {"binomial", "poisson"} and (not scale_known or known_scale != 1):
        raise ValueError("Binomial and Poisson ML/REML require scale 1")
    total_base = sum(penalties, np.zeros((X.shape[1], X.shape[1])))
    active_base = sum((matrix for value, matrix in zip(requested, penalties) if value != 0), np.zeros_like(total_base))
    eigvalues, eigvectors = linalg.eigh(active_base)
    positive = eigvalues > max(float(np.max(np.abs(eigvalues))), 1.0) * 1e-10
    random_space = np.asarray(eigvectors[:, positive], dtype=float)
    null_dimension = X.shape[1] - int(np.sum(positive))
    base_information = X.T @ X + total_base
    base_values, base_vectors = linalg.eigh(base_information)
    identifiable = base_values > max(float(np.max(base_values)), 1.0) * 1e-12
    coefficient_space = np.asarray(base_vectors[:, identifiable], dtype=float)
    rank_deficient = coefficient_space.shape[1] < X.shape[1]
    if rank_deficient:
        # Retain an explicit zero coefficient for a redundant unpenalized
        # smooth column (for example by*w's constant already modeled by w).
        # This preserves parametric coefficients and the persistent full design.
        retained = list(range(X.shape[1]))
        parametric = set(np.asarray(design.parametric_indices, dtype=int).tolist())
        zero_penalty = np.max(np.abs(total_base), axis=0) < max(1.0, np.max(np.abs(total_base))) * 1e-12
        target_rank = int(np.sum(identifiable))
        for column in np.flatnonzero(zero_penalty):
            if column in parametric:
                continue
            retained_trial = [index for index in retained if index != column]
            if np.linalg.matrix_rank(base_information[np.ix_(retained_trial, retained_trial)], tol=max(float(np.max(base_values)), 1.0)*1e-12) == target_rank:
                retained = retained_trial
            if len(retained) == target_rank:
                coefficient_space = np.eye(X.shape[1])[:, retained]
                break
    else:
        coefficient_space = np.eye(X.shape[1])
    # Solve in a penalty eigenbasis. Large smoothing parameters otherwise
    # subtract nearly equal matrix entries to recover the unpenalized subspace.
    reduced_base = coefficient_space.T @ active_base @ coefficient_space
    reduced_values, reduced_vectors = linalg.eigh(reduced_base)
    reduced_positive = reduced_values > max(float(np.max(np.abs(reduced_values))), 1.0) * 1e-10
    inner_space = coefficient_space @ reduced_vectors
    inner_null_dimension = int(np.sum(~reduced_positive))
    penalty_roots = [
        _component_penalty_root(matrix, inner_space, inner_null_dimension)
        for matrix in penalties
    ]
    n = len(y)
    effective_n = n / gamma
    cache: dict[tuple[float, ...], tuple[float, _Inner, Array, float, Array]] = {}

    def evaluate(log_sp: Array) -> tuple[float, _Inner, Array, float, Array]:
        key = tuple(float(value) for value in log_sp)
        if key in cache:
            return cache[key]
        smoothing = np.maximum(requested, 0).copy()
        smoothing[free] = np.exp(log_sp[:int(np.sum(free))])
        penalty = sum((value * matrix for value, matrix in zip(smoothing, penalties)), np.zeros_like(total_base))
        inner_penalty = inner_space.T @ penalty @ inner_space
        inner_penalty[:inner_null_dimension, :] = 0
        inner_penalty[:, :inner_null_dimension] = 0
        inner_root = np.vstack([np.sqrt(value)*root for value, root in zip(smoothing, penalty_roots)]) if penalty_roots else np.zeros((0, inner_space.shape[1]))
        reduced = _irls(X @ inner_space, y, prior_weights, offset_values, family_info,
                        inner_penalty, max_iter, tol, inner_root)
        information = inner_space @ reduced.information @ inner_space.T
        inverse = inner_space @ reduced.inverse @ inner_space.T
        influence = inverse @ information
        inner = _Inner(inner_space @ reduced.beta, reduced.mu, reduced.eta, information,
                       inner_space @ reduced.observed @ inner_space.T,
                       inverse, np.diag(influence), np.diag(2*influence-influence@influence),
                       reduced.deviance, reduced.penalty, reduced.iterations, reduced.converged, reduced.logdet_hessian)
        edf_total = float(np.sum(inner.edf))
        dispersion = known_scale if scale_known else max(inner.deviance / max(n - edf_total, 1e-8), 1e-12)
        if method == "GCV.Cp":
            score = inner.deviance / n - known_scale + 2 * gamma * known_scale * edf_total / n if scale_known else n * inner.deviance / max(n - gamma * edf_total, 1e-8)**2
        else:
            if not scale_known and family_info.name == "gaussian":
                denominator = effective_n - (null_dimension if method == "REML" else 0)
                dispersion = max((inner.deviance / gamma + inner.penalty) / denominator, 1e-12)
            elif not scale_known:
                numerator = inner.deviance/gamma+inner.penalty
                degrees = null_dimension if method == "REML" else 0
                def scale_equation(log_phi: float) -> float:
                    phi = np.exp(log_phi)
                    shape = prior_weights/phi
                    saturated_score = np.sum(prior_weights*(special.digamma(shape)-np.log(shape)))/gamma
                    return float(saturated_score+numerator/2+degrees*phi/2)
                low, high = -25.0, 10.0
                if scale_equation(low) <= 0:
                    dispersion = float(np.exp(low))
                else:
                    dispersion = float(np.exp(optimize.brentq(scale_equation, low, high, xtol=1e-13)))
            prior_det = 0.0
            if random_space.shape[1]:
                # Zero fixed smoothing parameters leave extra unpenalized directions.
                active_space = random_space
                prior_det = _root_logdet(inner_root[:, inner_null_dimension:])
            else:
                active_space = random_space
            information = inner.observed + penalty
            if method == "ML":
                first, second = family_info.derivatives(inner.eta, inner.mu)
                variance, variance_prime = family_info.variance(inner.mu)
                observed_weights = prior_weights*(first**2/variance-(y-inner.mu)*(second/variance-first**2*variance_prime/variance**2))
                if np.all(observed_weights > 0):
                    root = np.vstack([(X@inner_space)[:, inner_null_dimension:]*np.sqrt(observed_weights)[:, None], inner_root[:, inner_null_dimension:]])
                    determinant = _root_logdet(root)
                else:
                    determinant = _logdet(active_space.T @ information @ active_space) if active_space.shape[1] else 0.0
                null_correction = 0.0
            else:
                determinant = reduced.logdet_hessian
                # The nominal unpenalized dimension remains part of the REML
                # scale convention when a redundant numeric-by column is retained.
                null_correction = (X.shape[1] - active_space.shape[1]) * np.log(2 * np.pi * dispersion)
            loglik = family_info.selection_loglik(y, inner.mu, prior_weights, dispersion)
            score = -loglik / gamma + inner.penalty / (2 * dispersion)
            score += 0.5 * (determinant - prior_det - null_correction)
        result = (float(score), inner, smoothing, float(dispersion), penalty)
        if len(cache) > 500:
            cache.clear()
        cache[key] = result
        return result

    start = np.zeros(int(np.sum(free)), dtype=float)
    bounds = [(-25.0, 25.0)] * int(np.sum(free))
    outer = None
    gradient_function = None
    if len(start):
        def objective(value: Array) -> float:
            try:
                result = evaluate(value)[0]
                return result if np.isfinite(result) else 1e100
            except (ValueError, FloatingPointError, linalg.LinAlgError):
                return 1e100
        def gradient(value: Array) -> Array:
            if family_info.name == "gaussian" and family_info.link == "identity" and gamma == 1:
                _, inner, smoothing, dispersion, _ = evaluate(value)
                result = np.empty(len(value))
                reduced_X = X@inner_space
                roots = np.vstack([np.sqrt(lam)*root for lam, root in zip(smoothing, penalty_roots)]) if penalty_roots else np.zeros((0, inner_space.shape[1]))
                posterior_root = _inverse_root(reduced_X, prior_weights, roots)
                reduced_inverse = posterior_root@posterior_root.T
                reduced_beta = inner_space.T @ inner.beta
                residual = y-inner.mu
                weighted_predictor_root = np.sqrt(prior_weights)[:, None]*(reduced_X@posterior_root)
                random_roots = roots[:, inner_null_dimension:]
                if random_roots.shape[1]:
                    prior_root = _inverse_root(np.zeros((0, random_roots.shape[1])), np.zeros(0), random_roots)
                    random_root = _inverse_root(reduced_X[:, inner_null_dimension:], prior_weights, random_roots)
                else:
                    prior_root = random_root = np.zeros((0, 0))
                for index, penalty_index in enumerate(np.flatnonzero(free)):
                    root = penalty_roots[penalty_index]
                    lam = smoothing[penalty_index]
                    root_beta = root@reduced_beta
                    beta_derivative = -lam*reduced_inverse@root.T@root_beta
                    deviance_derivative = -2*float(residual@(prior_weights*(reduced_X@beta_derivative)))
                    edf_derivative = -lam*float(np.sum((weighted_predictor_root@(root@posterior_root).T)**2))
                    if method == "GCV.Cp":
                        denominator = n-float(np.sum(inner.edf))
                        result[index] = (deviance_derivative/n+2*known_scale*edf_derivative/n if scale_known
                                         else n*deviance_derivative/denominator**2+2*n*inner.deviance*edf_derivative/denominator**3)
                    else:
                        posterior_trace = lam*float(np.sum((root@posterior_root)**2)) if method == "REML" else lam*float(np.sum((root[:, inner_null_dimension:]@random_root)**2))
                        prior_trace = lam*float(np.sum((root[:, inner_null_dimension:]@prior_root)**2))
                        result[index] = .5*(lam*float(root_beta@root_beta)/dispersion+posterior_trace-prior_trace)
                return result
            if family_info.name in {"binomial", "poisson", "Gamma"} and gamma == 1:
                _, inner, smoothing, dispersion, _ = evaluate(value)
                first, second = family_info.derivatives(inner.eta, inner.mu)
                variance, variance_prime = family_info.variance(inner.mu)
                working = prior_weights*first**2/variance
                observed_weights = prior_weights*(first**2/variance-(y-inner.mu)*(second/variance-first**2*variance_prime/variance**2))
                if np.all(observed_weights > 0):
                    inner_X = X@inner_space
                    roots = np.vstack([np.sqrt(lam)*root for lam, root in zip(smoothing, penalty_roots)]) if penalty_roots else np.zeros((0, inner_space.shape[1]))
                    observed_inverse_root = _inverse_root(inner_X, observed_weights, roots)
                    expected_inverse_root = _inverse_root(inner_X, working, roots)
                    observed_inverse = observed_inverse_root@observed_inverse_root.T
                    reduced_beta = inner_space.T@inner.beta
                    if family_info.name == "binomial" and family_info.link == "logit":
                        observed_prime = working*(1-2*inner.mu)
                    elif family_info.name == "poisson" and family_info.link == "log":
                        observed_prime = working
                    elif family_info.name == "poisson":
                        observed_prime = -2*prior_weights*y/inner.mu**3
                    elif family_info.name == "Gamma" and family_info.link == "inverse":
                        observed_prime = -2*prior_weights*inner.mu**3
                    elif family_info.name == "Gamma" and family_info.link == "log":
                        observed_prime = -prior_weights*y/inner.mu
                    elif family_info.name == "Gamma":
                        observed_prime = prior_weights*(2/inner.mu**3-6*y/inner.mu**4)
                    else:
                        step_eta = 1e-4
                        def observed_at(eta: Array) -> Array:
                            mu = family_info.inverse(eta)
                            d, d2 = family_info.derivatives(eta, mu)
                            V, Vp = family_info.variance(mu)
                            return prior_weights*(d*d/V-(y-mu)*(d2/V-d*d*Vp/(V*V)))
                        observed_prime = (observed_at(inner.eta+step_eta)-observed_at(inner.eta-step_eta))/(2*step_eta)
                    working_prime = prior_weights*(2*first*second/variance-first**3*variance_prime/variance**2)
                    observed_leverage = np.sum((inner_X@observed_inverse_root)**2, axis=1)
                    expected_predictor_root = inner_X@expected_inverse_root
                    expected_leverage = np.sum(expected_predictor_root**2, axis=1)
                    covariance_rows = expected_predictor_root@expected_predictor_root.T*np.sqrt(working)[None, :]
                    frequentist_leverage = np.sum(covariance_rows**2, axis=1)
                    random_roots = roots[:, inner_null_dimension:]
                    prior_inverse_root = _inverse_root(np.zeros((0, random_roots.shape[1])), np.zeros(0), random_roots) if random_roots.shape[1] else np.zeros((0, 0))
                    result = np.zeros(len(value))
                    for index, penalty_index in enumerate(np.flatnonzero(free)):
                        root = penalty_roots[penalty_index]
                        lam = smoothing[penalty_index]
                        root_beta = root@reduced_beta
                        derivative_score = lam*root.T@root_beta
                        beta_derivative = -observed_inverse@derivative_score
                        eta_derivative = inner_X@beta_derivative
                        if method == "GCV.Cp":
                            deviance_derivative = -2*float((roots@reduced_beta)@(roots@beta_derivative))
                            d_weights = working_prime*eta_derivative
                            expected_penalty_root = root@expected_inverse_root
                            penalty_edf = lam*float(np.sum((np.sqrt(working)[:, None]*(expected_predictor_root@expected_penalty_root.T))**2))
                            edf_derivative = float(d_weights@(expected_leverage-frequentist_leverage))-penalty_edf
                            denominator = n-float(np.sum(inner.edf))
                            result[index] = (deviance_derivative/n+2*known_scale*edf_derivative/n if scale_known
                                             else n*deviance_derivative/denominator**2+2*n*inner.deviance*edf_derivative/denominator**3)
                        else:
                            posterior_trace = lam*float(np.sum((root@observed_inverse_root)**2))
                            prior_trace = lam*float(np.sum((root[:, inner_null_dimension:]@prior_inverse_root)**2))
                            hessian_derivative = float((observed_prime*eta_derivative)@observed_leverage)
                            if method == "ML":
                                random_inverse_root = _inverse_root(inner_X[:, inner_null_dimension:], observed_weights, random_roots)
                                posterior_trace = lam*float(np.sum((root[:, inner_null_dimension:]@random_inverse_root)**2))
                                random_leverage = np.sum((inner_X[:, inner_null_dimension:]@random_inverse_root)**2, axis=1)
                                hessian_derivative = float((observed_prime*eta_derivative)@random_leverage)
                            result[index] = .5*(lam*float(root_beta@root_beta)/dispersion+posterior_trace+hessian_derivative-prior_trace)
                    return result
            # Log-scale central differences avoid cancellation in likelihood
            # determinants that occurs with the optimizer's absolute 1e-8 step.
            step = 1e-4
            result = np.empty(len(value))
            for index in range(len(value)):
                delta = np.zeros(len(value))
                delta[index] = step
                result[index] = (objective(value + delta) - objective(value - delta)) / (2 * step)
            return result
        gradient_function = gradient
        outer = optimize.minimize(objective, start, jac=gradient, method="L-BFGS-B", bounds=bounds,
                                  options={"ftol": 1e-16, "gtol": 1e-9, "maxiter": outer_max, "maxls": 40, "finite_diff_rel_step": 1e-5})
        if int(np.sum(free)) > 1 and np.any(outer.x[:int(np.sum(free))] > 12):
            # A nearly flat infinite-smoothing limit can trap gradient methods
            # before another component reaches its finite optimum. Re-enter the
            # interior and retain the solution with the actual lower criterion.
            restart = np.asarray(outer.x, dtype=float).copy()
            restart[:int(np.sum(free))] = np.minimum(restart[:int(np.sum(free))], 6)
            candidate = optimize.minimize(objective, restart, jac=gradient, method="L-BFGS-B", bounds=bounds,
                                          options={"ftol": 1e-16, "gtol": 1e-9, "maxiter": outer_max, "maxls": 40})
            if candidate.fun < outer.fun - 1e-12:
                outer = candidate
        if method == "GCV.Cp" and int(np.sum(free)) > 1:
            # Prediction-error criteria can have distinct rough and smooth
            # minima. Search both sides of the initial scale for multiple terms.
            for displacement in (-2.0, 2.0):
                restart = start.copy()
                restart[:int(np.sum(free))] += displacement
                candidate = optimize.minimize(objective, restart, jac=gradient, method="L-BFGS-B", bounds=bounds,
                                              options={"ftol": 1e-16, "gtol": 1e-9, "maxiter": outer_max, "maxls": 40})
                if candidate.fun < outer.fun - 1e-12:
                    outer = candidate
        # Marginal likelihood and prediction-error criteria can both have a
        # competing boundary minimum beyond a finite local optimum.
        for index in range(int(np.sum(free))):
            for limit in (-12.0, 12.0):
                probe = np.asarray(outer.x, dtype=float).copy()
                probe[index] = limit
                if objective(probe) < outer.fun-1e-10:
                    candidate = optimize.minimize(objective, probe, jac=gradient, method="L-BFGS-B", bounds=bounds,
                                                  options={"ftol": 1e-16, "gtol": 1e-9, "maxiter": outer_max, "maxls": 40})
                    if candidate.fun < outer.fun-1e-12:
                        outer = candidate
        if not outer.success:
            alternate = optimize.minimize(objective, outer.x, method="Powell", bounds=bounds,
                                          options={"xtol": 1e-7, "ftol": 1e-12, "maxiter": outer_max})
            if alternate.fun < outer.fun:
                outer = alternate
        # Objective-relative stopping can leave a small but consequential
        # coefficient error when the optimum is shallow. Refine the actual
        # score equations for finite components, retaining the same minimum.
        finite = np.flatnonzero(np.abs(outer.x) < 18)
        if len(finite):
            reference = np.asarray(outer.x, dtype=float).copy()
            def finite_score(value: Array) -> Array:
                point = reference.copy()
                point[finite] = value
                clipped = np.clip(point, -25, 25)
                return gradient(clipped)[finite]+1e3*(point-clipped)[finite]
            polished = optimize.root(finite_score, reference[finite], method="hybr",
                                    options={"xtol": 1e-10, "maxfev": 100})
            candidate_point = reference.copy()
            candidate_point[finite] = polished.x
            if np.isfinite(candidate_point).all() and np.all(np.abs(candidate_point) <= 25):
                candidate_score = objective(candidate_point)
                if candidate_score <= outer.fun+1e-9 and np.max(np.abs(finite_score(polished.x))) < np.max(np.abs(finite_score(reference[finite]))):
                    outer.x = candidate_point
                    outer.fun = candidate_score
                    outer.jac = gradient(candidate_point)
        solution = np.asarray(outer.x, dtype=float)
    else:
        solution = start
    score, inner, smoothing, dispersion, penalty = evaluate(solution)
    likelihood_scale = dispersion
    if not scale_known:
        # Fletcher's variance estimator divides Pearson dispersion by
        # 1 + mean(V'(mu) (y-mu)/V(mu)); see Fletcher (2012), Biometrika 99.
        variance, variance_prime = family_info.variance(inner.mu)
        pearson = float(np.sum(prior_weights * (y-inner.mu)**2 / variance)) / max(n-float(np.sum(inner.edf)), 1e-8)
        scale_estimator = controls.get("scale_est", controls.get("scale.est", "fletcher"))
        if scale_estimator == "fletcher":
            correction = 1 + float(np.mean(variance_prime * (y-inner.mu) / variance))
            dispersion = pearson / max(correction, 1e-8)
        elif scale_estimator == "pearson":
            dispersion = pearson
        elif scale_estimator == "deviance":
            dispersion = inner.deviance / max(n-float(np.sum(inner.edf)), 1e-8)
        else:
            raise ValueError("scale_est must be fletcher, pearson or deviance")
        if method == "GCV.Cp":
            likelihood_scale = dispersion
    final_root = np.vstack([np.sqrt(value)*root for value, root in zip(smoothing, penalty_roots)]) if penalty_roots else np.zeros((0, inner_space.shape[1]))
    inner_stationary = _penalized_stationary(X@inner_space, y, prior_weights, family_info,
                                            inner, final_root, beta=inner_space.T@inner.beta)
    if outer is None:
        outer_stationary = True
    else:
        outer_gradient, outer_hessian = _criterion_derivatives(solution, lambda value: evaluate(value)[0],
                                                               gradient_function)
        outer_stationary = _criterion_stationary(outer_gradient, outer_hessian, score)
    converged = inner_stationary and outer_stationary
    if not converged:
        warnings.warn("GAM smoothing or IRLS optimization did not fully converge", RuntimeWarning, stacklevel=2)
    first, second = family_info.derivatives(inner.eta, inner.mu)
    variance, variance_prime = family_info.variance(inner.mu)
    observed_weights = prior_weights*(first**2/variance-(y-inner.mu)*(second/variance-first**2*variance_prime/variance**2))
    observed_root = (np.sqrt(observed_weights)[:, None] * (X@inner_space@inner_space.T)) if np.all(observed_weights > 0) else None
    return GamResult(formula, frame, design, X, y, inner.beta, dispersion * inner.inverse,
                     list(design.coef_names), family_info.name, family_info.link, method,
                     smoothing, dispersion, inner.edf, inner.edf1, prior_weights, offset_values,
                     inner.eta, inner.mu, score, converged, inner.iterations, inner, penalty,
                     outer, scale_known, likelihood_scale, lambda value: evaluate(value)[0], solution, gamma,
                     final_root@inner_space.T, observed_root, gradient_function)


def gam_check(
    model: GamResult, *, k_rep: int = 200, k_sample: int = 5000,
    seed: int | None = 0,
) -> dict[str, Any]:
    """Return convergence and numerical residual/basis diagnostics."""
    from ._check import gam_check as check_model

    return check_model(model, k_rep=k_rep, k_sample=k_sample, seed=seed)
