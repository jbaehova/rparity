"""Dense block GLS using profiled Gaussian ML and REML likelihoods.

The likelihood follows Pinheiro and Bates (2000), chapters 2 and 5. Matrix
factorizations use SciPy. R package source code is neither used nor required.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from formulaic import model_matrix
from scipy import linalg, optimize, special, stats
from statsmodels.tsa.arima_process import arma_acovf

from ._structures import Correlation, Variance


def _parts(form: str) -> tuple[str, str | None]:
    text = form.strip()
    if not text.startswith("~"):
        raise ValueError("Covariance formulas must be one-sided, starting with ~")
    pieces = text[1:].split("|")
    if len(pieces) > 2:
        raise ValueError("Only one grouping separator is allowed")
    return pieces[0].strip(), pieces[1].strip() if len(pieces) == 2 else None


def _groups(data: pd.DataFrame, expr: str | None) -> np.ndarray:
    if expr is None:
        return np.repeat("all", len(data))
    columns = [column.strip() for column in expr.replace("/", "*").split("*")]
    if any(column not in data for column in columns):
        raise ValueError(f"Grouping columns not found: {expr}")
    if data[columns].isna().any().any():
        raise ValueError("Missing grouping covariates")
    return data[columns].astype(str).agg("/".join, axis=1).to_numpy()


def _covariate(expr: str, data: pd.DataFrame, fitted: np.ndarray) -> np.ndarray:
    if expr.replace(" ", "") in {"fitted(.)", "fitted"}:
        return fitted.copy()
    if expr in data:
        return data[expr].to_numpy(dtype=float)
    if expr == "1":
        return np.ones(len(data))
    mat = model_matrix(f"0 + {expr}", data, na_action="raise")
    if mat.shape[1] != 1:
        raise ValueError("Variance/time formula must evaluate to one numeric covariate")
    return np.asarray(mat, dtype=float).ravel()


def _pacf_to_ar(values: np.ndarray) -> np.ndarray:
    coefficients = np.empty(0)
    for value in values:
        coefficients = np.r_[coefficients - value * coefficients[::-1], value]
    return coefficients


def _ar_to_pacf(coefficients: np.ndarray) -> np.ndarray:
    current = coefficients.copy()
    partials = np.empty(len(coefficients))
    for k in range(len(coefficients) - 1, -1, -1):
        value = current[-1]
        if abs(value) >= 1:
            raise ValueError("ARMA coefficients must define a stationary/invertible process")
        partials[k] = value
        current = (current[:-1] + value * current[:-1][::-1]) / (1 - value**2)
    return partials


def _vec(value: Any, count: int, default: float = 0.0) -> np.ndarray:
    if value is None:
        return np.full(count, default)
    result = np.atleast_1d(np.asarray(value, dtype=float))
    if len(result) == 1:
        result = np.repeat(result, count)
    if len(result) != count:
        raise ValueError(f"Expected {count} covariance parameters, got {len(result)}")
    return result


class _Covariance:
    def __init__(
        self,
        data: pd.DataFrame,
        correlation: Correlation | None,
        variance: Variance | None,
        fitted: np.ndarray,
    ) -> None:
        self.data, self.correlation, self.variance = data, correlation, variance
        self.fitted = fitted
        corr_form, corr_group = _parts(correlation.form if correlation else "~1")
        groups = _groups(data, corr_group)
        self.blocks = [np.flatnonzero(groups == group) for group in pd.unique(groups)]
        self.times = np.zeros(len(data), dtype=int)
        if corr_form == "1":
            for block in self.blocks:
                self.times[block] = np.arange(len(block))
        elif correlation:
            times = _covariate(corr_form, data, fitted)
            if not np.isfinite(times).all() or not np.allclose(times, np.round(times)):
                raise ValueError("Correlation covariates must be finite integers")
            self.times = times.astype(int)
        if correlation and correlation.kind != "CompSymm":
            for block in self.blocks:
                if len(np.unique(self.times[block])) != len(block):
                    raise ValueError("Time covariates must be unique within each group")
        self.cs_lower = -1 / max(1, max(map(len, self.blocks)) - 1)
        self.symm_size = len(np.unique(self.times))
        self.symm_times = np.unique(self.times)
        self.symm_rows, self.symm_cols = np.tril_indices(self.symm_size, -1)
        # R's public pairwise coefficients are ordered down columns.
        self.symm_order = np.lexsort((self.symm_rows, self.symm_cols))
        self.nc = 0
        corr_initial = np.empty(0)
        if correlation:
            if correlation.kind == "AR1":
                self.nc = 1
                corr_initial = np.arctanh(_vec(correlation.value, 1))
            elif correlation.kind == "CompSymm":
                self.nc = 1
                rho = float(_vec(correlation.value, 1)[0])
                if not self.cs_lower < rho < 1:
                    raise ValueError("Compound symmetry value is not positive definite")
                corr_initial = np.array(
                    [special.logit((rho - self.cs_lower) / (1 - self.cs_lower))]
                )
            elif correlation.kind == "ARMA":
                self.nc = correlation.p + correlation.q
                values = _vec(correlation.value, self.nc)
                corr_initial = np.arctanh(
                    np.r_[
                        _ar_to_pacf(values[: correlation.p]),
                        _ar_to_pacf(-values[correlation.p :]),
                    ]
                )
            elif correlation.kind == "Symm":
                self.nc = self.symm_size * (self.symm_size - 1) // 2
                matrix = np.eye(self.symm_size)
                rows = self.symm_rows[self.symm_order]
                cols = self.symm_cols[self.symm_order]
                matrix[rows, cols] = _vec(correlation.value, self.nc)
                matrix[cols, rows] = matrix[rows, cols]
                try:
                    factor = linalg.cholesky(matrix, lower=True)
                except linalg.LinAlgError as exc:
                    raise ValueError("Initial corSymm matrix is not positive definite") from exc
                partials = []
                for row in range(1, self.symm_size):
                    scale = 1.0
                    for col in range(row):
                        partial = factor[row, col] / scale
                        partials.append(partial)
                        scale *= np.sqrt(1 - partial**2)
                corr_initial = special.logit(np.arccos(partials) / np.pi)
            else:
                raise ValueError(f"Unknown correlation kind {correlation.kind}")
        self.var_names: list[str] = []
        self.var_levels: list[str] = []
        self.var_groups = np.repeat("all", len(data))
        var_initial = np.empty(0)
        self.var_expr = "1"
        var_fixed: dict[str, float] = {}
        if variance:
            self.var_expr, var_group = _parts(variance.form)
            self.var_groups = _groups(data, var_group)
            self.var_levels = list(pd.unique(self.var_groups))
            self.var_names = self.var_levels.copy()
            if variance.kind == "Ident":
                # Named initial values select their omitted level as the reference.
                if isinstance(variance.value, Mapping) and len(variance.value):
                    unspecified = [v for v in self.var_levels if v not in variance.value]
                    if len(unspecified) == 1:
                        self.var_levels = unspecified + [
                            v for v in self.var_levels if v not in unspecified
                        ]
                self.var_names = self.var_levels[1:]
            if isinstance(variance.value, Mapping):
                default = 1.0 if variance.kind == "Ident" else 0.0
                var_initial = np.array([variance.value.get(v, default) for v in self.var_names])
            else:
                var_initial = _vec(
                    variance.value, len(self.var_names), 1.0 if variance.kind == "Ident" else 0.0
                )
            if variance.kind == "Ident":
                if np.any(var_initial <= 0):
                    raise ValueError("varIdent standard deviation ratios must be positive")
                var_initial = np.log(var_initial)
            if variance.fixed is not None:
                if isinstance(variance.fixed, Mapping):
                    var_fixed = dict(variance.fixed)
                else:
                    var_fixed = dict.fromkeys(self.var_names, float(variance.fixed))
                for index, name in enumerate(self.var_names):
                    if name in var_fixed:
                        value = var_fixed[name]
                        if variance.kind == "Ident" and value <= 0:
                            raise ValueError("Fixed varIdent ratios must be positive")
                        var_initial[index] = np.log(value) if variance.kind == "Ident" else value
        self.initial_all = np.r_[corr_initial, var_initial]
        corr_free = np.repeat(not (correlation and correlation.fixed), self.nc)
        var_free = np.array([name not in var_fixed for name in self.var_names], dtype=bool)
        self.free = np.r_[corr_free, var_free]
        self.initial = self.initial_all[self.free]

    def expand(self, theta: np.ndarray) -> np.ndarray:
        result = self.initial_all.copy()
        result[self.free] = theta
        return result

    def corr_values(self, all_theta: np.ndarray) -> np.ndarray:
        correlation = self.correlation
        if not correlation:
            return np.empty(0)
        theta = all_theta[: self.nc]
        if correlation.kind == "AR1":
            return np.tanh(theta)
        if correlation.kind == "CompSymm":
            return self.cs_lower + (1 - self.cs_lower) * special.expit(theta)
        if correlation.kind == "ARMA":
            return np.r_[
                _pacf_to_ar(np.tanh(theta[: correlation.p])),
                -_pacf_to_ar(np.tanh(theta[correlation.p :])),
            ]
        matrix = self.symm_matrix(theta)
        return matrix[self.symm_rows[self.symm_order], self.symm_cols[self.symm_order]]

    def symm_matrix(self, theta: np.ndarray) -> np.ndarray:
        factor = np.zeros((self.symm_size, self.symm_size))
        factor[0, 0] = 1
        k = 0
        for row in range(1, self.symm_size):
            scale = 1.0
            for col in range(row):
                partial = np.cos(np.pi * special.expit(theta[k]))
                factor[row, col] = scale * partial
                scale *= np.sqrt(max(1 - partial**2, 1e-15))
                k += 1
            factor[row, row] = scale
        return factor @ factor.T

    def var_values(self, all_theta: np.ndarray) -> np.ndarray:
        values = all_theta[self.nc :]
        if self.variance and self.variance.kind == "Ident":
            return np.exp(values)
        return values

    def sd_multipliers(self, all_theta: np.ndarray) -> np.ndarray:
        if not self.variance:
            return np.ones(len(self.data))
        parameters = self.var_values(all_theta)
        if self.variance.kind == "Ident":
            values = dict(zip(self.var_names, parameters))
            return np.array([values.get(group, 1.0) for group in self.var_groups])
        values = dict(zip(self.var_names, parameters))
        power = np.array([values[group] for group in self.var_groups])
        covariate = _covariate(self.var_expr, self.data, self.fitted)
        if not np.isfinite(covariate).all():
            raise ValueError("Variance covariates must be finite")
        if self.variance.kind == "Power":
            if np.any(covariate == 0):
                raise ValueError("varPower requires nonzero variance covariates")
            log_sd = power * np.log(np.abs(covariate))
        else:
            log_sd = power * covariate
        return np.exp(log_sd)

    def matrices(self, theta: np.ndarray) -> list[np.ndarray]:
        all_theta = self.expand(theta)
        sd = self.sd_multipliers(all_theta)
        if not np.isfinite(sd).all() or np.any(sd <= 0):
            raise ValueError("Invalid residual standard deviations")
        corr = self.correlation
        corr_values = self.corr_values(all_theta)
        general = self.symm_matrix(all_theta[: self.nc]) if corr and corr.kind == "Symm" else None
        max_lag = max(int(np.ptp(self.times[block])) for block in self.blocks)
        acf = None
        if corr and corr.kind == "ARMA":
            acf = arma_acovf(
                np.r_[1, -corr_values[: corr.p]], np.r_[1, corr_values[corr.p :]], nobs=max_lag + 1
            )
            acf = acf / acf[0]
        result = []
        for block in self.blocks:
            distances = np.abs(self.times[block, None] - self.times[None, block])
            if corr is None:
                matrix = np.eye(len(block))
            elif corr.kind == "AR1":
                matrix = corr_values[0] ** distances
            elif corr.kind == "CompSymm":
                matrix = np.full((len(block), len(block)), corr_values[0])
                np.fill_diagonal(matrix, 1)
            elif corr.kind == "ARMA":
                assert acf is not None
                matrix = acf[distances]
            else:
                indices = np.searchsorted(self.symm_times, self.times[block])
                assert general is not None
                matrix = general[np.ix_(indices, indices)]
            result.append(sd[block, None] * matrix * sd[None, block])
        return result


@dataclass
class _Evaluation:
    nll: float
    beta: np.ndarray
    inverse_information: np.ndarray
    sigma2: float
    quadratic: float
    logdet: float
    logdet_information: float


def _evaluate(
    theta: np.ndarray, covariance: _Covariance, X: np.ndarray, y: np.ndarray, reml: bool
) -> _Evaluation:
    n, p = X.shape
    information = np.zeros((p, p))
    rhs = np.zeros(p)
    yvy = logdet = 0.0
    for block, matrix in zip(covariance.blocks, covariance.matrices(theta)):
        factor = linalg.cho_factor(matrix, lower=True, check_finite=False)
        xy = np.column_stack([X[block], y[block]])
        solved = linalg.cho_solve(factor, xy, check_finite=False)
        information += X[block].T @ solved[:, :p]
        rhs += X[block].T @ solved[:, p]
        yvy += float(y[block] @ solved[:, p])
        logdet += 2 * np.log(np.diag(factor[0])).sum()
    chol = linalg.cho_factor(information, lower=True)
    beta = linalg.cho_solve(chol, rhs)
    inverse = linalg.cho_solve(chol, np.eye(p))
    # Compute the residual quadratic directly to avoid cancellation for a strong signal.
    quadratic = 0.0
    residual = y - X @ beta
    for block, matrix in zip(covariance.blocks, covariance.matrices(theta)):
        factor = linalg.cho_factor(matrix, lower=True, check_finite=False)
        quadratic += float(residual[block] @ linalg.cho_solve(factor, residual[block]))
    df = n - p if reml else n
    if quadratic <= 0:
        raise ValueError("GLS residual variance must be positive")
    sigma2 = quadratic / df
    logdet_information = 2 * np.log(np.diag(chol[0])).sum()
    nll = 0.5 * (df * (np.log(2 * np.pi * sigma2) + 1) + logdet)
    if reml:
        nll += 0.5 * logdet_information
    return _Evaluation(
        float(nll),
        beta,
        inverse,
        float(sigma2),
        float(quadratic),
        float(logdet),
        float(logdet_information),
    )


def _hessian(function: Any, point: np.ndarray, step: float = 2e-4) -> np.ndarray:
    dim = len(point)
    result = np.zeros((dim, dim))
    steps = step * np.maximum(1.0, np.abs(point))
    center = function(point)
    for i in range(dim):
        a = np.zeros(dim)
        a[i] = steps[i]
        result[i, i] = (function(point + a) - 2 * center + function(point - a)) / steps[i] ** 2
        for j in range(i):
            b = np.zeros(dim)
            b[j] = steps[j]
            result[i, j] = result[j, i] = (
                function(point + a + b)
                - function(point + a - b)
                - function(point - a + b)
                + function(point - a - b)
            ) / (4 * steps[i] * steps[j])
    return result


def _gradient(function: Any, point: np.ndarray) -> np.ndarray:
    """Five-point central score, avoiding forward-difference optimum bias."""
    result = np.empty(len(point))
    for i in range(len(point)):
        step = 2e-4 * max(1.0, abs(point[i]))
        shift = np.zeros(len(point))
        shift[i] = step
        result[i] = (
            function(point - 2 * shift)
            - 8 * function(point - shift)
            + 8 * function(point + shift)
            - function(point + 2 * shift)
        ) / (12 * step)
    return result


class GLSResult:
    """A Gaussian GLS fit, with nlme-style extraction and inference methods."""

    family = "gaussian"

    def __init__(
        self,
        formula: str,
        data: pd.DataFrame,
        X: np.ndarray,
        y: np.ndarray,
        fixed_spec: Any,
        covariance: _Covariance,
        theta: np.ndarray,
        evaluation: _Evaluation,
        reml: bool,
        converged: bool,
    ) -> None:
        self.formula = self.fixed_formula = formula
        self.data, self.X, self.y, self.fixed_spec = data, X, y, fixed_spec
        self.beta = evaluation.beta
        self.coef_names = [
            "(Intercept)" if name == "Intercept" else re.sub(r"\[T\.([^\]]+)\]", r"\1", name)
            for name in fixed_spec.column_names
        ]
        self.reml = reml
        self.method = "REML" if reml else "ML"
        self.sigma = np.sqrt(evaluation.sigma2)
        # nlme reports an unbiased scale for fixed-effect covariance even under ML.
        coefficient_scale = evaluation.sigma2 * (1 if reml else len(y) / (len(y) - X.shape[1]))
        self.cov_beta = evaluation.inverse_information * coefficient_scale
        self.df_resid = len(y) - X.shape[1]
        self.nobs = len(y)
        self.converged = converged
        self.correlation = covariance.correlation
        self.weights = covariance.variance
        self._covariance, self._theta, self._evaluation = covariance, theta, evaluation
        self._parameter_cov: np.ndarray | None = None
        self._npar = X.shape[1] + len(theta) + 1
        self.correlation_parameters = pd.Series(
            covariance.corr_values(covariance.expand(theta)),
            index=self._cor_names(),
            dtype=float,
        )
        self.variance_parameters = pd.Series(
            covariance.var_values(covariance.expand(theta)),
            index=covariance.var_names,
            dtype=float,
        )

    def _cor_names(self) -> list[str]:
        if not self.correlation:
            return []
        if self.correlation.kind in {"AR1", "CompSymm"}:
            return ["Phi" if self.correlation.kind == "AR1" else "Rho"]
        if self.correlation.kind == "ARMA":
            return [f"Phi{i + 1}" for i in range(self.correlation.p)] + [
                f"Theta{i + 1}" for i in range(self.correlation.q)
            ]
        return [f"cor{i + 1}" for i in range(self._covariance.nc)]

    def fixef(self) -> pd.Series:
        """Return fixed-effect coefficients in design-column order."""
        return pd.Series(self.beta.copy(), index=self.coef_names)

    def coef(self) -> pd.Series:
        """Return GLS regression coefficients."""
        return self.fixef()

    def vcov(self) -> pd.DataFrame:
        """Return the estimated covariance matrix of regression coefficients."""
        return pd.DataFrame(self.cov_beta.copy(), index=self.coef_names, columns=self.coef_names)

    def logLik(self) -> float:
        """Return Gaussian maximized (restricted) log likelihood."""
        return -self._evaluation.nll

    def AIC(self) -> float:
        """Return Akaike's information criterion, counting all fitted parameters."""
        return -2 * self.logLik() + 2 * self._npar

    def BIC(self) -> float:
        """Return BIC using the restricted observation count for REML fits."""
        n = self.nobs - (len(self.beta) if self.reml else 0)
        return -2 * self.logLik() + np.log(n) * self._npar

    def deviance(self) -> float:
        """Return negative twice the fitted log likelihood."""
        return -2 * self.logLik()

    def fitted(self) -> pd.Series:
        """Return the estimated marginal mean for each retained observation."""
        return pd.Series(self.X @ self.beta, index=self.data.index)

    def residuals(self, type: str = "response") -> pd.Series:
        """Return response, Pearson, or covariance-whitened residuals."""
        residual = self.y - self.X @ self.beta
        if type in {"pearson", "normalized"}:
            if type == "pearson":
                sd = self._covariance.sd_multipliers(self._covariance.expand(self._theta))
                residual = residual / (self.sigma * sd)
            else:
                residual = residual.copy()
                matrices = self._covariance.matrices(self._theta)
                for block, matrix in zip(self._covariance.blocks, matrices):
                    factor = linalg.cholesky(matrix, lower=True)
                    residual[block] = linalg.solve_triangular(factor, residual[block], lower=True)
                residual /= self.sigma
        elif type != "response":
            raise ValueError("type must be response, pearson, or normalized")
        return pd.Series(residual, index=self.data.index)

    def predict(self, newdata: Any = None, *, se_fit: bool = False) -> pd.Series | pd.DataFrame:
        """Predict marginal means from the retained formulaic design specification."""
        data = self.data if newdata is None else _to_pandas(newdata)
        mat = self.fixed_spec.get_model_matrix(data, na_action="ignore")
        X = np.asarray(mat, dtype=float)
        values = X @ self.beta
        if se_fit:
            se = np.sqrt(np.einsum("ij,jk,ik->i", X, self.cov_beta, X))
            return pd.DataFrame({"fit": values, "se.fit": se}, index=data.index)
        return pd.Series(values, index=data.index)

    def coefficient_table(self, *, adjust_sigma: bool = True) -> pd.DataFrame:
        """Return the coefficient estimates, t tests, and residual degrees of freedom."""
        se = np.sqrt(np.diag(self.cov_beta))
        if not adjust_sigma and not self.reml:
            se *= np.sqrt(self.df_resid / self.nobs)
        statistic = self.beta / se
        return pd.DataFrame(
            {
                "Value": self.beta,
                "Std.Error": se,
                "t-value": statistic,
                "p-value": 2 * stats.t.sf(np.abs(statistic), self.df_resid),
            },
            index=self.coef_names,
        )

    @staticmethod
    def _horizontal_parameters(parameters: pd.Series) -> str:
        widths = [max(len(str(name)), len(f"{value:.7g}")) for name, value in parameters.items()]
        return "\n".join(
            [
                " ".join(str(name).rjust(width) for name, width in zip(parameters.index, widths)),
                " ".join(f"{value:.7g}".rjust(width) for value, width in zip(parameters, widths)),
            ]
        )

    @staticmethod
    def _triangular_correlation(
        matrix: np.ndarray, column_names: list[str], row_names: list[str] | None = None
    ) -> str:
        if len(column_names) < 2:
            return ""
        rows = column_names if row_names is None else row_names
        row_width = max(len(name) for name in rows[1:])
        widths = [max(6, len(name)) for name in column_names[:-1]]
        lines = [
            " " * (row_width + 1)
            + " ".join(name.ljust(width) for name, width in zip(column_names[:-1], widths))
        ]
        for i in range(1, len(column_names)):
            values = [
                f"{matrix[i, j]: .3f}".rjust(width) if j < i else " " * width
                for j, width in enumerate(widths)
            ]
            lines.append(rows[i].ljust(row_width) + " " + " ".join(values))
        return "\n".join(lines)

    def summary(self) -> str:
        """Format a concise nlme-style model and coefficient summary."""
        output = [
            f"Generalized least squares fit by {self.method if self.reml else 'maximum likelihood'}",
            f"  Model: {self.formula}",
            "  Data: data",
            "",
            "       AIC        BIC    logLik",
            f"{self.AIC():10.5f} {self.BIC():10.5f} {self.logLik():10.5f}",
        ]
        if self.correlation is not None:
            label = {
                "AR1": "AR(1)",
                "CompSymm": "Compound symmetry",
                "Symm": "General",
                "ARMA": f"ARMA({self.correlation.p},{self.correlation.q})",
            }[self.correlation.kind]
            output += [
                "",
                f"Correlation Structure: {label}",
                f" Formula: {self.correlation.form}",
                " Parameter estimate(s):",
            ]
            if self.correlation.kind == "Symm":
                names = [str(value) for value in range(1, self._covariance.symm_size + 1)]
                output += [
                    " Correlation:",
                    self._triangular_correlation(
                        self._covariance.symm_matrix(
                            self._covariance.expand(self._theta)[: self._covariance.nc]
                        ),
                        names,
                    ),
                ]
            else:
                output.append(self._horizontal_parameters(self.correlation_parameters))
        if self.weights is not None:
            label = {
                "Ident": "Different standard deviations per stratum",
                "Power": "Power of variance covariate",
                "Exp": "Exponential of variance covariate",
            }[self.weights.kind]
            parameters = self.variance_parameters.copy()
            if self.weights.kind == "Ident":
                parameters = pd.Series(
                    {self._covariance.var_levels[0]: 1.0, **parameters.to_dict()}
                )
            elif self._covariance.var_names == ["all"]:
                parameters.index = ["power" if self.weights.kind == "Power" else "expon"]
            if self.correlation is None:
                output.append("")
            output += [
                "Variance function:",
                f" Structure: {label}",
                f" Formula: {self.weights.form}",
                " Parameter estimates:",
                self._horizontal_parameters(parameters),
            ]
        output += [
            "",
            "Coefficients:",
            self.coefficient_table().to_string(),
            "",
            " Correlation:",
            self._triangular_correlation(
                self.cov_beta
                / np.sqrt(np.diag(self.cov_beta))[:, None]
                / np.sqrt(np.diag(self.cov_beta))[None, :],
                ["(Intr)" if name == "(Intercept)" else name for name in self.coef_names],
                self.coef_names,
            ),
            "",
            "Standardized residuals:",
            self._horizontal_parameters(
                pd.Series(
                    np.quantile(self.residuals("pearson"), [0, 0.25, 0.5, 0.75, 1]),
                    index=["Min", "Q1", "Med", "Q3", "Max"],
                )
            ),
            "",
            f"Residual standard error: {self.sigma:.7g}",
            f"Degrees of freedom: {self.nobs} total; {self.df_resid} residual",
        ]
        return "\n".join(output)

    def parameter_covariance(self) -> np.ndarray:
        """Return observed-information covariance for free structure parameters and log SD.

        This supports delta-method inference for marginal means. Coordinates are
        ``_theta`` followed by ``log(sigma)``. Singular observed information raises
        ``ValueError`` rather than silently returning invalid degrees of freedom.
        """
        if self._parameter_cov is not None:
            return self._parameter_cov.copy()
        point = np.r_[self._theta, np.log(self.sigma)]
        hessian = _hessian(self._parameter_objective, point)
        if np.min(linalg.eigvalsh(hessian)) <= 0:
            raise ValueError("Approximate parameter covariance is not positive definite")
        self._parameter_cov = linalg.inv(hessian)
        return self._parameter_cov.copy()

    def _parameter_objective(self, parameters: np.ndarray) -> float:
        ev = _evaluate(parameters[:-1], self._covariance, self.X, self.y, self.reml)
        count = self.df_resid if self.reml else self.nobs
        logsigma = parameters[-1]
        return float(
            0.5
            * (
                count * np.log(2 * np.pi)
                + 2 * count * logsigma
                + ev.logdet
                + ev.quadratic * np.exp(-2 * logsigma)
                + (ev.logdet_information if self.reml else 0)
            )
        )

    def fixed_covariance_at(self, parameters: np.ndarray) -> np.ndarray:
        """Evaluate fixed-effect covariance at structure parameters and log residual SD."""
        if len(parameters) != len(self._theta) + 1:
            raise ValueError("Parameters must contain free covariance coordinates and log(sigma)")
        ev = _evaluate(parameters[:-1], self._covariance, self.X, self.y, self.reml)
        scale = np.exp(2 * parameters[-1])
        if not self.reml:
            scale *= self.nobs / self.df_resid
        return ev.inverse_information * scale

    def satterthwaite_df(self, contrast: np.ndarray) -> float:
        """Return a delta-method Satterthwaite df for one estimable linear contrast.

        This moment approximation follows Satterthwaite (1946). It includes
        uncertainty in fitted correlations, variance powers, and residual scale.
        """
        vector = np.asarray(contrast, dtype=float).ravel()
        if len(vector) != len(self.beta):
            raise ValueError("Contrast length must match the number of coefficients")
        if len(self._theta) == 0:
            return float(self.df_resid)
        variance = float(vector @ self.cov_beta @ vector)
        if variance <= 0:
            return float("inf")
        point = np.r_[self._theta, np.log(self.sigma)]
        derivative = _gradient(
            lambda par: float(vector @ self.fixed_covariance_at(par) @ vector), point
        )
        denominator = float(derivative @ self.parameter_covariance() @ derivative)
        return 2 * variance**2 / denominator if denominator > 0 else float("inf")

    def anova(
        self, *models: GLSResult, type: str = "sequential", adjust_sigma: bool = True
    ) -> pd.DataFrame:
        """Return sequential/marginal fixed-effect F tests or a likelihood comparison.

        Sequential tests use a QR factorization after whitening the response and
        design by the fitted residual covariance. Comparisons retain the fitted
        ML/REML method and warn when restricted fixed-effect designs differ.
        """
        if models:
            fits = (self,) + models
            if any(fit.nobs != self.nobs for fit in fits):
                raise ValueError("Models must use the same observations")
            if any(fit.method != self.method for fit in fits):
                raise ValueError("Cannot compare ML and REML likelihoods")
            if self.reml and any(fit.fixed_formula != self.fixed_formula for fit in models):
                warnings.warn(
                    "REML likelihoods are not comparable with different fixed effects",
                    UserWarning,
                    stacklevel=2,
                )
            rows: list[dict[str, Any]] = []
            for i, fit in enumerate(fits):
                row: dict[str, Any] = {
                    "Model": i + 1,
                    "df": fit._npar,
                    "AIC": fit.AIC(),
                    "BIC": fit.BIC(),
                    "logLik": fit.logLik(),
                    "Test": "",
                    "L.Ratio": np.nan,
                    "p-value": np.nan,
                }
                if i:
                    df = abs(fit._npar - fits[i - 1]._npar)
                    lr = 2 * abs(fit.logLik() - fits[i - 1].logLik())
                    row.update(
                        {
                            "Test": f"{i} vs {i + 1}",
                            "L.Ratio": lr,
                            "p-value": stats.chi2.sf(lr, df) if df else np.nan,
                        }
                    )
                rows.append(row)
            return pd.DataFrame(rows)
        if type not in {"sequential", "marginal"}:
            raise ValueError("type must be sequential or marginal")
        Xw, yw = self.X.copy(), self.y.copy()
        for block, matrix in zip(self._covariance.blocks, self._covariance.matrices(self._theta)):
            factor = linalg.cholesky(matrix, lower=True)
            Xw[block] = linalg.solve_triangular(factor, self.X[block], lower=True)
            yw[block] = linalg.solve_triangular(factor, self.y[block], lower=True)
        q, _ = linalg.qr(Xw, mode="economic")
        effects = q.T @ yw
        variance = self.sigma**2
        if adjust_sigma and not self.reml:
            variance *= self.nobs / self.df_resid
        rows = []
        names = []
        for term, indices in self.fixed_spec.term_slices.items():
            selection = np.arange(self.X.shape[1])[indices]
            df = len(selection)
            if type == "sequential":
                statistic = float(np.sum(effects[selection] ** 2) / (df * variance))
            else:
                estimate = self.beta[selection]
                covariance = self.cov_beta[np.ix_(selection, selection)]
                if not adjust_sigma and not self.reml:
                    covariance = covariance * self.df_resid / self.nobs
                statistic = float(estimate @ linalg.solve(covariance, estimate) / df)
            rows.append(
                {
                    "numDF": df,
                    "F-value": statistic,
                    "p-value": stats.f.sf(statistic, df, self.df_resid),
                }
            )
            names.append("(Intercept)" if str(term) == "1" else str(term))
        return pd.DataFrame(rows, index=names)

    def intervals(self, level: float = 0.95, which: str = "all") -> dict[str, pd.DataFrame]:
        """Return t intervals for coefficients and Hessian intervals for covariance.

        Covariance intervals are computed on the optimization scale and then
        transformed. They can be unreliable near singular covariance boundaries.
        """
        if not 0 < level < 1 or which not in {"all", "coef", "var-cov"}:
            raise ValueError("Require 0 < level < 1 and which in all, coef, var-cov")
        result: dict[str, pd.DataFrame] = {}
        if which != "var-cov":
            delta = stats.t.ppf((1 + level) / 2, self.df_resid) * np.sqrt(np.diag(self.cov_beta))
            result["coef"] = pd.DataFrame(
                {"lower": self.beta - delta, "est.": self.beta, "upper": self.beta + delta},
                index=self.coef_names,
            )
        if which == "coef":
            return result
        if len(self._theta) == 0:
            count = self.df_resid if self.reml else self.nobs
            result["sigma"] = pd.DataFrame(
                {
                    "lower": [self.sigma * np.sqrt(count / stats.chi2.ppf((1 + level) / 2, count))],
                    "est.": [self.sigma],
                    "upper": [self.sigma * np.sqrt(count / stats.chi2.ppf((1 - level) / 2, count))],
                },
                index=["sigma"],
            )
            return result
        point = np.r_[self._theta, np.log(self.sigma)]

        parameter_cov = self.parameter_covariance()
        delta = stats.norm.ppf((1 + level) / 2) * np.sqrt(np.diag(parameter_cov))
        for section in ["corStruct", "varFunc"]:
            is_correlation = section == "corStruct"
            if is_correlation and not self.correlation:
                continue
            if not is_correlation and not self.weights:
                continue
            lo = self._covariance.expand(self._theta - delta[:-1])
            mid = self._covariance.expand(self._theta)
            hi = self._covariance.expand(self._theta + delta[:-1])
            transform = (
                self._covariance.corr_values if is_correlation else self._covariance.var_values
            )
            names = self._cor_names() if is_correlation else self._covariance.var_names
            lower, estimate, upper = transform(lo), transform(mid), transform(hi)
            if is_correlation and self.correlation is not None:
                # Natural correlation coordinates yield scalar intervals without
                # shifting unrelated Cholesky angles simultaneously.
                for i, value in enumerate(estimate):
                    if self.correlation.kind == "CompSymm":
                        lower_bound = self._covariance.cs_lower
                        forward = lambda v, lb=lower_bound: special.logit((v - lb) / (1 - lb))
                        inverse = lambda v, lb=lower_bound: lb + (1 - lb) * special.expit(v)
                    else:
                        forward = np.arctanh
                        inverse = np.tanh
                    if abs(value) >= 1:
                        continue
                    derivative = _gradient(
                        lambda par, fn=forward, tr=transform, j=i: float(
                            fn(tr(self._covariance.expand(par[:-1]))[j])
                        ),
                        point,
                    )
                    se = np.sqrt(max(float(derivative @ parameter_cov @ derivative), 0))
                    width = stats.norm.ppf((1 + level) / 2) * se
                    lower[i], upper[i] = (
                        inverse(forward(value) - width),
                        inverse(forward(value) + width),
                    )
            result[section] = pd.DataFrame(
                {
                    "lower": np.minimum(lower, upper),
                    "est.": estimate,
                    "upper": np.maximum(lower, upper),
                },
                index=names,
            )
        result["sigma"] = pd.DataFrame(
            {
                "lower": [np.exp(point[-1] - delta[-1])],
                "est.": [self.sigma],
                "upper": [np.exp(point[-1] + delta[-1])],
            },
            index=["sigma"],
        )
        return result


def _to_pandas(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if hasattr(data, "to_pandas"):
        try:
            return data.to_pandas()
        except ImportError:
            return pd.DataFrame(data.to_dict(as_series=False))
    return pd.DataFrame(data)


def gls(
    formula: str,
    data: Any,
    *,
    correlation: Correlation | None = None,
    weights: Variance | None = None,
    method: str = "REML",
    REML: bool | None = None,
    control: Mapping[str, Any] | None = None,
    na_action: str = "omit",
) -> GLSResult:
    """Fit Gaussian GLS with structured residual correlations and variances.

    Parameters use the public nlme convention. ``method`` selects ML or REML;
    ``control`` may contain ``maxiter``, ``ftol``, and ``multistart``. Formulaic
    supplies the fixed-effect design. Variance covariates depending on fitted
    values are updated iteratively until the fitted mean stabilizes.
    """
    if method.upper() not in {"ML", "REML"}:
        raise ValueError("method must be ML or REML")
    if na_action not in {"omit", "raise"}:
        raise ValueError("na_action must be omit or raise")
    if correlation is not None and not isinstance(correlation, Correlation):
        raise TypeError("correlation must be a correlation specification")
    if weights is not None and not isinstance(weights, Variance):
        raise TypeError("weights must be a variance specification")
    reml = method.upper() == "REML" if REML is None else REML
    frame = _to_pandas(data)
    matrices = model_matrix(formula, frame, na_action="drop" if na_action == "omit" else "raise")
    if not hasattr(matrices, "lhs") or matrices.lhs.shape[1] != 1:
        raise ValueError("GLS requires a two-sided formula with one response")
    frame = frame.loc[matrices.rhs.index].copy()
    X, y = np.asarray(matrices.rhs, dtype=float), np.asarray(matrices.lhs, dtype=float).ravel()
    if len(y) <= X.shape[1] or np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError("GLS requires a full-rank design and positive residual degrees of freedom")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Response and design must be finite")
    fixed_spec = matrices.rhs.model_spec
    fitted = X @ linalg.lstsq(X, y)[0]
    covariance = _Covariance(frame, correlation, weights, fitted)
    theta = covariance.initial.copy()
    settings = dict(control or {})
    dynamic = weights is not None and "fitted" in weights.form
    converged = True
    for outer in range(50 if dynamic else 1):
        covariance.fitted = fitted

        def objective(parameters: np.ndarray) -> float:
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    return _evaluate(parameters, covariance, X, y, reml).nll
            except (linalg.LinAlgError, ValueError, FloatingPointError):
                return 1e100

        if len(theta):
            starts = [theta]
            if settings.get("multistart", True) and not outer:
                for shift in [-0.35, 0.35]:
                    starts.append(theta + shift)
            solutions = [
                optimize.minimize(
                    objective,
                    start,
                    method="BFGS",
                    options={
                        "gtol": 2e-8,
                        "maxiter": settings.get("maxiter", 500),
                    },
                )
                for start in starts
            ]
            solution = min(solutions, key=lambda fit: fit.fun)
            # L-BFGS-B is useful when numerical BFGS stops on a precision-loss warning.
            if not solution.success:
                alternative = optimize.minimize(
                    objective,
                    solution.x,
                    method="L-BFGS-B",
                    options={
                        "ftol": settings.get("ftol", 1e-14),
                        "gtol": 2e-7,
                        "maxiter": settings.get("maxiter", 500),
                        "maxls": 40,
                    },
                )
                if alternative.fun <= solution.fun + 1e-10:
                    solution = alternative
            theta = solution.x
            # Refine the optimum against a symmetric score. This matters when a
            # tiny prediction or covariance entry is compared at relative precision.
            for _ in range(3):
                gradient = _gradient(objective, theta)
                hessian = _hessian(objective, theta)
                try:
                    if np.min(linalg.eigvalsh(hessian)) <= 1e-8:
                        break
                    step = linalg.solve(hessian, gradient)
                except linalg.LinAlgError:
                    break
                if np.linalg.norm(step) > 0.1:
                    break
                candidate = theta - step
                if objective(candidate) > objective(theta) + 1e-10:
                    break
                theta = candidate
                if np.linalg.norm(step) < 1e-9:
                    break
            converged = bool(solution.success or np.linalg.norm(solution.jac) < 1e-4)
        evaluation = _evaluate(theta, covariance, X, y, reml)
        updated = X @ evaluation.beta
        if not dynamic or np.max(np.abs(updated - fitted)) <= 1e-8 * (1 + np.max(np.abs(fitted))):
            break
        fitted = updated
    else:
        converged = False
    if not converged:
        warnings.warn("GLS optimization did not converge", RuntimeWarning, stacklevel=2)
    if correlation is not None:
        for matrix in covariance.matrices(theta):
            diagonal = np.sqrt(np.diag(matrix))
            corr_matrix = matrix / diagonal[:, None] / diagonal[None, :]
            if np.min(linalg.eigvalsh(corr_matrix)) < 1e-8:
                warnings.warn(
                    "GLS correlation matrix is nearly singular", RuntimeWarning, stacklevel=2
                )
                break
    return GLSResult(
        formula, frame, X, y, fixed_spec, covariance, theta, evaluation, reml, converged
    )
