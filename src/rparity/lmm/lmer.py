"""Gaussian mixed models, derived from Bates et al. (2015), JSS 67(1).

The profiled criteria follow equations 34 and 39-41, Sections 5.1.1 and
5.1.4 (doi:10.18637/jss.v067.i01).

The likelihood is evaluated through the marginal covariance. A relative
Cholesky factor allows positive semidefinite random-effect covariances,
including fits on their boundary, without inverting those covariances.
"""

from __future__ import annotations

import warnings
from typing import Any, cast

import numpy as np
import pandas as pd
from formulaic import model_matrix
from numpy.typing import NDArray
from scipy import linalg, optimize

FloatArray = NDArray[np.float64]


class SingularFitWarning(UserWarning):
    """A fitted random-effect covariance has a deficient rank."""


class ConvergenceWarning(UserWarning):
    """A numerical optimizer did not meet its convergence criterion."""


def _array_argument(value: Any, data: pd.DataFrame, default: float) -> FloatArray:
    if value is None:
        return np.full(len(data), default)
    if isinstance(value, str):
        value = data[value]
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(len(data), float(array))
    if array.shape != (len(data),):
        raise ValueError("weights and offset must have one value per observation")
    return array


class LmerResult:
    """Fitted Gaussian mixed model with conditional and marginal accessors."""

    family = "gaussian"

    def __init__(
        self,
        formula: str,
        data: Any,
        reml: bool = True,
        weights: Any = None,
        offset: Any = None,
        contrasts: str = "treatment",
        control: dict[str, Any] | None = None,
    ) -> None:
        from rparity.formula import (
            as_dataframe,
            build_fixed_design,
            build_random_design,
            parse_formula,
        )

        data = as_dataframe(data)
        # Formulaic handles exclusions in the fixed model; group and weight
        # exclusions are subsequently synchronized before fitting either design.
        original_weights = _array_argument(weights, data, 1.0)
        original_offset = _array_argument(offset, data, 0.0)
        parsed = parse_formula(formula)
        self._offset_expressions = list(getattr(parsed, "offsets", []))
        self._has_explicit_offset = offset is not None
        self._has_weights = weights is not None
        for expression in self._offset_expressions:
            original_offset += np.asarray(data.eval(expression), dtype=float)
        data["__rparity_weight__"] = original_weights
        data["__rparity_offset__"] = original_offset
        fixed = build_fixed_design(parsed.fixed, data, contrasts=contrasts)
        clean = fixed.data.copy()
        group_columns = {
            column for term in parsed.random_terms for column in term.group.split(":")
        }
        clean = clean.dropna(subset=list(group_columns) + [
            "__rparity_weight__", "__rparity_offset__"
        ])
        # Random covariates may add missing values absent from the fixed formula.
        for term in parsed.random_terms:
            matrix = model_matrix(term.effects, clean)
            clean = clean.loc[matrix.index]
        fixed = build_fixed_design(parsed.fixed, clean, contrasts=contrasts)
        self.formula = formula
        self.fixed_formula = parsed.fixed
        self.data = fixed.data.copy()
        self.fixed_spec = fixed.spec
        self.terms = fixed.terms
        self.X = np.asarray(fixed.X, dtype=float)
        self.y = np.asarray(fixed.y, dtype=float).reshape(-1)
        self.weights = self.data.pop("__rparity_weight__").to_numpy(dtype=float)
        self.offset = self.data.pop("__rparity_offset__").to_numpy(dtype=float)
        if np.any(self.weights <= 0) or not np.isfinite(self.weights).all():
            raise ValueError("weights must be finite and strictly positive")
        if not np.isfinite(self.y).all() or not np.isfinite(self.offset).all():
            raise ValueError("response and offset must be finite")
        self.reml = bool(reml)
        self.contrasts = contrasts
        self.coef_names = list(fixed.names)
        self._column_indices = np.arange(self.X.shape[1])
        if np.linalg.matrix_rank(self.X) < self.X.shape[1]:
            _, diagonal, pivot = linalg.qr(self.X, mode="economic", pivoting=True)
            rank = np.linalg.matrix_rank(diagonal)
            self._column_indices = np.sort(pivot[:rank])
            self.X = self.X[:, self._column_indices]
            self.coef_names = [self.coef_names[i] for i in self._column_indices]
            warnings.warn("fixed-effect model matrix is rank deficient; dropping columns",
                          UserWarning, stacklevel=3)
        self.nobs, self.p = self.X.shape
        if self.nobs <= self.p:
            raise ValueError("number of observations must exceed the fixed-effect rank")
        self.random_blocks = sorted(build_random_design(parsed, self.data),
                                    key=lambda block: len(block.levels), reverse=True)
        if not self.random_blocks:
            raise ValueError("lmer requires at least one random-effect term")
        self.Z = np.column_stack([block.Z for block in self.random_blocks])
        self._slices: list[slice] = []
        self._positions: list[list[tuple[int, int]]] = []
        self._bases: list[list[FloatArray]] = []
        bounds: list[tuple[float | None, float | None]] = []
        initial: list[float] = []
        pos = 0
        for block in self.random_blocks:
            k = len(block.names)
            positions = [(row, col) for col in range(k) for row in range(col, k)]
            self._positions.append(positions)
            self._slices.append(slice(pos, pos + len(positions)))
            pos += len(positions)
            basis: list[FloatArray] = []
            for row, col in positions:
                zr = block.Z[:, row::k]
                zc = block.Z[:, col::k]
                basis.append(zr @ zc.T if row == col else zr @ zc.T + zc @ zr.T)
                initial.append(1.0 if row == col else 0.0)
                bounds.append((0.0, None) if row == col else (None, None))
            self._bases.append(basis)
        self._bounds = bounds
        self._y_adjusted = self.y - self.offset
        options = {"maxiter": 1500, "ftol": 1e-14, "gtol": 1e-8,
                   "maxls": 40, **(control or {})}
        minimize = cast(Any, optimize.minimize)
        result = minimize(self._objective_and_gradient, initial, args=(), jac=True,
                                   bounds=bounds, method="L-BFGS-B", options=options)
        # A zero-variance starting point has zero Cholesky gradient. The second
        # interior start also protects against a locally stationary solution.
        alternate = np.asarray(initial) * 0.35
        second = minimize(self._objective_and_gradient, alternate, args=(), jac=True,
                                   bounds=bounds, method="L-BFGS-B", options=options)
        if second.fun < result.fun:
            result = second
        # A Cholesky diagonal has zero score at zero even when that boundary
        # is a local maximum. Profile each active diagonal before accepting it.
        for coordinate, bound in enumerate(bounds):
            if bound[0] != 0.0 or result.x[coordinate] > 1e-6:
                continue
            point: FloatArray = np.asarray(result.x, dtype=float).copy()

            def coordinate_objective(value: float, base: FloatArray = point,
                                     index: int = coordinate) -> float:
                trial = base.copy()
                trial[index] = value
                return self._objective_and_gradient(trial)[0]

            scalar = optimize.minimize_scalar(
                coordinate_objective, bounds=(0.0, max(2.0, float(np.max(np.abs(point))) * 2)),
                method="bounded", options={"xatol": 1e-12},
            )
            if scalar.fun < result.fun - 1e-10:
                point[coordinate] = scalar.x
                interior = minimize(
                    self._objective_and_gradient, point, args=(), jac=True, bounds=bounds,
                    method="L-BFGS-B", options=options,
                )
                if interior.fun < result.fun:
                    result = interior
        self.optimizer_result = result
        self.theta = np.asarray(result.x, dtype=float)
        self._criterion, self.beta, self._cov_relative, self._quadratic, self._Vinv = (
            self._evaluate(self.theta)
        )
        self.sigma = float(np.sqrt(self._quadratic / (self.nobs - self.p if reml
                                                    else self.nobs)))
        self.cov_beta = self._cov_relative * self.sigma**2
        self.variance_params = np.r_[self.theta, self.sigma]
        self._loglik = -0.5 * self._criterion
        self._relative_G = linalg.block_diag(*[
            np.kron(np.eye(len(block.levels)), factor @ factor.T)
            for block, factor in zip(self.random_blocks, self._factors(self.theta), strict=True)
        ])
        self._random_effects = self._relative_G @ self.Z.T @ (
            self._Vinv @ (self._y_adjusted - self.X @ self.beta)
        )
        self._conditional_covariance = self.sigma**2 * (
            self._relative_G - self._relative_G @ self.Z.T @ self._Vinv @ self.Z
            @ self._relative_G
        )
        if self.is_singular():
            warnings.warn("boundary (singular) fit", SingularFitWarning, stacklevel=3)
        projected_gradient = np.array(result.jac).copy()
        for i, (lower, _) in enumerate(bounds):
            if lower is not None and self.theta[i] <= 1e-8 and projected_gradient[i] > 0:
                projected_gradient[i] = 0
        if not result.success and np.max(np.abs(projected_gradient)) > 1e-5:
            warnings.warn(f"optimizer convergence warning: {result.message}",
                          ConvergenceWarning, stacklevel=3)

    def _factors(self, theta: FloatArray) -> list[FloatArray]:
        factors = []
        for block, index, positions in zip(self.random_blocks, self._slices,
                                           self._positions, strict=True):
            factor = np.zeros((len(block.names), len(block.names)))
            for value, (row, col) in zip(theta[index], positions, strict=True):
                factor[row, col] = value
            factors.append(factor)
        return factors

    def _relative_covariance(self, theta: FloatArray) -> FloatArray:
        covariance = np.diag(1.0 / self.weights)
        for factor, bases, positions in zip(self._factors(theta), self._bases,
                                            self._positions, strict=True):
            random_covariance = factor @ factor.T
            for basis, (row, col) in zip(bases, positions, strict=True):
                covariance += random_covariance[row, col] * basis
        return covariance

    def _evaluate(self, theta: FloatArray) -> tuple[float, FloatArray, FloatArray,
                                                   float, FloatArray]:
        covariance = self._relative_covariance(theta)
        factor = linalg.cho_factor(covariance, lower=True, check_finite=False)
        inverse = linalg.cho_solve(factor, np.eye(self.nobs), check_finite=False)
        inv_x = inverse @ self.X
        information = self.X.T @ inv_x
        if self.p:
            fixed_factor = linalg.cho_factor(information, lower=True, check_finite=False)
            cov_beta = linalg.cho_solve(fixed_factor, np.eye(self.p), check_finite=False)
            beta = cov_beta @ (inv_x.T @ self._y_adjusted)
            logdet_fixed = float(2 * np.log(np.diag(fixed_factor[0])).sum())
        else:
            cov_beta = np.empty((0, 0))
            beta = np.empty(0)
            logdet_fixed = 0.0
        residual = self._y_adjusted - self.X @ beta
        quadratic = float(residual @ inverse @ residual)
        dof = self.nobs - self.p if self.reml else self.nobs
        logdet = 2 * np.log(np.diag(factor[0])).sum()
        criterion = logdet + (logdet_fixed if self.reml else 0) + dof * (
            1 + np.log(2 * np.pi * max(quadratic, np.finfo(float).tiny) / dof)
        )
        return float(criterion), beta, cov_beta, quadratic, inverse

    def _objective_and_gradient(self, theta: FloatArray) -> tuple[float, FloatArray]:
        criterion, beta, cov, quadratic, inverse = self._evaluate(theta)
        inv_x = inverse @ self.X
        projection = inverse - inv_x @ cov @ inv_x.T if self.reml else inverse
        residual_score = inverse @ (self._y_adjusted - self.X @ beta)
        dof = self.nobs - self.p if self.reml else self.nobs
        score = projection - dof / max(quadratic, np.finfo(float).tiny) * np.outer(
            residual_score, residual_score
        )
        gradient: list[float] = []
        for factor, bases, positions in zip(self._factors(theta), self._bases,
                                            self._positions, strict=True):
            component_scores = np.array([np.sum(score * basis) for basis in bases])
            for row, col in positions:
                derivative = np.zeros_like(factor)
                derivative[row, col] = 1
                dk = derivative @ factor.T + factor @ derivative.T
                gradient.append(float(component_scores @ [dk[a, b] for a, b in positions]))
        return criterion, np.asarray(gradient)

    def covariance_components(self) -> list[FloatArray]:
        """Return the linear covariance-entry basis, ending in residual variance."""
        return [basis.copy() for bases in self._bases for basis in bases] + [
            np.diag(1.0 / self.weights)
        ]

    def marginal_covariance_at(self, params: FloatArray) -> FloatArray:
        """Evaluate observation covariance at relative Cholesky and scale parameters."""
        return float(params[-1])**2 * self._relative_covariance(np.asarray(params[:-1]))

    variance_matrix = marginal_covariance_at

    def covariance_at(self, params: FloatArray) -> FloatArray:
        """Evaluate fixed-effect covariance for inference differentiation."""
        return float(params[-1])**2 * self._evaluate(np.asarray(params[:-1]))[2]

    def variance_objective(self, params: FloatArray) -> float:
        """Unprofiled negative twice ML or REML likelihood, with beta eliminated."""
        sigma2 = float(params[-1])**2
        if sigma2 <= 0:
            return np.inf
        criterion, _, _, quadratic, _ = self._evaluate(np.asarray(params[:-1]))
        dof = self.nobs - self.p if self.reml else self.nobs
        return float(criterion + dof * np.log(sigma2 / (quadratic / dof))
                     + quadratic / sigma2 - dof)

    def fixef(self) -> pd.Series:
        """Return fixed-effect coefficients in design-matrix order."""
        return pd.Series(self.beta.copy(), index=self.coef_names, name="Estimate")

    def vcov(self) -> pd.DataFrame:
        """Return the estimated covariance matrix of the fixed effects."""
        return pd.DataFrame(self.cov_beta, index=self.coef_names, columns=self.coef_names)

    def ranef(self, cond_var: bool = True, **kwargs: Any) -> dict[str, pd.DataFrame]:
        """Return conditional modes, with conditional covariance in frame attributes."""
        cond_var = kwargs.get("condVar", cond_var)
        output: dict[str, pd.DataFrame] = {}
        group_indices: dict[str, list[NDArray[np.int64]]] = {}
        start = 0
        for block in self.random_blocks:
            k, levels = len(block.names), len(block.levels)
            size = k * levels
            frame = pd.DataFrame(self._random_effects[start:start + size].reshape(levels, k),
                                 index=block.levels, columns=block.names)
            group = block.term.group
            group_indices.setdefault(group, []).append(
                np.arange(start, start + size).reshape(levels, k)
            )
            if group in output:
                previous = output[group]
                frame = pd.concat([previous, frame], axis=1)
            output[group] = frame
            start += size
        if cond_var:
            for group, frame in output.items():
                indices = np.column_stack(group_indices[group]).astype(int)
                frame.attrs["postVar"] = np.stack([
                    self._conditional_covariance[np.ix_(index, index)] for index in indices
                ], axis=2)
        return output

    def VarCorr(self) -> dict[str, pd.DataFrame]:
        """Return absolute group covariance matrices and residual variance."""
        output: dict[str, pd.DataFrame] = {}
        for block, factor in zip(self.random_blocks, self._factors(self.theta), strict=True):
            matrix = self.sigma**2 * factor @ factor.T
            group = block.term.group
            frame = pd.DataFrame(matrix, index=block.names, columns=block.names)
            if group in output:
                suffix = 1
                while f"{group}.{suffix}" in output:
                    suffix += 1
                group = f"{group}.{suffix}"
            output[group] = frame
        output["Residual"] = pd.DataFrame([[self.sigma**2]], index=["Residual"],
                                          columns=["Residual"])
        return output

    def logLik(self) -> float:
        """Return the maximized ML or REML log likelihood."""
        return self._loglik

    @property
    def df_model(self) -> int:
        """Number of estimated fixed, covariance, and scale parameters."""
        return self.p + len(self.theta) + 1

    def AIC(self) -> float:
        """Return Akaike's information criterion for the fitted likelihood."""
        return -2 * self.logLik() + 2 * self.df_model

    def BIC(self) -> float:
        """Return the Bayesian information criterion for the fitted likelihood."""
        return -2 * self.logLik() + np.log(self.nobs) * self.df_model

    def deviance(self) -> float:
        """Return negative twice the fitted likelihood (REML criterion when restricted)."""
        return self._criterion

    def fitted(self) -> FloatArray:
        """Return conditional fitted means including the supplied offset."""
        return self.X @ self.beta + self.Z @ self._random_effects + self.offset

    def residuals(self, type: str = "response") -> FloatArray:
        """Return response or Pearson residuals."""
        residual = self.y - self.fitted()
        if type == "response":
            return residual
        if type == "pearson":
            return residual * np.sqrt(self.weights) / self.sigma
        raise ValueError("residual type must be 'response' or 'pearson'")

    def is_singular(self, tol: float = 1e-4) -> bool:
        """Identify deficient random-effect covariance factors."""
        return any(np.min(np.abs(np.diag(factor))) < tol for factor in self._factors(self.theta))

    def predict(self, newdata: Any = None, re_form: Any = None,
                allow_new_levels: bool = False, offset: Any = None, **kwargs: Any) -> FloatArray:
        """Predict conditional means, or population means with re_form='NA'."""
        from rparity.formula import as_dataframe, group_values

        if "re.form" in kwargs:
            re_form = kwargs["re.form"]
        allow_new_levels = kwargs.get("allow.new.levels", allow_new_levels)
        population = re_form is False or (isinstance(re_form, str) and re_form in {"NA", "~0"})
        population = population or (isinstance(re_form, float) and np.isnan(re_form))
        if newdata is None:
            return self.X @ self.beta + self.offset if population else self.fitted()
        data = as_dataframe(newdata)
        matrix = np.asarray(self.fixed_spec.get_model_matrix(data), dtype=float)
        prediction = matrix[:, self._column_indices] @ self.beta
        if offset is not None:
            prediction += _array_argument(offset, data, 0.0)
        elif self._has_explicit_offset and np.any(self.offset):
            raise ValueError("newdata prediction requires offset when the fitted offset is nonzero")
        for expression in self._offset_expressions:
            prediction += np.asarray(data.eval(expression), dtype=float)
        if population:
            return prediction
        start = 0
        for fitted_block in self.random_blocks:
            k = len(fitted_block.names)
            new_values = fitted_block.spec.get_model_matrix(data)
            columns = ["Intercept" if name == "(Intercept)" else name
                       for name in fitted_block.names]
            values = np.asarray(new_values[columns], dtype=float)
            groups = group_values(fitted_block.term.group, data)
            modes = self._random_effects[start:start + k * len(fitted_block.levels)].reshape(-1, k)
            mapping = {level: modes[i] for i, level in enumerate(fitted_block.levels)}
            for i, group in enumerate(groups):
                if group not in mapping:
                    if not allow_new_levels:
                        raise ValueError(f"new grouping level {group!r}; set allow_new_levels=True")
                else:
                    prediction[i] += values[i] @ mapping[group]
            start += len(fitted_block.levels) * k
        return prediction

    def refit(self, reml: bool = False) -> LmerResult:
        """Refit the same observations with ML or REML."""
        formula_offset = sum((np.asarray(self.data.eval(expression), dtype=float)
                              for expression in self._offset_expressions),
                             np.zeros(self.nobs))
        return lmer(self.formula, self.data, reml=reml, weights=self.weights,
                    offset=self.offset - formula_offset, contrasts=self.contrasts)

    def summary(self, ddf: str = "satterthwaite") -> str:
        """Format R's mixed-model sections, labels, and coefficient columns.

        Numerical t tests follow Kuznetsova et al. (2017); printed model-fit
        sections follow the documented ``summary.merMod`` public interface.
        """
        restricted = "REML" if self.reml else "maximum likelihood"
        tested = ddf.lower() != "asymptotic"
        method = "Kenward-Roger" if ddf.lower().replace("_", "-") in {
            "kr", "kenward-roger", "kenwardroger"
        } else "Satterthwaite"
        header = f"Linear mixed model fit by {restricted}"
        if tested:
            header += f". t-tests use {method}'s method [lmerModLmerTest]"
        else:
            header += " [lmerMod]"
        lines = [header, f"Formula: {self.formula}", "   Data: data"]
        if self._has_weights:
            lines.append("Weights: weights")
        lines += ["Control: optimizer = 'L-BFGS-B'", ""]
        if self.reml:
            lines += [f"REML criterion at convergence: {self._criterion:.1f}", ""]
        else:
            fit_table = pd.DataFrame({
                "AIC": [self.AIC()], "BIC": [self.BIC()], "logLik": [self.logLik()],
                "-2*log(L)": [self.deviance()], "df.resid": [self.nobs-self.df_model]
            })
            lines += [fit_table.to_string(index=False, float_format=lambda x: f"{x:.1f}"), ""]
        quantiles = np.quantile(self.residuals("pearson"), [0, .25, .5, .75, 1])
        lines += ["Scaled residuals:", "    Min      1Q  Median      3Q     Max",
                  " ".join(f"{value:7.4f}" for value in quantiles), "", "Random effects:"]
        random_rows = []
        has_correlation = any(len(block.names) > 1 for block in self.random_blocks)
        for group, covariance in self.VarCorr().items():
            std = np.sqrt(np.diag(covariance))
            for i, (name, variance) in enumerate(zip(covariance.index, np.diag(covariance),
                                                    strict=True)):
                row = {"Groups": group if i == 0 else "", "Name": "" if group == "Residual" else name,
                       "Variance": f"{variance:.5g}", "Std.Dev.": f"{std[i]:.5g}"}
                if has_correlation:
                    row["Corr"] = " ".join(
                        f"{covariance.iloc[i, j]/(std[i]*std[j]):.2f}"
                        if std[i]*std[j] > 0 else "NaN" for j in range(i)
                    )
                random_rows.append(row)
        lines.append(pd.DataFrame(random_rows).to_string(index=False))
        groups = dict.fromkeys(block.term.group for block in self.random_blocks)
        lines += [f"Number of obs: {self.nobs}, groups: " + "; ".join(
            f"{group}, {len(next(block.levels for block in self.random_blocks if block.term.group == group))}"
            for group in groups
        ), "", "Fixed effects:"]
        if tested:
            from rparity import inference
            table = inference.coefficient_tests(self, ddf=ddf)
        else:
            table = pd.DataFrame({"Estimate": self.beta, "Std. Error": np.sqrt(np.diag(self.cov_beta))},
                                 index=self.coef_names)
            table["t value"] = table["Estimate"] / table["Std. Error"]
        if tested:
            pvalues = table["Pr(>|t|)"].to_numpy()
            table[""] = np.select([pvalues < .001, pvalues < .01, pvalues < .05, pvalues < .1],
                                  ["***", "**", "*", "."], default="")
        lines.append(table.to_string(float_format=lambda x: f"{x:.5g}"))
        if tested:
            lines += ["---", "Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1"]
        if self.p > 1:
            correlations = self.cov_beta / np.sqrt(np.outer(np.diag(self.cov_beta),
                                                           np.diag(self.cov_beta)))
            lines += ["", "Correlation of Fixed Effects:"]
            labels = ["(Intr)" if name == "(Intercept)" else name for name in self.coef_names]
            lines.append(" " * (max(map(len, self.coef_names)) + 1) + " ".join(labels[:-1]))
            for i in range(1, self.p):
                lines.append(self.coef_names[i] + " " + " ".join(
                    f"{correlations[i, j]:.3f}" for j in range(i)
                ))
        if self.is_singular():
            code = 0 if self.optimizer_result.success else self.optimizer_result.status
            lines.append(f"optimizer (L-BFGS-B) convergence code: {code} (OK)")
            lines.append("boundary (singular) fit: see help('isSingular')")
        return "\n".join(lines)


def lmer(formula: str, data: Any, reml: bool = True, weights: Any = None,
         offset: Any = None, contrasts: str = "treatment",
         control: dict[str, Any] | None = None, **kwargs: Any) -> LmerResult:
    """Fit a Gaussian mixed model by profiled ML or REML.

    Implements marginal covariance likelihoods from Bates et al. (2015),
    Sections 3.4 and 5.1. Random covariance factors permit singular solutions.
    ``REML`` is accepted as an alias for ``reml``.
    """
    if "REML" in kwargs:
        reml = bool(kwargs.pop("REML"))
    if kwargs:
        raise TypeError(f"unexpected lmer arguments: {', '.join(kwargs)}")
    return LmerResult(formula, data, reml, weights, offset, contrasts, control)
