"""Generalized mixed models fitted by a clean-room PIRLS Laplace likelihood.

The algorithm follows Ly et al. (2026), sections 2.5-2.6.
R is used only as a development-time numerical oracle.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import linalg, optimize, stats

from rparity.formula import (
    as_dataframe,
    build_fixed_design,
    build_random_design,
    group_values,
    parse_formula,
)

from ._glmer_math import (
    conditional_mode,
    conditional_terms,
    information_derivative,
    inverse_link,
    observed_information,
    score_hessian,
)

Array = NDArray[np.float64]


def _as_frame(data: Any) -> pd.DataFrame:
    return as_dataframe(data)


def _vector(value: Any, frame: pd.DataFrame, default: float) -> Array:
    if value is None:
        return np.full(len(frame), default, dtype=float)
    if isinstance(value, str):
        return np.asarray(frame[value], dtype=float).copy()
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(len(frame), float(array))
    if array.shape != (len(frame),):
        raise ValueError("weights and offset must have one value per data row")
    return array.copy()


def _family(family: Any, link: str | None) -> tuple[str, str]:
    if isinstance(family, str):
        match = re.fullmatch(r"\s*(binomial|poisson)(?:\((\w+)\))?\s*", family)
        if match is None:
            raise ValueError("Supported families are binomial and poisson")
        name = match.group(1)
        selected = link or match.group(2) or ("logit" if name == "binomial" else "log")
    else:
        name = type(family).__name__.lower()
        selected = link or type(family.link).__name__.lower()
        selected = {"loglog": "cloglog"}.get(selected, selected)
    if (
        (name == "binomial" and selected not in {"logit", "probit", "cloglog"})
        or (name == "poisson" and selected != "log")
        or name not in {"binomial", "poisson"}
    ):
        raise ValueError(f"Unsupported family/link combination: {name}/{selected}")
    return name, selected


@dataclass
class GlmerResult:
    """Laplace maximum likelihood fit with fixed and conditional random effects."""

    formula: str
    data: pd.DataFrame
    fixed_formula: str
    X: Array
    y: Array
    beta: Array
    cov_beta: Array
    coef_names: list[str]
    fixed_spec: Any
    family: str
    link: str
    theta: Array
    random_blocks: list[Any]
    random_covariances: list[Array]
    random_modes: Array
    random_cond_cov: Array
    linear_predictor: Array
    trials: Array
    weights: Array
    offset: Array
    objective: float
    terms: dict[str, list[int]]
    contrasts: str = "treatment"
    converged: bool = True
    reml: bool = False
    offset_expressions: list[str] = field(default_factory=list)
    offset_column: str | None = None
    offset_requires_new_values: bool = False

    @property
    def nobs(self) -> int:
        """Number of retained observations."""
        return len(self.y)

    @property
    def prior_weights(self) -> Array:
        """Original effective prior weights, including binomial trial counts."""
        return self.weights * self.trials if self.family == "binomial" else self.weights.copy()

    @property
    def df_resid(self) -> int:
        """Residual degrees of freedom used for model summaries."""
        return self.nobs - len(self.beta) - len(self.theta)

    def fixef(self) -> pd.Series:
        """Return fixed effects on the link scale."""
        return pd.Series(self.beta, index=self.coef_names)

    def vcov(self) -> pd.DataFrame:
        """Return the fixed-effect covariance from the Laplace Hessian."""
        return pd.DataFrame(self.cov_beta, index=self.coef_names, columns=self.coef_names)

    def logLik(self) -> float:
        """Return the maximized Laplace log likelihood, including constants."""
        return -self.objective

    def AIC(self) -> float:
        """Return Akaike's information criterion."""
        return 2 * self.objective + 2 * (len(self.beta) + len(self.theta))

    def BIC(self) -> float:
        """Return Schwarz's information criterion."""
        return 2 * self.objective + np.log(self.nobs) * (len(self.beta) + len(self.theta))

    def fitted(self) -> Array:
        """Conditional fitted probabilities or Poisson means."""
        return inverse_link(self.linear_predictor, self.link)[0]

    def residuals(self, type: str = "deviance") -> Array:
        """Return response, Pearson, or conditional deviance residuals."""
        mu = self.fitted()
        raw = self.y - mu
        if type == "response":
            return raw
        variance = mu if self.family == "poisson" else mu * (1 - mu) / self.trials
        if type == "pearson":
            return raw * np.sqrt(self.weights / np.maximum(variance, 1e-15))
        if type != "deviance":
            raise ValueError("Residual type must be response, pearson, or deviance")
        if self.family == "poisson":
            from scipy.special import xlogy

            dev = 2 * self.weights * (xlogy(self.y, self.y / mu) - self.y + mu)
        else:
            from scipy.special import xlogy

            p = np.clip(mu, 1e-15, 1 - 1e-15)
            dev = (
                2
                * self.weights
                * self.trials
                * (xlogy(self.y, self.y / p) + xlogy(1 - self.y, (1 - self.y) / (1 - p)))
            )
        return np.sign(raw) * np.sqrt(np.maximum(dev, 0))

    def deviance(self) -> float:
        """Return the sum of conditional deviance residuals squared."""
        return float(np.sum(self.residuals() ** 2))

    def is_singular(self, tol: float = 1e-4) -> bool:
        """Detect covariance components on the positive-semidefinite boundary."""
        cursor = 0
        for block in self.random_blocks:
            width = len(block.names)
            for i in range(width):
                if self.theta[cursor + i] < tol:
                    return True
                cursor += i + 1
        return False

    def VarCorr(self) -> dict[str, pd.DataFrame]:
        """Return group-specific covariance matrices on the latent scale."""
        result: dict[str, pd.DataFrame] = {}
        for i, (block, cov) in enumerate(zip(self.random_blocks, self.random_covariances)):
            key = block.term.group
            if key in result:
                key = f"{key}.{i}"
            result[key] = pd.DataFrame(cov, index=block.names, columns=block.names)
        return result

    def ranef(self, cond_var: bool = False) -> dict[str, pd.DataFrame]:
        """Return conditional random-effect modes, optionally with covariance attributes."""
        result: dict[str, pd.DataFrame] = {}
        cursor = 0
        for i, block in enumerate(self.random_blocks):
            width = len(block.names)
            count = len(block.levels) * width
            values = self.random_modes[cursor : cursor + count].reshape(-1, width)
            table = pd.DataFrame(values, index=block.levels, columns=block.names)
            if cond_var:
                table.attrs["postVar"] = np.stack(
                    [
                        self.random_cond_cov[
                            cursor + j * width : cursor + (j + 1) * width,
                            cursor + j * width : cursor + (j + 1) * width,
                        ]
                        for j in range(len(block.levels))
                    ],
                    axis=2,
                )
            key = block.term.group
            if key in result:
                key = f"{key}.{i}"
            result[key] = table
            cursor += count
        return result

    def predict(
        self,
        newdata: Any = None,
        type: str = "response",
        re_form: Any = None,
        allow_new_levels: bool = False,
        offset: Any = None,
    ) -> Array:
        """Predict conditional or population-level means on link or response scale."""
        if type not in {"response", "link"}:
            raise ValueError("Prediction type must be response or link")
        population = re_form is False or (isinstance(re_form, str) and re_form in {"NA", "~0"})
        if re_form is not None and not population:
            if not (isinstance(re_form, float) and np.isnan(re_form)):
                raise NotImplementedError("Only all or no random effects are supported")
            population = True
        if newdata is None:
            eta = self.X @ self.beta + self.offset if population else self.linear_predictor.copy()
        else:
            frame = _as_frame(newdata)
            matrix = np.asarray(self.fixed_spec.get_model_matrix(frame), dtype=float)
            eta = matrix @ self.beta
            for expression in self.offset_expressions:
                eta += np.asarray(frame.eval(expression), dtype=float)
            if offset is not None:
                eta += _vector(offset, frame, 0)
            elif self.offset_column is not None:
                eta += _vector(self.offset_column, frame, 0)
            elif self.offset_requires_new_values:
                raise ValueError(
                    "Prediction with numeric training offsets requires new offset values"
                )
            if not population:
                cursor = 0
                for fitted in self.random_blocks:
                    full_values = fitted.spec.get_model_matrix(frame)
                    indices = getattr(fitted, "column_indices", None)
                    if indices is None:
                        columns = [
                            "Intercept" if name == "(Intercept)" else name for name in fitted.names
                        ]
                        values = np.asarray(full_values[columns], dtype=float)
                    else:
                        values = np.asarray(full_values.iloc[:, indices], dtype=float)
                    groups = group_values(fitted.term.group, frame)
                    width = len(fitted.names)
                    effects = self.random_modes[cursor : cursor + len(fitted.levels) * width]
                    effects = effects.reshape(-1, width)
                    mapping = {level: effects[j] for j, level in enumerate(fitted.levels)}
                    for j, level in enumerate(groups):
                        if level not in mapping:
                            if not allow_new_levels:
                                raise ValueError(f"New group level: {level}")
                        else:
                            eta[j] += float(values[j] @ mapping[level])
                    cursor += len(fitted.levels) * width
        return eta if type == "link" else inverse_link(eta, self.link)[0]

    def summary(self) -> str:
        """Format model criteria, random effects, and fixed-effect Wald tests."""
        se = np.sqrt(np.diag(self.cov_beta))
        z = self.beta / se
        p = 2 * stats.norm.sf(np.abs(z))
        scaled = np.quantile(self.residuals(type="pearson"), [0, 0.25, 0.5, 0.75, 1])
        lines = [
            "Generalized linear mixed model fit by maximum likelihood (Laplace",
            "  Approximation) [glmerMod]",
            f" Family: {self.family}  ( {self.link} )",
            f"Formula: {self.formula}",
            "   Data: data",
            "",
            "      AIC       BIC    logLik -2*log(L)  df.resid",
            (
                f"{self.AIC():9.1f} {self.BIC():9.1f} {self.logLik():9.1f} "
                f"{2 * self.objective:9.1f} {self.df_resid:9d}"
            ),
            "",
            "Scaled residuals:",
            "    Min      1Q  Median      3Q     Max",
            " ".join(f"{value:7.4f}" for value in scaled),
            "",
            "Random effects:",
        ]
        correlated = any(len(block.names) > 1 for block in self.random_blocks)
        lines.append(" Groups Name        Variance Std.Dev." + (" Corr" if correlated else ""))
        for block, covariance in zip(self.random_blocks, self.random_covariances):
            sd = np.sqrt(np.diag(covariance))
            for i, name in enumerate(block.names):
                correlations = " ".join(
                    f"{covariance[i, j] / (sd[i] * sd[j]):5.2f}" if sd[i] * sd[j] else "  NaN"
                    for j in range(i)
                )
                group = block.term.group if i == 0 else ""
                lines.append(
                    f" {group:6s} {name:11s} {covariance[i, i]:8.4f} {sd[i]:7.4f} " + correlations
                )
        groups = {block.term.group: len(block.levels) for block in self.random_blocks}
        group_description = "; ".join(f"{name}, {count}" for name, count in groups.items())
        lines += [
            f"Number of obs: {self.nobs}, groups:  {group_description}",
            "",
            "Fixed effects:",
            "            Estimate Std. Error z value Pr(>|z|)",
        ]
        for name, estimate, stderr, statistic, probability in zip(
            self.coef_names, self.beta, se, z, p
        ):
            star = (
                "***"
                if probability < 0.001
                else "**"
                if probability < 0.01
                else "*"
                if probability < 0.05
                else "."
                if probability < 0.1
                else ""
            )
            lines.append(
                f"{name:12s} {estimate:9.5f} {stderr:10.5f} {statistic:7.3f} "
                f"{probability:9.5g} {star}"
            )
        if np.any(p < 0.1):
            lines += ["---", "Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1"]
        if len(self.beta) > 1:
            correlation = self.cov_beta / np.outer(se, se)
            labels = ["(Intr)" if name == "(Intercept)" else name[:6] for name in self.coef_names]
            lines += ["", "Correlation of Fixed Effects:", "       " + " ".join(labels[:-1])]
            for i in range(1, len(labels)):
                values = " ".join(f"{correlation[i, j]:6.3f}" for j in range(i))
                lines.append(f"{labels[i]:6s} {values}")
        if self.is_singular():
            lines += ["", "boundary (singular) fit"]
        return "\n".join(lines)


def glmer(
    formula: str,
    data: Any,
    family: Any = "binomial",
    *,
    link: str | None = None,
    weights: Any = None,
    offset: Any = None,
    nAGQ: int = 1,
    contrasts: str = "treatment",
    control: dict[str, Any] | None = None,
) -> GlmerResult:
    """Fit binomial or Poisson random-effect models by Laplace ML.

    Binomial responses may be binary, proportions with trial weights,
    or ``cbind(success, failure)``. Only nAGQ=1 is implemented. Newton conditional
    modes and the Laplace determinant follow Ly et al. (2026), appendix 6.1
    and equations 18-19.
    """
    if nAGQ != 1:
        raise NotImplementedError("Only the Laplace approximation (nAGQ=1) is implemented")
    if control and set(control) - {"maxiter", "optimizer"}:
        raise ValueError("Unsupported optimizer control")
    family_name, selected_link = _family(family, link)
    frame = _as_frame(data)
    prior = _vector(weights, frame, 1)
    offs = _vector(offset, frame, 0)
    explicit_offset = offs.copy()
    parsed = parse_formula(formula)
    for expression in parsed.offsets:
        offs += np.asarray(frame.eval(expression), dtype=float)
    cbind = re.fullmatch(r"cbind\(\s*([^,]+)\s*,\s*([^)]+)\s*\)", parsed.response)
    trial = np.ones(len(frame))
    fixed_formula = parsed.fixed
    if cbind:
        if family_name != "binomial":
            raise ValueError("cbind responses require the binomial family")
        success = np.asarray(frame.eval(cbind.group(1).strip()), dtype=float)
        failure = np.asarray(frame.eval(cbind.group(2).strip()), dtype=float)
        if np.any(success < 0) or np.any(failure < 0):
            raise ValueError("Binomial successes and failures must be nonnegative")
        trial = success + failure
        frame["_rparity_response"] = np.divide(
            success, trial, out=np.zeros_like(success), where=trial > 0
        )
        fixed_formula = "_rparity_response ~" + parsed.fixed.split("~", 1)[1]
    frame["_rparity_weight"] = prior
    frame["_rparity_offset"] = offs
    frame["_rparity_trial"] = trial
    # Drop missing values once, keeping ancillary vectors and grouping rows aligned.
    needed = set(re.findall(r"\b[A-Za-z_]\w*\b", formula)) & set(frame.columns)
    needed.update({"_rparity_weight", "_rparity_offset", "_rparity_trial"})
    if cbind:
        needed.add("_rparity_response")
    frame = frame.dropna(subset=sorted(needed)).reset_index(drop=True)
    fixed = build_fixed_design(fixed_formula, frame, contrasts=contrasts)
    frame = fixed.data
    X = np.asarray(fixed.X, dtype=float)
    y = np.asarray(fixed.y, dtype=float).reshape(-1)
    prior = np.asarray(frame["_rparity_weight"], dtype=float)
    offs = np.asarray(frame["_rparity_offset"], dtype=float)
    trial = np.asarray(frame["_rparity_trial"], dtype=float)
    if len(y) <= X.shape[1] or np.linalg.matrix_rank(X) != X.shape[1]:
        raise ValueError("Fixed-effect design must have full rank and residual observations")
    if np.any(prior < 0) or not np.all(np.isfinite(prior)):
        raise ValueError("Prior weights must be finite and nonnegative")
    if not np.all(np.isfinite(offs)):
        raise ValueError("Offsets must be finite")
    if family_name == "binomial":
        if np.any((y < 0) | (y > 1)):
            raise ValueError("Binomial response must be between zero and one")
        binary_response = np.all((y == 0) | (y == 1))
        if not cbind and not binary_response:
            trial = prior.copy()
            prior = np.ones(len(y))
        count = y * trial
        if np.any(np.abs(count - np.round(count)) > 1e-7) or (
            not cbind and binary_response and np.any(np.abs(prior * y - np.round(prior * y)) > 1e-7)
        ):
            message = (
                "non-integer counts in a binomial glm!"
                if cbind
                else "non-integer #successes in a binomial glm!"
            )
            warnings.warn(message, UserWarning, stacklevel=2)
    elif np.any(y < 0) or np.any(np.abs(y - np.round(y)) > 1e-7):
        raise ValueError("Poisson response must contain nonnegative integers")
    blocks = build_random_design(parsed, frame)
    if not blocks:
        raise ValueError("glmer requires at least one random-effect term")
    Z = np.column_stack([block.Z for block in blocks])
    sizes = [len(block.names) for block in blocks]
    theta_count = sum(k * (k + 1) // 2 for k in sizes)
    p = X.shape[1]
    bounds: list[tuple[float | None, float | None]] = []
    initial_theta: list[float] = []
    for k in sizes:
        for i in range(k):
            for j in range(i + 1):
                bounds.append((0, None) if i == j else (None, None))
                initial_theta.append(0.5 if i == j else 0.0)
    bounds += [(None, None)] * p

    def covariance_factor(theta: Array) -> tuple[Array, list[Array]]:
        chunks: list[Array] = []
        covariances: list[Array] = []
        cursor = 0
        for block, k in zip(blocks, sizes):
            factor = np.zeros((k, k))
            for i in range(k):
                for j in range(i + 1):
                    factor[i, j] = theta[cursor]
                    cursor += 1
            chunks.append(np.asarray(np.kron(np.eye(len(block.levels)), factor), dtype=float))
            covariances.append(factor @ factor.T)
        return linalg.block_diag(*chunks), covariances

    # The covariance factor is linear in its free Cholesky entries. Its
    # derivatives are therefore constant even at a variance boundary.
    factor_derivatives = [
        Z @ covariance_factor(direction)[0] for direction in np.eye(theta_count)
    ]

    rounded_trial = np.round(trial)
    rounded_y = np.divide(
        np.round(trial * y), rounded_trial, out=np.zeros_like(y), where=rounded_trial > 0
    )
    density_weights = np.divide(
        prior * trial, rounded_trial, out=np.zeros_like(prior), where=rounded_trial > 0
    )
    rounded_density = family_name == "binomial" and (
        np.any(np.abs(trial - rounded_trial) > 1e-7)
        or np.any(np.abs(trial * y - np.round(trial * y)) > 1e-7)
    )

    def evaluate(parameters: Array) -> tuple[float, Array, Array, Array]:
        factor, _ = covariance_factor(parameters[:theta_count])
        value, u, eta, hessian = conditional_mode(
            X @ parameters[theta_count:] + offs,
            Z @ factor,
            y,
            trial,
            prior,
            family_name,
            selected_link,
        )
        if rounded_density and np.isfinite(value):
            # R's public binomial behavior keeps the original response and prior
            # weights for conditional modes, but its likelihood density rounds
            # trial/success counts and rescales the prior weight per trial.
            original = conditional_terms(eta, y, trial, prior, family_name, selected_link)[0]
            density = conditional_terms(
                eta, rounded_y, rounded_trial, density_weights, family_name, selected_link
            )[0]
            value += density - original
        return value, u, eta, hessian

    def objective(parameters: Array) -> float:
        return evaluate(parameters)[0]

    def objective_gradient(parameters: Array) -> Array:
        """Differentiate the likelihood and determinant through its inner mode.

        Implicit differentiation of A.T @ score + u = 0 accounts for the
        movement of the conditional mode. This avoids finite-difference
        cancellation and truncation in the outer score equations.
        """
        value, u, eta, hessian = evaluate(parameters)
        if not np.isfinite(value):
            return np.full(len(parameters), np.nan)
        factor, _ = covariance_factor(parameters[:theta_count])
        A = Z @ factor
        _, score, information = conditional_terms(
            eta, y, trial, prior, family_name, selected_link
        )
        curvature = observed_information(eta, y, trial, prior, family_name, selected_link)
        mode_hessian = (A.T * curvature) @ A + np.eye(A.shape[1])
        base_derivatives = np.column_stack([dA @ u for dA in factor_derivatives] + [X])
        rhs = A.T @ (curvature[:, None] * base_derivatives)
        for i, dA in enumerate(factor_derivatives):
            rhs[:, i] += dA.T @ score
        du = -linalg.solve(mode_hessian, rhs, assume_a="pos")
        eta_derivatives = base_derivatives + A @ du
        inverse_A = linalg.solve(hessian, A.T, assume_a="pos")
        leverage = np.einsum("ij,ji->i", A, inverse_A)
        information_slope = information_derivative(
            eta, trial, prior, family_name, selected_link
        )
        gradient = base_derivatives.T @ score
        gradient += eta_derivatives.T @ (leverage * information_slope) / 2
        for i, dA in enumerate(factor_derivatives):
            gradient[i] += np.sum(inverse_A.T * (information[:, None] * dA))
        if rounded_density:
            density_score = conditional_terms(
                eta, rounded_y, rounded_trial, density_weights, family_name, selected_link
            )[1]
            gradient += eta_derivatives.T @ (density_score - score)
        return gradient

    def glm_objective(beta: Array) -> tuple[float, Array]:
        value, score, _ = conditional_terms(
            X @ beta + offs, y, trial, prior, family_name, selected_link
        )
        return value, X.T @ score

    glm_fit = optimize.minimize(
        glm_objective, np.zeros(p), jac=True, method="BFGS", options={"gtol": 1e-8, "maxiter": 200}
    )
    start = np.r_[initial_theta, glm_fit.x]
    maxiter = int((control or {}).get("maxiter", 1000))
    optimizer = str((control or {}).get("optimizer", "L-BFGS-B"))
    if optimizer not in {"L-BFGS-B", "Powell"}:
        raise ValueError("optimizer must be L-BFGS-B or Powell")
    fit = optimize.minimize(
        objective,
        start,
        method=cast(Literal["L-BFGS-B", "Powell"], optimizer),
        jac=objective_gradient if optimizer == "L-BFGS-B" else None,
        bounds=bounds,
        options={
            "maxiter": maxiter,
            "ftol": 1e-15,
            **({"gtol": 1e-8, "maxls": 40} if optimizer == "L-BFGS-B" else {}),
        },
    )
    if not fit.success or not np.isfinite(fit.fun):
        retry = optimize.minimize(
            objective,
            fit.x if np.isfinite(fit.fun) else start,
            method="Powell",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-12, "xtol": 1e-8},
        )
        if np.isfinite(retry.fun) and retry.fun < fit.fun:
            fit = retry
    # Standard-deviation parameterization has zero derivative at zero, including
    # saddles. Test inward curvature before accepting a constrained solution.
    restart = np.asarray(fit.x, dtype=float).copy()
    restart_value = float(fit.fun)
    for i, bound in enumerate(bounds[:theta_count]):
        if bound[0] == 0 and restart[i] < 1e-5:
            candidate = restart.copy()
            candidate[i] = 0.01
            candidate_value = objective(candidate)
            if candidate_value < restart_value - 1e-8:
                restart, restart_value = candidate, candidate_value
    if restart_value < fit.fun - 1e-8:
        escaped = optimize.minimize(
            objective,
            restart,
            method="L-BFGS-B",
            jac=objective_gradient,
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-8, "maxls": 40},
        )
        if escaped.fun < fit.fun:
            fit = optimize.minimize(
                objective,
                escaped.x,
                jac=objective_gradient,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-8, "maxls": 40},
            )
            if escaped.fun < fit.fun:
                fit = escaped
    parameters = np.asarray(fit.x, dtype=float)
    # A likelihood-scale stopping test can terminate quasi-Newton updates while
    # individual coefficients still move appreciably. Polish the score equations.
    for _ in range(3):
        free_polish = np.ones(len(parameters), dtype=bool)
        for i, bound in enumerate(bounds[:theta_count]):
            if bound[0] == 0 and parameters[i] < 1e-6:
                free_polish[i] = False
        score = objective_gradient(parameters)[free_polish]
        if np.max(np.abs(score), initial=0) < 1e-10:
            break
        curvature = score_hessian(objective_gradient, parameters)[
            np.ix_(free_polish, free_polish)
        ]
        if np.min(np.linalg.eigvalsh(curvature)) <= 1e-7:
            break
        step = np.zeros(len(parameters))
        step[free_polish] = linalg.solve(curvature, score, assume_a="pos")
        scale = 1.0
        accepted = False
        while scale > 2**-12:
            candidate = parameters - scale * step
            if (
                all(bound[0] is None or candidate[i] >= bound[0] for i, bound in enumerate(bounds))
                and objective(candidate) <= objective(parameters) + 1e-11
            ):
                parameters = candidate
                accepted = True
                break
            scale /= 2
        if not accepted:
            break
    value, u, eta, inner_hessian = evaluate(parameters)
    factor, covariances = covariance_factor(parameters[:theta_count])
    hessian = score_hessian(objective_gradient, parameters)
    # At an active variance boundary nuisance parameters are constrained.
    free = np.ones(len(parameters), dtype=bool)
    for i, bound in enumerate(bounds[:theta_count]):
        if bound[0] == 0 and parameters[i] < 1e-6:
            free[i] = False
    covariance = np.linalg.pinv(hessian[np.ix_(free, free)], hermitian=True)
    covariance_beta = covariance[-p:, -p:]
    random_cov = factor @ linalg.solve(inner_hessian, factor.T, assume_a="pos")
    result = GlmerResult(
        formula,
        frame,
        parsed.fixed,
        X,
        y,
        parameters[theta_count:],
        covariance_beta,
        list(fixed.names),
        fixed.spec,
        family_name,
        selected_link,
        parameters[:theta_count],
        blocks,
        covariances,
        factor @ u,
        random_cov,
        eta,
        trial,
        prior,
        offs,
        value,
        fixed.terms,
        contrasts,
        bool(fit.success),
    )
    result.offset_expressions = parsed.offsets
    result.offset_column = offset if isinstance(offset, str) else None
    result.offset_requires_new_values = not isinstance(offset, str) and bool(
        np.any(explicit_offset)
    )
    if result.is_singular():
        warnings.warn("boundary (singular) fit", UserWarning, stacklevel=2)
    if not fit.success:
        warnings.warn(f"Model failed to converge: {fit.message}", RuntimeWarning, stacklevel=2)
    return result
