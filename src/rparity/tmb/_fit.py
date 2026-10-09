"""Dense clean-room Laplace maximum likelihood for distributional GLMMs.

The normalized random-effect integral and its implicit derivatives follow
Kristensen et al. (2016). Observation densities follow Brooks et al. (2017).
No target-package source or automatic-differentiation runtime is used.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import linalg, optimize, special, stats

from rparity.formula import (
    as_dataframe,
    build_fixed_design,
    build_random_design,
    group_values,
    parse_formula,
)
from rparity.lmm._glmer_math import score_hessian

from ._covariance import (
    centered_score_hessian,
    native_coordinates,
    richardson_score_hessian,
    weak_component_information,
)
from ._families import Terms, inverse_link, observation_terms

Array = NDArray[np.float64]
_LINKS = {
    "poisson": "log",
    "nbinom1": "log",
    "nbinom2": "log",
    "binomial": "logit",
    "beta": "logit",
    "gaussian": "identity",
}


def _vector(value: Any, frame: pd.DataFrame, default: float) -> Array:
    if value is None:
        return np.full(len(frame), default, dtype=float)
    array = np.asarray(frame[value] if isinstance(value, str) else value, dtype=float)
    if array.ndim == 0:
        return np.full(len(frame), float(array))
    if array.shape != (len(frame),):
        raise ValueError("weights and offset must have one value per data row")
    return array.copy()


def _family(family: Any, link: str | None) -> tuple[str, str]:
    if isinstance(family, str):
        match = re.fullmatch(r"\s*(\w+)(?:\((\w+)\))?\s*", family)
        if match is None:
            raise ValueError("Invalid family specification")
        name = match.group(1).lower()
        chosen = link or match.group(2)
    else:
        name = type(family).__name__.lower()
        chosen = link or type(family.link).__name__.lower()
    if name == "beta_family":
        name = "beta"
    if name not in _LINKS:
        raise ValueError(f"Unsupported family: {name}")
    chosen = chosen or _LINKS[name]
    if chosen != _LINKS[name]:
        raise NotImplementedError(f"The {name} family currently requires the {_LINKS[name]} link")
    return name, chosen


def _component_formula(formula: str, conditional_rhs: str | None = None) -> str:
    if "~" not in formula:
        raise ValueError("Component formulas must contain '~'")
    rhs = formula.split("~", 1)[1].strip()
    if rhs == ".":
        if conditional_rhs is None:
            raise ValueError("~. requires a conditional formula")
        rhs = conditional_rhs
    if "|" in rhs:
        raise NotImplementedError("Zero-inflation and dispersion formulas require fixed effects")
    return "_rparity_response ~ " + rhs


@dataclass
class GlmmTMBResult:
    """Conditional, zero-inflation, and dispersion components of a Laplace fit.

    ``outer_gradient`` uses the fitted raw parameters. ``outer_information``
    differentiates scores in ``information_parameters`` coordinates; the
    ``information_transform`` Jacobian maps these coordinates back to raw
    parameters. These numeric diagnostics expose weak information directions
    without assigning them finite coefficient uncertainty.
    """

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
    ziformula: str
    dispformula: str
    zi_beta: Array
    disp_beta: Array
    zi_names: list[str]
    disp_names: list[str]
    zi_spec: Any
    disp_spec: Any
    zi_X: Array
    disp_X: Array
    joint_cov: Array
    parameters: Array
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
    converged: bool
    pd_hessian: bool
    outer_gradient: Array
    outer_information: Array
    information_parameters: Array
    information_transform: Array
    contrasts: str = "treatment"
    reml: bool = False
    offset_expressions: list[str] = field(default_factory=list)
    offset_column: str | None = None
    offset_requires_new_values: bool = False
    weights_provided: bool = False
    unidentified_fixed_indices: list[int] = field(default_factory=list)
    _model_type: str = field(default="glmmTMB", init=False)

    @property
    def joint_fixed_covariance(self) -> Array:
        """Joint covariance including cross-component fixed-effect blocks."""
        count = len(self.beta) + len(self.zi_beta) + len(self.disp_beta)
        return self.joint_cov[:count, :count].copy()

    @property
    def fixed_component_indices(self) -> dict[str, list[int]]:
        """Positions of each fixed component in the joint parameter vector."""
        p, z, d = len(self.beta), len(self.zi_beta), len(self.disp_beta)
        return {
            "cond": list(range(p)),
            "zi": list(range(p, p + z)),
            "disp": list(range(p + z, p + z + d)),
        }

    def component_data(self, component: str = "cond") -> dict[str, Any]:
        """Adapter metadata for component-specific Wald inference and EMMs."""
        if component not in {"cond", "zi", "disp"}:
            raise ValueError("component must be cond, zi, or disp")
        effects = self.fixef()[component]
        formulas = {"cond": self.fixed_formula, "zi": self.ziformula, "disp": self.dispformula}
        specs = {"cond": self.fixed_spec, "zi": self.zi_spec, "disp": self.disp_spec}
        links = {"cond": self.link, "zi": "logit", "disp": "log"}
        df = (
            self.df_resid
            if self.family == "gaussian"
            and not self.random_blocks
            and not len(self.zi_beta)
            and len(self.disp_beta) == 1
            else float("inf")
        )
        return {
            "beta": effects.to_numpy(),
            "covariance": self.vcov(component).to_numpy(),
            "X": self.model_matrix(component),
            "fixed_spec": specs[component],
            "coef_names": list(effects.index),
            "formula": formulas[component],
            "link": links[component],
            "df": df,
        }

    @property
    def nobs(self) -> int:
        """Number of retained complete-case observations."""
        return len(self.y)

    @property
    def df_resid(self) -> int:
        """Observation count less all estimated outer model parameters."""
        return self.nobs - len(self.parameters)

    def fixef(self) -> dict[str, pd.Series]:
        """Fixed coefficients for cond, zi, and disp on their link scales."""
        return {
            "cond": pd.Series(self.beta, index=self.coef_names),
            "zi": pd.Series(self.zi_beta, index=self.zi_names),
            "disp": pd.Series(self.disp_beta, index=self.disp_names),
        }

    def vcov(self, component: str = "cond", full: bool = False) -> pd.DataFrame:
        """Return a component covariance or the joint outer-parameter covariance."""
        if full:
            names = self.coef_names + ["zi~" + name for name in self.zi_names]
            names += ["disp~" + name for name in self.disp_names]
            names += [f"theta_{i + 1}" for i in range(len(self.theta))]
            return pd.DataFrame(self.joint_cov, index=names, columns=names)
        effects = self.fixef().get(component)
        if effects is None:
            raise ValueError("component must be cond, zi, or disp")
        start = 0 if component == "cond" else len(self.beta)
        if component == "disp":
            start += len(self.zi_beta)
        indices = slice(start, start + len(effects))
        return pd.DataFrame(
            self.joint_cov[indices, indices], index=effects.index, columns=effects.index
        )

    def logLik(self) -> float:
        """Maximized normalized Laplace log likelihood."""
        return -self.objective

    def AIC(self) -> float:
        """Akaike information criterion counting every estimated component."""
        return 2 * self.objective + 2 * len(self.parameters)

    def BIC(self) -> float:
        """Schwarz information criterion based on the retained observation count."""
        return 2 * self.objective + np.log(self.nobs) * len(self.parameters)

    def sigma(self) -> float:
        """Intercept-only dispersion, with Gaussian standard-deviation convention."""
        if len(self.disp_beta) == 0:
            return 1.0
        if len(self.disp_beta) != 1:
            return float("nan")
        return float(np.exp(self.disp_beta[0]))

    def VarCorr(self) -> dict[str, dict[str, pd.DataFrame]]:
        """Conditional random-effect covariance matrices, grouped by component."""
        conditional: dict[str, pd.DataFrame] = {}
        for i, (block, covariance) in enumerate(zip(self.random_blocks, self.random_covariances)):
            key = block.term.group
            if key in conditional:
                key = f"{key}.{i}"
            conditional[key] = pd.DataFrame(covariance, index=block.names, columns=block.names)
        return {"cond": conditional, "zi": {}}

    def is_singular(self, tol: float = 1e-4) -> bool:
        """Detect a random-effect covariance with a vanishing standard deviation."""
        return any(
            np.min(np.linalg.eigvalsh(covariance)) < tol**2
            for covariance in self.random_covariances
        )

    def ranef(self, cond_var: bool = True) -> dict[str, dict[str, pd.DataFrame]]:
        """Random conditional modes with optional per-level conditional covariance."""
        conditional: dict[str, pd.DataFrame] = {}
        cursor = 0
        for i, block in enumerate(self.random_blocks):
            width = len(block.names)
            count = len(block.levels) * width
            table = pd.DataFrame(
                self.random_modes[cursor : cursor + count].reshape(-1, width),
                index=block.levels,
                columns=block.names,
            )
            if cond_var:
                table.attrs["condVar"] = np.stack(
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
            if key in conditional:
                key = f"{key}.{i}"
            conditional[key] = table
            cursor += count
        return {"cond": conditional, "zi": {}}

    def model_matrix(self, component: str = "cond", newdata: Any = None) -> Array:
        """Evaluate one component model matrix using the training contrasts."""
        specs = {"cond": self.fixed_spec, "zi": self.zi_spec, "disp": self.disp_spec}
        matrices = {"cond": self.X, "zi": self.zi_X, "disp": self.disp_X}
        if component not in specs:
            raise ValueError("component must be cond, zi, or disp")
        if newdata is None:
            return matrices[component].copy()
        frame = as_dataframe(newdata)
        if specs[component] is None:
            return np.zeros((len(frame), 0))
        return np.asarray(specs[component].get_model_matrix(frame), dtype=float)

    def predict(
        self,
        newdata: Any = None,
        type: str = "response",
        re_form: Any = None,
        allow_new_levels: bool = False,
        offset: Any = None,
    ) -> Array:
        """Predict response means, conditional means, zero probability, or dispersion.

        ``re_form='NA'`` gives population predictions with random modes set to
        zero. Response predictions combine conditional means and zero inflation.
        """
        if type not in {"response", "conditional", "cond", "link", "zprob", "zlink", "disp"}:
            raise ValueError("Unsupported prediction type")
        population = re_form is False or (isinstance(re_form, str) and re_form in {"NA", "~0"})
        if isinstance(re_form, float) and np.isnan(re_form):
            population = True
        elif re_form is not None and not population:
            raise NotImplementedError("Only all or no random effects are supported")
        frame = self.data if newdata is None else as_dataframe(newdata)
        if type in {"zprob", "zlink"}:
            if not len(self.zi_beta):
                return np.full(len(frame), -np.inf if type == "zlink" else 0.0)
            zeta = self.model_matrix("zi", frame) @ self.zi_beta
            return zeta if type == "zlink" else np.asarray(special.expit(zeta), dtype=float)
        if type == "disp":
            delta = self.model_matrix("disp", frame) @ self.disp_beta
            return np.exp(delta)
        if newdata is None:
            eta = self.X @ self.beta + self.offset if population else self.linear_predictor.copy()
        else:
            eta = self.model_matrix("cond", frame) @ self.beta
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
                for block in self.random_blocks:
                    values = np.asarray(
                        block.spec.get_model_matrix(frame).iloc[:, block.column_indices],
                        dtype=float,
                    )
                    groups = group_values(block.term.group, frame)
                    width = len(block.names)
                    effects = self.random_modes[
                        cursor : cursor + len(block.levels) * width
                    ].reshape(-1, width)
                    mapping = {level: effects[j] for j, level in enumerate(block.levels)}
                    for j, level in enumerate(groups):
                        if level not in mapping:
                            if not allow_new_levels:
                                raise ValueError(f"New group level: {level}")
                        else:
                            eta[j] += float(values[j] @ mapping[level])
                    cursor += len(block.levels) * width
        if type == "link":
            return eta
        mean = inverse_link(eta, self.link)
        if type == "response" and len(self.zi_beta):
            mean *= 1 - self.predict(frame, type="zprob")
        return mean

    def fitted(self) -> Array:
        """Conditional response means including the zero-inflation component."""
        return self.predict()

    def residuals(self, type: str = "response") -> Array:
        """Response or Pearson residuals using the mixture's total variance."""
        raw = self.y - self.fitted()
        if type == "response":
            return raw
        if type != "pearson":
            raise ValueError("Residual type must be response or pearson")
        mu = self.predict(type="conditional")
        phi = np.exp((2 if self.family == "gaussian" else 1) * (self.disp_X @ self.disp_beta))
        variance = {
            "gaussian": phi,
            "poisson": mu,
            "nbinom1": mu * (1 + phi),
            "nbinom2": mu + mu**2 / phi,
            "binomial": mu * (1 - mu) / self.trials,
            "beta": mu * (1 - mu) / (1 + phi),
        }[self.family]
        probability = self.predict(type="zprob")
        variance = (1 - probability) * variance + probability * (1 - probability) * mu**2
        return raw * np.sqrt(self.weights / np.maximum(variance, 1e-300))

    def summary(self) -> str:
        """Print component-specific Wald tables and model fit criteria."""
        lines = [f" Family: {self.family}  ( {self.link} )", f"Formula: {self.formula}"]
        if len(self.zi_beta):
            lines.append(f"Zero inflation: {self.ziformula}")
        if self.dispformula.replace(" ", "") != "~1":
            lines.append(f"Dispersion: {self.dispformula}")
        lines += ["Data: data"]
        if self.weights_provided:
            lines += ["Weights: weights"]
        lines += [
            "",
            "     AIC      BIC   logLik -2*log(L) df.resid",
            f"{self.AIC():8.1f} {self.BIC():8.1f} {self.logLik():8.1f} {2 * self.objective:8.1f} {self.df_resid:8d}",
        ]
        if self.random_blocks:
            has_correlation = any(len(block.names) > 1 for block in self.random_blocks)
            lines += [
                "",
                "Random effects:",
                "Conditional model:",
                " Groups Name        Variance Std.Dev." + (" Corr" if has_correlation else ""),
            ]
            for block, covariance in zip(self.random_blocks, self.random_covariances):
                sd = np.sqrt(np.maximum(np.diag(covariance), 0))
                for i, name in enumerate(block.names):
                    corr = " ".join(
                        f"{covariance[i, j] / (sd[i] * sd[j]):.2f}" if sd[i] * sd[j] else "NaN"
                        for j in range(i)
                    )
                    lines.append(
                        f" {block.term.group if i == 0 else '':6s} {name:11s} {covariance[i, i]:8.4f} {sd[i]:8.4f} {corr}"
                    )
            if self.family == "gaussian" and len(self.disp_beta) == 1:
                sigma = self.sigma()
                lines += [f" Residual             {sigma**2:8.4f} {sigma:8.4f}"]
        groups = {block.term.group: len(block.levels) for block in self.random_blocks}
        group_description = "; ".join(f"{name}, {count}" for name, count in groups.items())
        lines += [
            f"Number of obs: {self.nobs}" + (f", groups:  {group_description}" if groups else "")
        ]
        if len(self.disp_beta) == 1:
            lines += (
                [f"Dispersion estimate for gaussian family (sigma^2): {self.sigma() ** 2:.3g}"]
                if self.family == "gaussian"
                else [f"Dispersion parameter for {self.family} family (): {self.sigma():.3g}"]
            )
        for component, title in (
            ("cond", "Conditional model:"),
            ("zi", "Zero-inflation model:"),
            ("disp", "Dispersion model:"),
        ):
            effects = self.fixef()[component]
            if not len(effects) or (component == "disp" and len(effects) == 1):
                continue
            diagonal = np.diag(self.vcov(component))
            stderr = np.sqrt(np.where(diagonal > 0, diagonal, np.nan))
            unidentified = np.isin(
                self.fixed_component_indices[component], self.unidentified_fixed_indices
            )
            stderr[unidentified] = np.nan
            statistic = effects.to_numpy() / stderr
            probability = 2 * stats.norm.sf(np.abs(statistic))
            lines += ["", title, "             Estimate Std. Error z value Pr(>|z|)"]
            for name, estimate, se, z, p in zip(
                effects.index, effects, stderr, statistic, probability
            ):
                star = (
                    "***"
                    if p < 0.001
                    else "**"
                    if p < 0.01
                    else "*"
                    if p < 0.05
                    else "."
                    if p < 0.1
                    else ""
                )
                lines.append(f"{name:13s} {estimate:9.5f} {se:10.5f} {z:7.3f} {p:9.5g} {star}")
            if np.any(probability < 0.1):
                lines += ["---", "Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1"]
        if not self.converged:
            lines += ["", "Model convergence warning"]
        if not self.pd_hessian:
            lines += ["", "Non-positive-definite Hessian"]
        return "\n".join(lines)


def glmmTMB(
    formula: str,
    data: Any,
    family: Any = "gaussian",
    *,
    ziformula: str = "~0",
    dispformula: str = "~1",
    link: str | None = None,
    weights: Any = None,
    offset: Any = None,
    contrasts: str = "treatment",
    control: dict[str, Any] | None = None,
) -> GlmmTMBResult:
    """Fit six distribution families with fixed zero inflation and dispersion.

    The conditional random effects are integrated using their observed
    Hessian and the normalized Laplace approximation. Gaussian fits use ML.
    Dense matrix operations are intended for small and medium grouped models.
    """
    if control and set(control) - {"maxiter", "optimizer"}:
        raise ValueError("Unsupported optimizer control")
    family_name, selected_link = _family(family, link)
    frame = as_dataframe(data).reset_index(drop=True)
    parsed = parse_formula(formula)
    prior = _vector(weights, frame, 1)
    offs = _vector(offset, frame, 0)
    explicit_offset = offs.copy()
    for expression in parsed.offsets:
        offs += np.asarray(frame.eval(expression), dtype=float)
    trial = np.ones(len(frame))
    cbind = re.fullmatch(r"cbind\(\s*([^,]+)\s*,\s*([^)]+)\s*\)", parsed.response)
    fixed_formula = parsed.fixed
    if cbind:
        if family_name != "binomial":
            raise ValueError("cbind responses require the binomial family")
        success = np.asarray(frame.eval(cbind.group(1)), dtype=float)
        failure = np.asarray(frame.eval(cbind.group(2)), dtype=float)
        if np.any(success < 0) or np.any(failure < 0):
            raise ValueError("Binomial counts must be nonnegative")
        trial = success + failure
        frame["_rparity_response"] = np.divide(
            success, trial, out=np.zeros_like(success), where=trial > 0
        )
        fixed_formula = "_rparity_response ~ " + parsed.fixed.split("~", 1)[1]
    frame["_rparity_weight"] = prior
    frame["_rparity_offset"] = offs
    frame["_rparity_trial"] = trial
    needed = set(re.findall(r"\b[A-Za-z_]\w*\b", formula + ziformula + dispformula)) & set(
        frame.columns
    )
    needed.update({"_rparity_weight", "_rparity_offset", "_rparity_trial"})
    frame = frame.dropna(subset=sorted(needed)).reset_index(drop=True)
    fixed = build_fixed_design(fixed_formula, frame, contrasts=contrasts)
    frame = fixed.data
    frame["_rparity_response"] = fixed.y
    X, y = fixed.X, fixed.y
    prior = np.asarray(frame["_rparity_weight"], dtype=float)
    offs = np.asarray(frame["_rparity_offset"], dtype=float)
    trial = np.asarray(frame["_rparity_trial"], dtype=float)
    if len(y) <= X.shape[1] or np.linalg.matrix_rank(X) != X.shape[1]:
        raise ValueError(
            "Conditional fixed-effect design must have full rank and residual observations"
        )
    if np.any(prior < 0) or not np.all(np.isfinite(prior)):
        raise ValueError("Weights must be finite and nonnegative")
    if not np.all(np.isfinite(offs)):
        raise ValueError("Offsets must be finite")
    zi = build_fixed_design(
        _component_formula(ziformula, parsed.fixed.split("~", 1)[1]), frame, contrasts
    )
    has_dispersion = family_name not in {"poisson", "binomial"}
    if has_dispersion:
        if dispformula.replace(" ", "") in {"~0", "~0+0", "~1-1"}:
            raise NotImplementedError("Zero dispersion is not supported")
        disp = build_fixed_design(_component_formula(dispformula), frame, contrasts)
        D, disp_names, disp_spec = disp.X, disp.names, disp.spec
    else:
        D, disp_names, disp_spec = np.zeros((len(frame), 0)), [], None
    U = zi.X
    if (U.shape[1] and np.linalg.matrix_rank(U) != U.shape[1]) or (
        D.shape[1] and np.linalg.matrix_rank(D) != D.shape[1]
    ):
        raise ValueError("Zero-inflation and dispersion designs must have full rank")
    if family_name in {"poisson", "nbinom1", "nbinom2"}:
        if np.any(y < 0) or np.any(np.abs(y - np.round(y)) > 1e-7):
            raise ValueError("Count responses must contain nonnegative integers")
    elif family_name == "binomial":
        if np.any((y < 0) | (y > 1)):
            raise ValueError("Binomial response must be between zero and one")
        if not cbind and not np.all((y == 0) | (y == 1)):
            trial = prior.copy()
            prior = np.ones(len(y))
        if np.any(np.abs(trial * y - np.round(trial * y)) > 1e-7):
            warnings.warn("non-integer #successes in a binomial model", UserWarning, stacklevel=2)
    elif family_name == "beta" and (
        np.any((y < 0) | (y >= 1)) or (not U.shape[1] and np.any(y == 0))
    ):
        raise ValueError("Beta response must be in (0,1), or [0,1) with zero inflation")
    blocks = build_random_design(parsed, frame)
    Z = np.column_stack([block.Z for block in blocks]) if blocks else np.zeros((len(y), 0))
    p, pzi, pd = X.shape[1], U.shape[1], D.shape[1]
    theta_count = sum(len(block.names) * (len(block.names) + 1) // 2 for block in blocks)
    theta_start: list[float] = []
    theta_bounds: list[tuple[float | None, float | None]] = []
    for block in blocks:
        for i in range(len(block.names)):
            for j in range(i + 1):
                theta_start.append(0.4 if i == j else 0.0)
                theta_bounds.append((0, None) if i == j else (None, None))
    coefficients = p + pzi + pd
    dispersion_scale = 2.0 if family_name == "gaussian" else 1.0

    def factor(theta: Array) -> tuple[Array, list[Array]]:
        chunks, covariances = [], []
        cursor = 0
        for block in blocks:
            width = len(block.names)
            lower = np.zeros((width, width))
            for i in range(width):
                for j in range(i + 1):
                    lower[i, j] = theta[cursor]
                    cursor += 1
            chunks.append(np.kron(np.eye(len(block.levels)), lower))
            covariances.append(lower @ lower.T)
        return (
            np.asarray(linalg.block_diag(*chunks), dtype=float) if chunks else np.zeros((0, 0))
        ), covariances

    factor_derivatives = [Z @ factor(direction)[0] for direction in np.eye(theta_count)]
    cache_x: Array | None = None
    cache_result: tuple[float, Array, Array, Array, Terms] | None = None

    def evaluate(parameters: Array) -> tuple[float, Array, Array, Array, Terms]:
        nonlocal cache_x, cache_result
        if cache_x is not None and np.array_equal(parameters, cache_x):
            assert cache_result is not None
            return cache_result
        base = X @ parameters[:p] + offs
        zeta = U @ parameters[p : p + pzi] if pzi else None
        delta = dispersion_scale * (D @ parameters[p + pzi : coefficients])
        A = Z @ factor(parameters[coefficients:])[0]
        q = A.shape[1]
        u = np.zeros(q)
        identity = np.eye(q)
        for _ in range(100):
            terms = observation_terms(base + A @ u, delta, zeta, y, trial, prior, family_name)
            gradient = A.T @ terms.score + u
            hessian = (A.T * terms.curvature) @ A + identity
            if not np.all(np.isfinite(gradient)) or not np.all(np.isfinite(hessian)):
                return float("inf"), u, base + A @ u, hessian, terms
            if np.max(np.abs(gradient), initial=0) < 2e-12:
                break
            try:
                chol = linalg.cho_factor(hessian, lower=True, check_finite=False)
                step = linalg.cho_solve(chol, gradient, check_finite=False)
            except linalg.LinAlgError:
                # A scalar ridge can disappear when added to an indefinite
                # matrix with a large spectral range. Invert a positive
                # spectral proposal directly instead. This affects only the
                # descent direction; the likelihood and final observed
                # information still use the unmodified inner Hessian.
                eigenvalues, eigenvectors = linalg.eigh(hessian, check_finite=False)
                floor = max(1e-4, np.max(np.abs(eigenvalues), initial=0) * 1e-12)
                step = eigenvectors @ (
                    (eigenvectors.T @ gradient) / np.maximum(eigenvalues, floor)
                )
            criterion = float(np.sum(terms.value) + u @ u / 2)
            scale = 1.0
            while scale > 2**-30:
                candidate = u - scale * step
                candidate_terms = observation_terms(
                    base + A @ candidate, delta, zeta, y, trial, prior, family_name
                )
                value = float(np.sum(candidate_terms.value) + candidate @ candidate / 2)
                if np.isfinite(value) and value <= criterion - min(
                    1e-4 * scale * float(gradient @ step), 1e-12
                ):
                    break
                candidate_gradient = A.T @ candidate_terms.score + candidate
                if np.max(np.abs(candidate_gradient), initial=0) < 0.9 * np.max(
                    np.abs(gradient), initial=0
                ) and value <= criterion + 1e-10 * max(1, abs(criterion)):
                    # Gamma subtraction near an NB Poisson limit can perturb
                    # function values while the analytic score stays precise.
                    # Accept corrections that reduce that score within the
                    # floating-point function-value envelope.
                    break
                # At roundoff a stationary mode can have equal function values.
                if np.max(np.abs(scale * step), initial=0) < 1e-9 and value <= criterion + 2e-12:
                    break
                scale /= 2
            u = candidate
            if np.max(np.abs(scale * step), initial=0) < 2e-13:
                break
        eta = base + A @ u
        terms = observation_terms(eta, delta, zeta, y, trial, prior, family_name)
        hessian = (A.T * terms.curvature) @ A + identity
        try:
            chol = linalg.cholesky(hessian, lower=True, check_finite=False)
            value = float(np.sum(terms.value) + u @ u / 2 + np.log(np.diag(chol)).sum())
        except linalg.LinAlgError:
            value = float("inf")
        if np.max(np.abs(A.T @ terms.score + u), initial=0) > 1e-6:
            value = float("inf")
        cache_x = parameters.copy()
        cache_result = (value, u, eta, hessian, terms)
        return cache_result

    def objective(parameters: Array) -> float:
        return evaluate(parameters)[0]

    def mode_derivatives(u: Array, A: Array, hessian: Array, terms: Terms) -> tuple[Array, Array]:
        """Direct predictor and implicit standardized-mode Jacobians."""
        B = np.column_stack(
            [X, np.zeros((len(y), pzi + pd))]
            + [derivative @ u for derivative in factor_derivatives]
        )
        if not A.shape[1]:
            return B, np.zeros((0, coefficients + theta_count))
        conditional_score_derivative = terms.curvature[:, None] * B
        conditional_score_derivative[:, p : p + pzi] += terms.score_zi[:, None] * U
        conditional_score_derivative[:, p + pzi : coefficients] += (
            dispersion_scale * terms.score_disp[:, None] * D
        )
        rhs = A.T @ conditional_score_derivative
        for i, derivative in enumerate(factor_derivatives):
            rhs[:, coefficients + i] += derivative.T @ terms.score
        decomposition = linalg.cho_factor(hessian, lower=True, check_finite=False)
        du = -linalg.cho_solve(decomposition, rhs, check_finite=False)
        return B, du

    def gradient(parameters: Array) -> Array:
        value, u, _, hessian, terms = evaluate(parameters)
        if not np.isfinite(value):
            return np.full(len(parameters), np.nan)
        A = Z @ factor(parameters[coefficients:])[0]
        B, du = mode_derivatives(u, A, hessian, terms)
        direct_score = B.T @ terms.score
        direct_score[p : p + pzi] += U.T @ terms.zi_score
        direct_score[p + pzi : coefficients] += dispersion_scale * (D.T @ terms.disp_score)
        if not A.shape[1]:
            return direct_score
        decomposition = linalg.cho_factor(hessian, lower=True, check_finite=False)
        eta_derivative = B + A @ du
        inverse_A = linalg.cho_solve(decomposition, A.T, check_finite=False)
        leverage = np.einsum("ij,ji->i", A, inverse_A)
        direct_score += eta_derivative.T @ (leverage * terms.third) / 2
        direct_score[p : p + pzi] += U.T @ (leverage * terms.curvature_zi) / 2
        direct_score[p + pzi : coefficients] += (
            dispersion_scale * (D.T @ (leverage * terms.curvature_disp)) / 2
        )
        for i, derivative in enumerate(factor_derivatives):
            direct_score[coefficients + i] += np.sum(
                inverse_A.T * (terms.curvature[:, None] * derivative)
            )
        return np.asarray(direct_score, dtype=float)

    if family_name in {"poisson", "nbinom1", "nbinom2"}:
        initial_response = np.log(y + 0.25) - offs
    elif family_name in {"binomial", "beta"}:
        initial_response = special.logit(np.clip(y, 0.05, 0.95)) - offs
    else:
        initial_response = y - offs
    initial_beta = linalg.lstsq(X, initial_response)[0]
    initial_zi = np.zeros(pzi)
    initial_disp = np.zeros(pd)
    if pd:
        value = (
            np.log(max(np.var(y - X @ initial_beta), 0.01)) / 2
            if family_name == "gaussian"
            else np.log(10 if family_name == "beta" else 1)
        )
        initial_disp = linalg.lstsq(D, np.full(len(y), value))[0]
    start = np.r_[initial_beta, initial_zi, initial_disp, theta_start]
    dispersion_bounds = (-100, 100) if family_name in {"nbinom1", "nbinom2"} else (-20, 20)
    bounds = [(None, None)] * (p + pzi) + [dispersion_bounds] * pd + theta_bounds
    maxiter = int((control or {}).get("maxiter", 1000))
    optimizer = str((control or {}).get("optimizer", "L-BFGS-B"))
    if optimizer not in {"L-BFGS-B", "Powell"}:
        raise ValueError("optimizer must be L-BFGS-B or Powell")

    def escape_variance_boundary(current: Any) -> Any:
        # Standard deviations have zero score at zero even at a saddle.
        # Repeat after each round of starts, since a later optimum can also
        # land on such a boundary.
        for _ in range(3):
            restart = np.asarray(current.x, dtype=float).copy()
            restart_value = float(current.fun)
            for i, bound in enumerate(bounds[coefficients:], start=coefficients):
                if bound[0] == 0 and abs(restart[i]) < 1e-5:
                    for interior_step in (0.03, 0.15):
                        candidate = restart.copy()
                        candidate[i] = interior_step
                        candidate_value = objective(candidate)
                        if candidate_value < restart_value - 1e-8:
                            restart, restart_value = candidate, candidate_value
            if restart_value >= current.fun - 1e-8:
                break
            escaped = optimize.minimize(
                objective,
                restart,
                method="L-BFGS-B",
                jac=gradient,
                bounds=bounds,
                options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-8, "maxls": 40},
            )
            if escaped.fun < current.fun:
                current = escaped
            else:
                break
        return current

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        fit = optimize.minimize(
            objective,
            start,
            method=cast(Literal["L-BFGS-B", "Powell"], optimizer),
            jac=gradient if optimizer == "L-BFGS-B" else None,
            bounds=bounds,
            options={
                "maxiter": maxiter,
                "ftol": 1e-14,
                **({"gtol": 1e-8, "maxls": 40} if optimizer == "L-BFGS-B" else {"xtol": 1e-8}),
            },
        )
        if not np.isfinite(fit.fun):
            raise RuntimeError("Unable to find a finite Laplace likelihood")
        if not fit.success:
            retry = optimize.minimize(
                objective,
                fit.x,
                method="Powell",
                bounds=bounds,
                options={"maxiter": maxiter, "ftol": 1e-12, "xtol": 1e-7},
            )
            if np.isfinite(retry.fun) and retry.fun < fit.fun:
                fit = optimize.minimize(
                    objective,
                    retry.x,
                    method="L-BFGS-B",
                    jac=gradient,
                    bounds=bounds,
                    options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-8, "maxls": 40},
                )
        if pzi and np.max(np.abs(fit.x[p : p + pzi])) > 20:
            # A mixture can approach a separated zero-class solution before
            # the conditional component settles. Re-enter from an interior
            # mixture with the fitted conditional/dispersion parameters.
            interior = np.asarray(fit.x, dtype=float).copy()
            interior[p : p + pzi] = initial_zi
            for i, bound in enumerate(bounds[coefficients:], start=coefficients):
                if bound[0] == 0 and interior[i] < 0.1:
                    interior[i] = 0.1
            alternative = optimize.minimize(
                objective,
                interior,
                method="L-BFGS-B",
                jac=gradient,
                bounds=bounds,
                options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-8, "maxls": 40},
            )
            if np.isfinite(alternative.fun) and alternative.fun < fit.fun:
                fit = alternative
        fit = escape_variance_boundary(fit)
        # Cholesky column signs do not change the covariance. An unconstrained
        # score-based polish can cross a zero diagonal to rotate a rank-one
        # factor, avoiding artificial corners from its sign convention.
        polished = optimize.minimize(
            objective,
            fit.x,
            method="BFGS",
            jac=gradient,
            options={"maxiter": maxiter, "gtol": 2e-8},
        )
        if np.isfinite(polished.fun) and polished.fun < fit.fun:
            fit = polished
        current_gradient = gradient(np.asarray(fit.x, dtype=float))
        restart_needed = (
            np.max(np.abs(current_gradient), initial=0) > 1e-4
            or (pd and np.max(np.abs(fit.x[p + pzi : coefficients])) > 15)
            or (pzi and np.max(np.abs(fit.x[p : p + pzi])) > 15)
            or any(
                bound[0] == 0 and abs(fit.x[i]) < 1e-5
                for i, bound in enumerate(bounds[coefficients:], start=coefficients)
            )
        )
        if restart_needed:
            neutral = np.zeros_like(start)
            for i, bound in enumerate(bounds[coefficients:], start=coefficients):
                if bound[0] == 0:
                    neutral[i] = 1
            alternative = optimize.minimize(
                objective,
                neutral,
                method="L-BFGS-B",
                jac=gradient,
                bounds=bounds,
                options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-8, "maxls": 40},
            )
            if np.isfinite(alternative.fun):
                alternative_polish = optimize.minimize(
                    objective,
                    alternative.x,
                    method="BFGS",
                    jac=gradient,
                    options={"maxiter": maxiter, "gtol": 2e-8},
                )
                if np.isfinite(alternative_polish.fun) and alternative_polish.fun < alternative.fun:
                    alternative = alternative_polish
                if alternative.fun < fit.fun:
                    fit = alternative
        if pzi > 1 and np.max(np.abs(fit.x[p : p + pzi])) > 15:
            # Flat zero-class boundaries have multiple local maxima. Explore
            # each standardized predictor direction using deterministic starts,
            # retaining the actual highest normalized Laplace likelihood.
            for j, name in enumerate(zi.names):
                if name == "(Intercept)":
                    continue
                scale = max(float(np.std(U[:, j])), 0.1)
                for sign in (-1, 1):
                    directional = start.copy()
                    directional[p : p + pzi] = 0
                    if "(Intercept)" in zi.names:
                        directional[p + zi.names.index("(Intercept)")] = -4
                    directional[p + j] = sign * 4 / scale
                    alternative = optimize.minimize(
                        objective,
                        directional,
                        method="L-BFGS-B",
                        jac=gradient,
                        bounds=bounds,
                        options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-8, "maxls": 40},
                    )
                    if np.isfinite(alternative.fun):
                        alternative_polish = optimize.minimize(
                            objective,
                            alternative.x,
                            method="BFGS",
                            jac=gradient,
                            options={"maxiter": maxiter, "gtol": 2e-8},
                        )
                        if (
                            np.isfinite(alternative_polish.fun)
                            and alternative_polish.fun < alternative.fun
                        ):
                            alternative = alternative_polish
                        if alternative.fun < fit.fun:
                            fit = alternative
        fit = escape_variance_boundary(fit)
        # Function values can round to equality before the score is stationary.
        # Finish regular optima with Newton score corrections, accepting only
        # reduced scores inside the floating-point likelihood envelope.
        for _ in range(8):
            values = np.asarray(fit.x, dtype=float)
            current_score = gradient(values)
            norm = np.max(np.abs(current_score), initial=0)
            if not np.isfinite(norm) or norm < 2e-10:
                break
            information = score_hessian(gradient, values)
            if not np.all(np.isfinite(information)):
                break
            eigenvalues, eigenvectors = linalg.eigh(information)
            active = eigenvalues > max(1e-8, float(np.max(eigenvalues, initial=0)) * 1e-10)
            if not np.any(active):
                break
            direction = eigenvectors[:, active] @ (
                (eigenvectors[:, active].T @ current_score) / eigenvalues[active]
            )
            scale = 1.0
            accepted = False
            while scale >= 2**-12:
                candidate = values - scale * direction
                candidate_value = objective(candidate)
                candidate_score = gradient(candidate)
                if (
                    np.isfinite(candidate_value)
                    and candidate_value <= fit.fun + 1e-10 * max(1, abs(fit.fun))
                    and np.max(np.abs(candidate_score), initial=0) < norm
                ):
                    fit.x, fit.fun = candidate, candidate_value
                    accepted = True
                    break
                scale /= 2
            if not accepted:
                break
        native = native_coordinates(np.asarray(fit.x, dtype=float), blocks, coefficients)
        covariance_transform = np.eye(len(fit.x))
        if native is None:
            information_parameters = np.asarray(fit.x, dtype=float)
            hessian = centered_score_hessian(gradient, np.asarray(fit.x, dtype=float))
        else:
            natural_parameters, transform = native
            information_parameters = natural_parameters

            def native_gradient(values: Array) -> Array:
                raw, jacobian = transform(values)
                return jacobian.T @ gradient(raw)

            covariance_transform = transform(natural_parameters)[1]
            if np.any(np.all(covariance_transform == 0, axis=0)):
                # At exact rank one, the divergent correlation coordinate
                # has no finite derivative. Resolve the retained tangent
                # more accurately instead of amplifying a coarse score step.
                hessian = richardson_score_hessian(native_gradient, natural_parameters)
            else:
                hessian = centered_score_hessian(native_gradient, natural_parameters)
    # Fixed coefficients can remain identifiable when a variance component is
    # on its boundary. The Moore-Penrose inverse retains that information.
    eigenvalues = np.linalg.eigvalsh(hessian) if np.all(np.isfinite(hessian)) else np.array([-1])
    pd_hessian = bool(np.min(eigenvalues) > 0)
    if np.all(np.isfinite(hessian)):
        if pd_hessian:
            try:
                native_covariance = linalg.cho_solve(
                    linalg.cho_factor(hessian), np.eye(len(hessian))
                )
            except linalg.LinAlgError:
                pd_hessian = False
                native_covariance = linalg.pinvh(hessian, rtol=1e-10)
        else:
            native_covariance = linalg.pinvh(hessian, rtol=1e-10)
        covariance = np.asarray(
            covariance_transform @ native_covariance @ covariance_transform.T, dtype=float
        )
    else:
        covariance = np.full(hessian.shape, np.nan)
    value, u, eta, mode_hessian, final_terms = evaluate(np.asarray(fit.x, dtype=float))
    L, covariances = factor(np.asarray(fit.x[coefficients:], dtype=float))
    random_modes = L @ u
    if len(u):
        random_cov = L @ linalg.solve(mode_hessian, L.T, assume_a="pos")
        _, du = mode_derivatives(u, Z @ L, mode_hessian, final_terms)
        mode_jacobian = L @ du
        for i, direction in enumerate(np.eye(theta_count)):
            mode_jacobian[:, coefficients + i] += factor(direction)[0] @ u
        # The generalized delta method adds outer-parameter estimation
        # uncertainty to the random-mode conditional covariance (TMB sdreport).
        random_cov += mode_jacobian @ covariance @ mode_jacobian.T
        random_cov = (random_cov + random_cov.T) / 2
    else:
        random_cov = np.zeros((0, 0))
    projected_gradient = gradient(np.asarray(fit.x, dtype=float))
    for i, bound in enumerate(bounds):
        if bound[0] == 0 and abs(fit.x[i]) < 1e-7 and projected_gradient[i] > 0:
            projected_gradient[i] = 0
    score_norm = np.max(np.abs(projected_gradient), initial=0)
    converged = bool(score_norm < 1e-5 or (fit.success and score_norm < 1e-4))
    if not converged:
        warnings.warn("Model convergence problem", RuntimeWarning, stacklevel=2)
    if not pd_hessian:
        warnings.warn("Non-positive-definite Hessian", RuntimeWarning, stacklevel=2)
    component_information = np.diag(hessian)[p:coefficients]
    component_parameters = np.asarray(fit.x[p:coefficients], dtype=float)
    unidentified_components = component_information < 1e-8
    if pd_hessian and (
        np.any(unidentified_components & (np.abs(component_parameters) > 10))
        or weak_component_information(hessian, np.asarray(fit.x, dtype=float), p, coefficients)
    ):
        # A positive but effectively flat mixture/dispersion direction can
        # have a finite enormous covariance. This occurs, for example, when
        # only one dispersion stratum approaches its Poisson limit.
        warnings.warn("Component parameters are weakly identified", RuntimeWarning, stacklevel=2)
    if any(np.min(np.linalg.eigvalsh(covariance)) < 1e-8 for covariance in covariances):
        warnings.warn("boundary (singular) fit", RuntimeWarning, stacklevel=2)
    dispersion_values = np.exp(D @ np.asarray(fit.x[p + pzi : coefficients], dtype=float))
    if (family_name == "nbinom1" and np.max(dispersion_values, initial=0) < 1e-6) or (
        family_name == "nbinom2" and np.min(dispersion_values, initial=np.inf) > 1e6
    ):
        warnings.warn("Dispersion is on the Poisson boundary", RuntimeWarning, stacklevel=2)
    return GlmmTMBResult(
        formula=formula,
        data=frame,
        fixed_formula=fixed_formula,
        X=X,
        y=y,
        beta=np.asarray(fit.x[:p], dtype=float),
        cov_beta=covariance[:p, :p],
        coef_names=fixed.names,
        fixed_spec=fixed.spec,
        family=family_name,
        link=selected_link,
        ziformula=ziformula,
        dispformula=dispformula,
        zi_beta=np.asarray(fit.x[p : p + pzi], dtype=float),
        disp_beta=np.asarray(fit.x[p + pzi : coefficients], dtype=float),
        zi_names=zi.names,
        disp_names=disp_names,
        zi_spec=zi.spec,
        disp_spec=disp_spec,
        zi_X=U,
        disp_X=D,
        joint_cov=covariance,
        parameters=np.asarray(fit.x, dtype=float),
        theta=np.asarray(fit.x[coefficients:], dtype=float),
        random_blocks=blocks,
        random_covariances=covariances,
        random_modes=random_modes,
        random_cond_cov=random_cov,
        linear_predictor=eta,
        trials=trial,
        weights=prior,
        offset=offs,
        objective=value,
        terms=fixed.terms,
        converged=converged,
        pd_hessian=pd_hessian,
        outer_gradient=gradient(np.asarray(fit.x, dtype=float)).copy(),
        outer_information=hessian.copy(),
        information_parameters=information_parameters.copy(),
        information_transform=covariance_transform.copy(),
        contrasts=contrasts,
        offset_expressions=parsed.offsets,
        offset_column=offset if isinstance(offset, str) else None,
        offset_requires_new_values=offset is not None
        and not isinstance(offset, str)
        and bool(np.any(explicit_offset != 0)),
        weights_provided=weights is not None,
        unidentified_fixed_indices=(p + np.flatnonzero(unidentified_components)).tolist(),
    )
