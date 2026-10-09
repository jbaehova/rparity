"""Reference-grid linear algebra and simultaneous inference."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ._adapter import ModelAdapter


def adjusted_pvalues(
    statistic: NDArray[np.float64], degrees: NDArray[np.float64],
    method: str, family_size: int,
) -> NDArray[np.float64]:
    """Apply scalar-family corrections, including the studentized range."""
    raw = np.asarray(2 * stats.t.sf(np.abs(statistic), degrees), dtype=float)
    if method in {"none", "identity"} or len(raw) == 0:
        return np.asarray(raw)
    if method == "tukey":
        return np.asarray(stats.studentized_range.sf(
            np.abs(statistic) * np.sqrt(2), max(2, family_size), degrees,
        ))
    names = {"bonferroni": "bonferroni", "holm": "holm", "sidak": "sidak",
             "fdr": "fdr_bh", "BH": "fdr_bh", "fdr_bh": "fdr_bh"}
    if method not in names:
        raise ValueError(f"Unsupported multiple-comparison adjustment: {method}")
    return np.asarray(multipletests(raw, method=names[method])[1])


@dataclass
class EmmGrid:
    """Estimated means represented as linear functions of fitted coefficients.

    Estimates and covariances follow population marginal means as described by
    Searle, Speed and Milliken (1980). Response-scale standard errors use the
    first-order delta method; inference remains on the linear-predictor scale
    until :func:`regrid` is called.
    """

    grid: pd.DataFrame
    linfct: NDArray[np.float64]
    adapter: ModelAdapter
    specs: list[str]
    by: list[str] = field(default_factory=list)
    offsets: NDArray[np.float64] | None = None
    df_method: str = "satterthwaite"
    adjust: str = "none"
    type: str = "link"
    is_contrast: bool = False
    family_size: int = 1
    estimate_name: str = "emmean"
    beta: NDArray[np.float64] | None = None
    cov_beta: NDArray[np.float64] | None = None
    regridded: bool = False
    df_values: NDArray[np.float64] | None = None
    contrast_family: str | None = None

    def __post_init__(self) -> None:
        if self.beta is None:
            self.beta = self.adapter.beta
        if self.cov_beta is None:
            self.cov_beta = self.adapter.covariance
        if self.offsets is None:
            self.offsets = np.zeros(len(self.grid))

    @property
    def estimates(self) -> NDArray[np.float64]:
        """Return estimates on the grid's current linear scale."""
        assert self.beta is not None and self.offsets is not None
        return np.asarray(self.linfct @ self.beta + self.offsets, dtype=float)

    @property
    def covariance(self) -> NDArray[np.float64]:
        """Return the joint covariance of all estimated means."""
        assert self.cov_beta is not None
        return np.asarray(self.linfct @ self.cov_beta @ self.linfct.T, dtype=float)

    @property
    def df(self) -> NDArray[np.float64]:
        """Compute a denominator degree of freedom for each linear function."""
        if self.df_values is not None:
            return self.df_values.copy()
        if (self.adapter.mixed and self.df_method == "asymptotic") or np.isinf(self.adapter.df):
            return np.full(len(self.grid), np.inf)
        if self.df_method == "gls:satterthwaite":
            return np.asarray([self.adapter.model.satterthwaite_df(row) for row in self.linfct])
        if self.adapter.mixed:
            from rparity.inference import contrast_df

            return np.asarray([
                contrast_df(self.adapter.model, row, method=self.df_method)
                for row in self.linfct
            ], dtype=float)
        return np.full(len(self.grid), self.adapter.df)

    def _families(self) -> list[NDArray[np.int64]]:
        if not self.by:
            return [np.arange(len(self.grid))]
        return [np.asarray(index, dtype=np.int64) for index in
                self.grid.groupby(self.by, observed=True, sort=False, dropna=False).indices.values()]

    def summary(
        self, *, type: str | None = None, level: float = 0.95,
        adjust: str | None = None, infer: bool | tuple[bool, bool] | None = None,
        null: float = 0.0, side: str = "two-sided",
    ) -> pd.DataFrame:
        """Return estimates, intervals and optional tests in an R-style table."""
        if not 0 < level < 1:
            raise ValueError("level must lie strictly between zero and one")
        if side != "two-sided":
            raise ValueError("Only two-sided inference is supported")
        if hasattr(self.adapter.model, "model"):
            family = getattr(self.adapter.model.model, "family", None)
            family_name = "" if family is None else family.__class__.__name__.lower()
            fitted = np.asarray(getattr(self.adapter.model, "fittedvalues", []))
            boundary = family_name in {"binomial", "poisson"} and bool(np.any(fitted < 1e-10))
            boundary = boundary or (family_name == "binomial" and bool(np.any(fitted > 1 - 1e-10)))
            if boundary:
                warnings.warn(
                    "Boundary GLM fit: fitted probabilities or rates approach zero or one; "
                    "marginal means involving a non-finite coefficient are unstable",
                    RuntimeWarning, stacklevel=2,
                )
        scale = self.type if type is None else type
        if scale not in {"link", "response"}:
            raise ValueError("type must be 'link' or 'response'")
        adjustment = self.adjust if adjust is None else adjust
        if adjustment == "tukey" and self.contrast_family not in {"pairwise", "revpairwise", "tukey"}:
            warnings.warn(
                "adjust='tukey' changed to 'sidak': Tukey adjustment requires one family of pairwise comparisons",
                UserWarning, stacklevel=2,
            )
            adjustment = "sidak"
        intervals, tests = (not self.is_contrast, self.is_contrast)
        if isinstance(infer, tuple):
            intervals, tests = infer
        elif infer is not None:
            intervals = tests = infer
        estimate = self.estimates
        se = np.sqrt(np.maximum(np.diag(self.covariance), 0))
        degrees = self.df
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (estimate - null) / se
        probability = np.full(len(estimate), np.nan)
        quantile = np.asarray(stats.t.ppf(1 - (1 - level) / 2, degrees), dtype=float)
        for index in self._families():
            probability[index] = adjusted_pvalues(
                ratio[index], degrees[index], adjustment, self.family_size,
            )
            if adjustment == "tukey":
                quantile[index] = stats.studentized_range.ppf(
                    level, max(2, self.family_size), degrees[index],
                ) / np.sqrt(2)
            elif adjustment in {"bonferroni", "holm", "fdr", "fdr_bh", "BH"}:
                quantile[index] = stats.t.ppf(1 - (1 - level) / (2 * len(index)), degrees[index])
            elif adjustment == "sidak":
                quantile[index] = stats.t.ppf((1 + level ** (1 / len(index))) / 2, degrees[index])
        lower, upper = estimate - quantile * se, estimate + quantile * se
        name = self.estimate_name
        displayed_null = null
        if scale == "response" and not self.regridded:
            if self.is_contrast and self.adapter.link in {"log", "log10", "log2", "logit", "logistic"}:
                name = "odds.ratio" if self.adapter.link in {"logit", "logistic"} else "ratio"
                base = {"log10": np.log(10.0), "log2": np.log(2.0)}.get(self.adapter.link, 1.0)
                se = base * np.exp(base * estimate) * se
                estimate, lower, upper = np.exp(base * estimate), np.exp(base * lower), np.exp(base * upper)
                displayed_null = float(np.exp(base * null))
            elif not self.is_contrast:
                se = np.abs(self.adapter.derivative(estimate)) * se
                estimate = self.adapter.inverse(estimate)
                lo, hi = self.adapter.inverse(lower), self.adapter.inverse(upper)
                lower, upper = np.minimum(lo, hi), np.maximum(lo, hi)
                if self.adapter.link in {"logit", "logistic", "probit", "cloglog"}:
                    name = "prob"
                elif self.adapter.link != "identity":
                    family = getattr(self.adapter.model, "family", None)
                    if family is None and hasattr(self.adapter.model, "model"):
                        family = getattr(self.adapter.model.model, "family", None)
                    family_name = family if isinstance(family, str) else family.__class__.__name__
                    name = "rate" if str(family_name).lower() == "poisson" else "response"
                if tests:
                    displayed_null = float(self.adapter.inverse(np.asarray([null]))[0])
        output = self.grid.copy()
        output[name] = estimate
        output["SE"] = se
        output["df"] = degrees
        if intervals:
            asymptotic = np.all(np.isinf(degrees))
            output["asymp.LCL" if asymptotic else "lower.CL"] = lower
            output["asymp.UCL" if asymptotic else "upper.CL"] = upper
        if tests:
            if displayed_null != 0:
                output["null"] = displayed_null
            output["z.ratio" if np.all(np.isinf(degrees)) else "t.ratio"] = ratio
            output["p.value"] = probability
        if not self.regridded:
            projection = np.linalg.pinv(self.adapter.design) @ self.adapter.design
            estimable = np.linalg.norm(self.linfct @ (np.eye(len(projection)) - projection), axis=1)
            bad = estimable > 1e-7 * np.maximum(1, np.linalg.norm(self.linfct, axis=1))
            output.loc[bad, output.columns[len(self.grid.columns):]] = np.nan
        output.attrs.update({"adjust": adjustment, "type": scale, "df_method": self.df_method})
        return output

    def pairs(self, *, reverse: bool = False, **kwargs: Any) -> EmmGrid:
        """Compare every pair within each by group."""
        from ._operations import contrast

        return contrast(self, "revpairwise" if reverse else "pairwise", **kwargs)

    def contrast(self, method: Any = "pairwise", **kwargs: Any) -> EmmGrid:
        """Construct a named family or caller-supplied linear contrasts."""
        from ._operations import contrast

        return contrast(self, method, **kwargs)

    def regrid(self, *, transform: str = "response") -> EmmGrid:
        """Apply the inverse link before subsequent averaging or contrasts."""
        from ._operations import regrid

        return regrid(self, transform=transform)

    def confint(self, *, level: float = 0.95, **kwargs: Any) -> pd.DataFrame:
        """Return confidence intervals for all grid rows."""
        return self.summary(level=level, infer=(True, False), **kwargs)

    def test(self, *, null: float = 0.0, **kwargs: Any) -> pd.DataFrame:
        """Test every estimate against a specified linear-scale null."""
        return self.summary(null=null, infer=(False, True), **kwargs)

    def to_dataframe(self, **kwargs: Any) -> pd.DataFrame:
        """Materialize the default summary table."""
        return self.summary(**kwargs)

    def __getitem__(self, key: Any) -> Any:
        return self.summary()[key]

    def __len__(self) -> int:
        return len(self.grid)

    def __repr__(self) -> str:
        return self.summary().to_string(index=False)
