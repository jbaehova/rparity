"""Marginal means, contrasts, trends and joint linear tests."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from fractions import Fraction
from functools import reduce
from itertools import combinations, product
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from ._adapter import adapt
from ._grid import EmmGrid


def _names(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[~*+:]", value) if part.strip() != "1" and part.strip()]
    return list(value)


def _specs(specs: str | Sequence[str], by: str | Sequence[str] | None) -> tuple[list[str], list[str], str | None]:
    method = None
    if isinstance(specs, str):
        if "|" in specs:
            specs, groups = specs.split("|", 1)
            if by is None:
                by = groups
        if "~" in specs:
            lhs, specs = specs.split("~", 1)
            method = lhs.strip() or None
    return _names(specs), _names(by), method


def _cartesian(levels: dict[str, list[Any]]) -> pd.DataFrame:
    names = list(levels)
    if not names:
        return pd.DataFrame(index=[0])
    rows = [tuple(reversed(row)) for row in product(*(levels[name] for name in reversed(names)))]
    return pd.DataFrame(rows, columns=names)


def ref_grid(
    model: Any, *, at: Mapping[str, Any] | None = None, data: Any = None,
    cov_reduce: Callable[[Any], Any] = np.mean, cov_keep: int | Sequence[str] = 2,
    lmer_df: str = "kenward-roger", df: float | None = None,
    type: str = "link", regrid: str | None = None, mode: str = "auto",
    component: str = "cond",
) -> EmmGrid:
    """Construct a full factorial prediction grid using original contrast coding.

    Numeric covariates are reduced to their means unless specified in ``at`` or
    retained by ``cov_keep``. Factor levels retain categorical ordering. This
    implements the public reference-grid specification of Lenth's emmeans.
    """
    if lmer_df not in {"kenward-roger", "satterthwaite", "asymptotic"}:
        raise ValueError("Unknown mixed-model degree-of-freedom method")
    adapter = adapt(model, data, component=component)
    df_method = lmer_df
    if hasattr(model, "_covariance"):
        if mode not in {"auto", "df.error", "satterthwaite", "asymptotic"}:
            raise ValueError("Unknown GLS degree-of-freedom mode")
        if mode == "auto":
            mode = "satterthwaite" if len(model._theta) else "df.error"
        if mode == "df.error":
            adapter.df -= len(model._theta)
        elif mode == "asymptotic":
            adapter.df = np.inf
        else:
            df_method = "gls:satterthwaite"
    levels: dict[str, list[Any]] = {}
    factors: list[str] = []
    at = {} if at is None else dict(at)
    unknown = set(at) - set(adapter.variables)
    if unknown:
        raise ValueError(f"at contains non-predictors: {sorted(unknown)}")
    for name in adapter.variables:
        column = adapter.data[name].dropna()
        if isinstance(column.dtype, pd.CategoricalDtype):
            values = list(column.cat.categories)
            is_factor = True
        elif not pd.api.types.is_numeric_dtype(column):
            values = sorted(column.unique().tolist(), key=str)
            is_factor = True
        else:
            values = sorted(column.unique().tolist())
            is_factor = (len(values) <= cov_keep if isinstance(cov_keep, int) else name in cov_keep)
        if is_factor:
            factors.append(name)
        if name in at:
            value = at[name]
            levels[name] = [value] if np.isscalar(value) else list(value)
        elif is_factor:
            levels[name] = values
        else:
            value = cov_reduce(column)
            levels[name] = [value] if np.isscalar(value) else list(value)
    frame = _cartesian(levels)
    matrix = adapter.matrix(frame)
    covariance = adapter.covariance
    if adapter.mixed and np.isfinite(adapter.df) and lmer_df == "kenward-roger":
        from rparity.inference import adjusted_covariance

        covariance = np.asarray(adjusted_covariance(model))
    output = EmmGrid(
        frame, matrix, adapter, list(levels), offsets=adapter.offsets(frame),
        df_method=df_method, type=type, cov_beta=covariance,
        df_values=None if df is None else np.full(len(frame), df),
    )
    output.grid.attrs["factors"] = factors
    if regrid is not None:
        output = globals()["regrid"](output, transform=regrid)
    return output


def _weights(grid: EmmGrid, sub: pd.DataFrame, keys: list[str], weights: str | ArrayLike) -> NDArray[np.float64]:
    if not isinstance(weights, str):
        result = np.asarray(weights, dtype=float)
        if result.ndim != 1 or len(result) != len(sub):
            raise ValueError("weights must have one value per reference-grid row being averaged")
    elif weights == "equal":
        result = np.ones(len(sub))
    elif weights in {"proportional", "outer", "cells", "flat"}:
        factors = grid.grid.attrs.get("factors", [])
        averaged = [name for name in factors if name not in keys and name in sub]
        count_names = [name for name in factors if name in sub] if weights in {"cells", "flat"} else averaged
        if not count_names:
            result = np.ones(len(sub))
        elif weights == "outer":
            result = np.ones(len(sub))
            for name in count_names:
                counts = grid.adapter.data[name].value_counts(dropna=False)
                result *= sub[name].map(counts).fillna(0).to_numpy(dtype=float)
        else:
            counts = grid.adapter.data.groupby(count_names, observed=True, dropna=False).size()
            result = np.array([
                float(counts.get(tuple(row) if len(count_names) > 1 else row[0], 0))
                for row in sub[count_names].itertuples(index=False, name=None)
            ])
            if weights == "flat":
                result = (result > 0).astype(float)
    else:
        raise ValueError(f"Unknown marginal-mean weighting method: {weights}")
    if not np.all(np.isfinite(result)) or np.any(result < 0) or result.sum() <= 0:
        raise ValueError("weights must be finite, nonnegative and have positive total")
    return result / result.sum()


def emmeans(
    model: Any, specs: str | Sequence[str], *, by: str | Sequence[str] | None = None,
    at: Mapping[str, Any] | None = None, weights: str | ArrayLike = "equal",
    type: str = "link", pairwise: bool = False, adjust: str | None = None,
    lmer_df: str = "kenward-roger", **kwargs: Any,
) -> EmmGrid | dict[str, EmmGrid]:
    """Estimate marginal means by averaging fixed-effect predictions.

    The linear functions and joint covariance implement population marginal
    means (Searle, Speed and Milliken, 1980). ``type='response'`` back-transforms
    the completed means; ``regrid='response'`` transforms grid cells first.
    """
    primary, groups, method = _specs(specs, by)
    grid = model if isinstance(model, EmmGrid) else ref_grid(
        model, at=at, lmer_df=lmer_df, type=type, **kwargs,
    )
    assert grid.offsets is not None
    keys = list(dict.fromkeys(primary + groups))
    unknown = set(keys) - set(grid.grid.columns)
    if unknown:
        raise ValueError(f"Unknown reference-grid predictors: {sorted(unknown)}")
    if keys:
        grouped = grid.grid.groupby(keys, sort=False, observed=True, dropna=False)
        indices = []
        for values in grid.grid[keys].drop_duplicates().itertuples(index=False, name=None):
            key = values[0] if len(keys) == 1 else values
            indices.append(np.asarray(grouped.indices[key], dtype=int))
    else:
        indices = [np.arange(len(grid))]
    rows: list[dict[str, Any]] = []
    matrices: list[NDArray[np.float64]] = []
    offsets: list[float] = []
    degrees: list[float] = []
    for index in indices:
        sub = grid.grid.iloc[index]
        weight = _weights(grid, sub, keys, weights)
        rows.append({name: sub.iloc[0][name] for name in keys})
        matrices.append(weight @ grid.linfct[index])
        offsets.append(float(weight @ grid.offsets[index]))
        if grid.df_values is not None:
            degrees.append(float(np.min(grid.df_values[index])))
    frame = pd.DataFrame(rows, columns=keys)
    frame.attrs = grid.grid.attrs.copy()
    output = replace(
        grid, grid=frame, linfct=np.asarray(matrices), offsets=np.asarray(offsets),
        specs=primary, by=groups, type=type, family_size=len(rows),
        adjust="none" if adjust is None else adjust,
        df_values=np.asarray(degrees) if degrees else None,
    )
    if pairwise or method:
        comparisons = contrast(output, method or "pairwise", adjust=adjust)
        return {"emmeans": output, "contrasts": comparisons}
    return output


def _polynomials(k: int, degree: int) -> NDArray[np.float64]:
    # Rational Gram-Schmidt yields the conventional smallest integer columns,
    # rather than unit-length polynomials.
    basis: list[list[Fraction]] = []
    for exponent in range(degree + 1):
        vector = [Fraction(x**exponent) for x in range(1, k + 1)]
        for previous in basis:
            scale = sum((a * b for a, b in zip(vector, previous, strict=True)), Fraction(0)) / sum((a * a for a in previous), Fraction(0))
            vector = [a - scale * b for a, b in zip(vector, previous, strict=True)]
        basis.append(vector)
    result = []
    for vector in basis[1:]:
        denominator = math.lcm(*(x.denominator for x in vector))
        integers = [int(x * denominator) for x in vector]
        divisor = reduce(math.gcd, integers)
        result.append(np.asarray(integers, dtype=float) / abs(divisor))
    return np.asarray(result)


def _contrasts(
    method: str | Mapping[str, ArrayLike], labels: list[str], *, ref: int | Sequence[int],
    max_degree: int | None,
) -> tuple[list[str], NDArray[np.float64], str]:
    k = len(labels)
    identity = np.eye(k)
    if isinstance(method, Mapping):
        names = list(method)
        matrix = np.asarray(list(method.values()), dtype=float)
        if matrix.shape != (len(names), k):
            raise ValueError("Each custom contrast must have one coefficient per estimated mean")
        return names, matrix, "none"
    if method in {"pairwise", "revpairwise", "tukey"}:
        pairs = list(combinations(range(k), 2))
        reverse = method == "revpairwise"
        rows = [identity[j] - identity[i] if reverse else identity[i] - identity[j] for i, j in pairs]
        names = [f"{labels[j]} - {labels[i]}" if reverse else f"{labels[i]} - {labels[j]}" for i, j in pairs]
        return names, np.asarray(rows).reshape(-1, k), "tukey"
    if method in {"trt.vs.ctrl", "trt.vs.ctrl1", "trt.vs.ctrlk"}:
        if method == "trt.vs.ctrlk":
            ref = k
        references = [ref] if isinstance(ref, int) else list(ref)
        references = [i - 1 for i in references]
        if not references or any(i < 0 or i >= k for i in references):
            raise ValueError("ref uses one-based level indices within each by group")
        baseline = identity[references].mean(axis=0)
        control = labels[references[0]] if len(references) == 1 else "avg(" + ",".join(labels[i] for i in references) + ")"
        indices = [i for i in range(k) if i not in references]
        return [f"{labels[i]} - {control}" for i in indices], np.asarray([identity[i] - baseline for i in indices]).reshape(-1, k), "none"
    if method == "consec":
        return [f"{labels[i + 1]} - {labels[i]}" for i in range(k - 1)], np.diff(identity, axis=0), "none"
    if method == "poly":
        degree = min(6, k - 1) if max_degree is None else min(max_degree, k - 1)
        names = ["linear", "quadratic", "cubic", "quartic", "degree 5", "degree 6"]
        return names[:degree], _polynomials(k, degree), "none"
    raise ValueError(f"Unknown contrast family: {method}")


def contrast(
    object: EmmGrid, method: str | Mapping[str, ArrayLike] = "pairwise", *,
    by: str | Sequence[str] | None = None, adjust: str | None = None,
    ref: int | Sequence[int] = 1, max_degree: int | None = None,
) -> EmmGrid:
    """Construct pairwise, control, consecutive, polynomial or custom contrasts.

    ``ref`` uses the one-based convention of R emmeans. Polynomial coefficients
    are orthogonal, integer-scaled polynomials on equally spaced factor levels.
    """
    groups = object.by if by is None else _names(by)
    assert object.offsets is not None
    indices = [np.arange(len(object))] if not groups else [
        np.asarray(index, dtype=int) for index in
        object.grid.groupby(groups, sort=False, observed=True, dropna=False).indices.values()
    ]
    rows: list[dict[str, Any]] = []
    matrices: list[NDArray[np.float64]] = []
    offsets: list[float] = []
    degrees: list[float] = []
    default = "none"
    family_size = 1
    for index in indices:
        sub = object.grid.iloc[index]
        label_names = [name for name in object.specs if name not in groups]
        labels = [" ".join(str(row[name]) for name in label_names) for _, row in sub.iterrows()]
        labels = labels or [str(i + 1) for i in range(len(sub))]
        names, matrix, default = _contrasts(method, labels, ref=ref, max_degree=max_degree)
        family_size = max(family_size, len(sub))
        for name, coefficient in zip(names, matrix, strict=True):
            rows.append({**{key: sub.iloc[0][key] for key in groups}, "contrast": name})
            matrices.append(coefficient @ object.linfct[index])
            offsets.append(float(coefficient @ object.offsets[index]))
            if object.df_values is not None:
                support = index[np.abs(coefficient) > 1e-12]
                degrees.append(float(np.min(object.df_values[support])) if len(support) else np.inf)
    return replace(
        object, grid=pd.DataFrame(rows, columns=groups + ["contrast"]),
        linfct=np.asarray(matrices).reshape(-1, object.linfct.shape[1]),
        offsets=np.asarray(offsets), specs=["contrast"], by=groups,
        adjust=default if adjust is None else adjust, is_contrast=True,
        family_size=family_size, estimate_name="estimate",
        df_values=np.asarray(degrees) if degrees else None,
        contrast_family=method if isinstance(method, str) else "custom",
    )


def pairs(object: EmmGrid, *, reverse: bool = False, **kwargs: Any) -> EmmGrid:
    """Compare all pairs using a studentized-range adjustment by default."""
    return contrast(object, "revpairwise" if reverse else "pairwise", **kwargs)


def regrid(object: EmmGrid, *, transform: str = "response") -> EmmGrid:
    """Make transformed predictions the new linear parameters via the delta method."""
    if transform not in {"response", "none"}:
        raise ValueError("transform must be 'response' or 'none'")
    estimates = object.estimates
    covariance = object.covariance
    name = object.estimate_name
    if transform == "response" and not object.regridded:
        derivative = object.adapter.derivative(estimates)
        covariance = derivative[:, None] * covariance * derivative[None, :]
        estimates = object.adapter.inverse(estimates)
        if object.adapter.link in {"logit", "logistic", "probit", "cloglog"}:
            name = "prob"
        elif object.adapter.link != "identity":
            family = getattr(object.adapter.model, "family", None)
            if family is None and hasattr(object.adapter.model, "model"):
                family = getattr(object.adapter.model.model, "family", None)
            family_name = family if isinstance(family, str) else family.__class__.__name__
            name = "rate" if str(family_name).lower() == "poisson" else "response"
    return replace(
        object, linfct=np.eye(len(object)), beta=estimates, cov_beta=covariance,
        offsets=np.zeros(len(object)), regridded=True, type="response",
        df_values=object.df, estimate_name=name,
    )


def emtrends(
    model: Any, specs: str | Sequence[str], *, var: str,
    delta_var: float | None = None, at: Mapping[str, Any] | None = None,
    by: str | Sequence[str] | None = None, weights: str | ArrayLike = "equal",
    type: str = "link", lmer_df: str = "kenward-roger", **kwargs: Any,
) -> EmmGrid | dict[str, EmmGrid]:
    """Estimate covariate trends by the documented forward-difference construction.

    The default step is one thousandth of the observed covariate range, matching
    the public emtrends specification. Trends for nonlinear terms therefore
    refer to this finite difference, rather than an exact symbolic derivative.
    """
    grid = ref_grid(model, at=at, lmer_df=lmer_df, type=type, **kwargs)
    assert grid.offsets is not None
    if var not in grid.grid:
        raise ValueError(f"{var!r} is not a model predictor")
    values = np.asarray(grid.adapter.data[var], dtype=float)
    step = delta_var if delta_var is not None else 0.001 * float(np.ptp(values))
    if step == 0:
        raise ValueError("delta_var must be nonzero; constant covariates need an explicit step")
    upper = grid.grid.copy()
    upper[var] = upper[var] + step
    matrix = (grid.adapter.matrix(upper) - grid.linfct) / step
    offsets = (grid.adapter.offsets(upper) - grid.offsets) / step
    trend = replace(grid, linfct=matrix, offsets=offsets, estimate_name=f"{var}.trend")
    output = emmeans(trend, specs, by=by, weights=weights, type=type)
    return output


def joint_tests(
    model: Any, *, by: str | Sequence[str] | None = None,
    at: Mapping[str, Any] | None = None, weights: str | ArrayLike = "equal",
    lmer_df: str = "kenward-roger", **kwargs: Any,
) -> pd.DataFrame:
    """Test estimable reference-grid term contrasts jointly.

    Factor tests average over other predictors. Continuous predictors use two
    points symmetric about their means, providing estimable slope contrasts.
    Wald quadratic forms are converted to F tests when denominator df are finite.
    """
    adapter = model.adapter if isinstance(model, EmmGrid) else adapt(model)
    points = {} if at is None else dict(at)
    for name in adapter.variables:
        column = adapter.data[name]
        if pd.api.types.is_numeric_dtype(column) and column.nunique() > 2 and name not in points:
            mean = float(column.mean())
            points[name] = [mean - 1, mean + 1]
    grid = model if isinstance(model, EmmGrid) else ref_grid(model, at=points, lmer_df=lmer_df, **kwargs)
    if hasattr(adapter.model, "terms"):
        term_labels = list(adapter.model.terms)
    elif hasattr(adapter.model, "fixed_spec"):
        term_labels = list(adapter.model.fixed_spec.terms)
    else:
        info = getattr(adapter.model.model.data, "design_info", None)
        if info is None:
            info = getattr(adapter.model.model.data, "model_spec", None)
        term_labels = list(getattr(info, "term_names", getattr(info, "terms", adapter.variables)))
    terms = []
    for label in term_labels:
        variables = [name for name in adapter.variables
                     if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", str(label))]
        if variables:
            term = ":".join(variables)
            if term not in terms:
                terms.append(term)
    groups = _names(by)
    rows = []
    for term in terms:
        term_names = term.split(":")
        if any(name in groups or name not in grid.grid for name in term_names):
            continue
        marginal = emmeans(grid, term_names, by=groups, weights=weights)
        assert isinstance(marginal, EmmGrid)
        for index in marginal._families():
            sub = marginal.grid.iloc[index]
            if len(index) < 2:
                continue
            if len(term_names) == 1:
                coefficient = np.eye(len(index))[1:] - np.eye(len(index))[0]
            else:
                # Build by labels because pandas grouping may reorder cells.
                differences = []
                for name in term_names:
                    levels = list(pd.unique(sub[name]))
                    codes = np.array([levels.index(value) for value in sub[name]])
                    difference = np.eye(len(levels))[1:] - np.eye(len(levels))[0]
                    differences.append(difference[:, codes])
                coefficient = np.asarray([
                    np.prod(np.asarray(rows), axis=0)
                    for rows in product(*differences)
                ])
            matrix = coefficient @ marginal.linfct[index]
            if np.max(np.abs(matrix)) < 1e-10:
                continue
            covariance = matrix @ marginal.cov_beta @ matrix.T
            rank = int(np.linalg.matrix_rank(covariance))
            if not rank:
                continue
            estimate = coefficient @ marginal.estimates[index]
            statistic = float(estimate @ np.linalg.pinv(covariance) @ estimate / rank)
            if marginal.df_values is not None:
                denominator = float(np.min(marginal.df_values[index]))
            else:
                hypotheses = replace(
                    marginal, grid=pd.DataFrame(index=range(len(matrix))), linfct=matrix,
                    offsets=np.zeros(len(matrix)), df_values=None,
                )
                denominator = float(np.min(hypotheses.df))
            pvalue = float(stats.chi2.sf(statistic * rank, rank) if np.isinf(denominator)
                           else stats.f.sf(statistic, rank, denominator))
            rows.append({**{key: sub.iloc[0][key] for key in groups}, "model term": term,
                         "df1": rank, "df2": round(denominator, 2), "F.ratio": round(statistic, 3), "p.value": pvalue})
    output = pd.DataFrame(rows)
    if len(output) and np.all(np.isinf(output["df2"])):
        output.insert(len(output.columns) - 1, "Chisq", output["F.ratio"] * output["df1"])
    return output
