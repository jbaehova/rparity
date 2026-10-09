"""Term-wise Wald, small-sample F and likelihood-ratio model comparisons."""
from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import linalg, stats

from rparity.inference import adjusted_covariance, contrast_test

Array = NDArray[np.float64]


def _type(value: int | str) -> int:
    result = {"I": 1, "II": 2, "III": 3}.get(str(value).upper(), value)
    if result not in (1, 2, 3):
        raise ValueError("ANOVA type must be I, II or III")
    return int(result)


def _terms(model: Any) -> dict[str, list[int]]:
    spec = getattr(model, "fixed_spec", None)
    if spec is not None:
        return {"Intercept" if str(k) == "1" else str(k): list(v)
                for k, v in spec.term_indices.items()}
    design = getattr(model.model.data, "design_info", None)
    if design is None:
        design = getattr(model.model.data, "model_spec", None)
    if design is not None:
        return {name: list(range(sl.start, sl.stop))
                for name, sl in design.term_name_slices.items()}
    names = model.model.exog_names
    return {str(name): [i] for i, name in enumerate(names)}


def _relatives(term: str, terms: dict[str, list[int]]) -> list[str]:
    factors = set(term.split(":"))
    return [t for t in terms if factors < set(t.split(":"))]


def _hypothesis(term: str, terms: dict[str, list[int]], kind: int,
                x: Array, covariance: Array, v: Array | None = None,
                relatives: list[str] | None = None) -> Array:
    identity = np.eye(x.shape[1])
    cols = terms[term]
    if kind == 3 or term == "Intercept":
        return identity[cols]
    if kind == 2:
        higher_terms = _relatives(term, terms) if relatives is None else relatives
        higher = [j for t in higher_terms for j in terms[t]]
        l1 = identity[cols + higher]
        if not higher:
            return l1
        # Remove hypotheses about containing interactions in the covariance metric.
        l2 = identity[higher]
        complement = linalg.null_space(l2 @ covariance @ l1.T)
        return complement.T @ l1
    before: list[int] = []
    for name, indices in terms.items():
        if name == term:
            break
        before.extend(indices)
    white_x = x
    if v is not None:
        chol = linalg.cholesky(v, lower=True)
        white_x = linalg.solve_triangular(chol, x, lower=True)
    current = white_x[:, cols].copy()
    if before:
        qb = linalg.orth(white_x[:, before])
        current -= qb @ (qb.T @ current)
    q = linalg.orth(current)
    return q.T @ white_x


def _sas_sequential(x: Array, indices: list[int], order: list[int]) -> Array:
    """Forward-Doolittle rows via equivalent QR, normalized by their pivots."""
    _, r = linalg.qr(x[:, order], mode="economic")
    positions = [order.index(i) for i in indices]
    rows = r[positions].copy()
    rows /= np.diag(r)[positions, None]
    result = np.zeros((len(positions), x.shape[1]))
    result[:, order] = rows
    return result


def _marginal_hypotheses(model: Any) -> dict[str, Array]:
    """Express SAS Type III effects in the fitted coefficient coordinates."""
    from formulaic import model_matrix

    spec = model.fixed_spec
    categorical = {key for key, (kind, _) in spec.encoder_state.items()
                   if kind.value == "categorical"}
    mapped = {}
    expressions = []
    for term in spec.terms:
        factors = [f"C({factor.expr}, Sum)" if factor.expr in categorical else factor.expr
                   for factor in term.factors]
        expression = ":".join(factors)
        expressions.append(expression)
        name = "Intercept" if str(term) == "1" else str(term)
        mapped[name] = frozenset(factors)
    formula = "0 + " + " + ".join(expressions)
    matrix = model_matrix(formula, model.data)
    alternative = np.asarray(matrix, dtype=float)
    transform = linalg.pinv(alternative) @ np.asarray(model.X, dtype=float)
    column_indices = {frozenset(f.expr for f in term.factors): list(indices)
                      for term, indices in matrix.model_spec.term_indices.items()}
    hypotheses = {}
    for name, factor_set in mapped.items():
        multiplier = np.ones((1, 1))
        original_term = next(term for term in spec.terms
                             if ("Intercept" if str(term) == "1" else str(term)) == name)
        for factor in original_term.factors:
            if factor.expr in categorical:
                expression = f"C({factor.expr}, Sum)"
                state = matrix.model_spec.encoder_state[expression][1]
                coding = np.asarray(state["contrasts"].get_coding_matrix(reduced_rank=True))
                difference = coding[1:] - coding[0]
                multiplier = np.kron(difference, multiplier)
        rows = transform[column_indices[factor_set]]
        hypotheses[name] = multiplier @ rows if multiplier.shape[1] == len(rows) else rows
    return hypotheses


def _glm_refit(model: Any, columns: list[int]) -> Any:
    import statsmodels.api as sm

    original = model.model
    kwargs: dict[str, Any] = {"family": original.family}
    for field in ("offset", "exposure", "freq_weights", "var_weights"):
        value = getattr(original, field, None)
        if value is not None:
            # statsmodels stores exposure on the log scale.
            kwargs[field] = np.exp(value) if field == "exposure" else value
    y = np.asarray(original.endog)
    if y.ndim == 1 and hasattr(original, "n_trials"):
        n_trials = np.asarray(original.n_trials)
        if np.any(n_trials != 1):
            y = np.column_stack((y * n_trials, (1 - y) * n_trials))
    return sm.GLM(y, np.asarray(original.exog)[:, columns], **kwargs).fit()


def _dispersion(model: Any, estimate: str) -> float:
    if estimate == "pearson":
        return float(model.pearson_chi2 / model.df_resid)
    if estimate == "deviance":
        return float(model.deviance / model.df_resid)
    if estimate != "dispersion":
        raise ValueError("error_estimate must be pearson, deviance or dispersion")
    if model.model.family.__class__.__name__ in {"Poisson", "Binomial"}:
        warnings.warn("Fixed dispersion replaced by Pearson dispersion for F test",
                      UserWarning, stacklevel=3)
        return float(model.pearson_chi2 / model.df_resid)
    return float(model.scale)


def Anova(model: Any, type: int | str = 2, test_statistic: str | None = None,
          error_estimate: str = "pearson", component: str = "cond") -> pd.DataFrame:
    """Return car-style Type II/III tests for mixed models and statsmodels fits.

    Type III hypotheses depend on factor coding. Use sum contrasts when an
    overall main effect in a model containing interactions is intended.
    GLM F tests use likelihood-ratio deviances and an estimated dispersion.
    Term hypotheses follow Fox and Weisberg (2019); mixed-model F adjustments
    follow Kenward and Roger (1997).
    """
    if getattr(model, "_model_type", None) == "glmmTMB":
        if component not in {"cond", "zi", "disp"}:
            raise ValueError("component must be 'cond', 'zi' or 'disp'")
        if test_statistic is not None and test_statistic.lower() not in {"chisq", "wald"}:
            raise ValueError("glmmTMB component ANOVA supports Wald chi-squared tests")
        info = model.component_data(component)
        if not len(info["beta"]):
            raise ValueError(f"The model has no estimated {component} component")
        view = SimpleNamespace(
            beta=info["beta"], cov_beta=info["covariance"], X=info["X"],
            fixed_spec=info["fixed_spec"], family=model.family,
        )
        table = Anova(view, type=type, test_statistic="Chisq")
        table = table.rename(index={"Intercept": "(Intercept)"})
        table.attrs["component"] = component
        return table
    if component != "cond":
        raise ValueError("Separate model components require a glmmTMB result")
    kind = _type(type)
    if kind == 1:
        raise ValueError("Anova supports Type II and III; use anova for Type I")
    mixed = hasattr(model, "cov_beta")
    glm = not mixed and hasattr(model.model, "family")
    statistic = test_statistic or ("Chisq" if mixed else "LR" if glm else "F")
    statistic = statistic.lower()
    if statistic not in {"chisq", "wald", "lr", "f"}:
        raise ValueError("test_statistic must be Chisq, Wald, LR or F")
    if mixed and statistic == "f" and getattr(model, "family", None) not in (None, "gaussian"):
        raise ValueError("Kenward-Roger F tests require a Gaussian linear mixed model")
    if mixed and statistic == "lr":
        raise ValueError("Term-wise LR tests are available for GLM fits")
    beta = np.asarray(model.beta if mixed else model.params, dtype=float)
    covariance = np.asarray(model.cov_beta if mixed else model.cov_params(), dtype=float)
    if mixed and statistic == "f":
        covariance = adjusted_covariance(model)
    x = np.asarray(model.X if mixed else model.model.exog, dtype=float)
    terms = _terms(model)
    if kind == 3 and np.linalg.matrix_rank(x) < x.shape[1]:
        raise ValueError("Type III tests require non-aliased fixed effects")
    dispersion = _dispersion(model, error_estimate) if glm and statistic == "f" else 1.0
    rows: dict[str, dict[str, float]] = {}
    for term in terms:
        if term == "Intercept" and (kind == 2 or (glm and statistic in {"lr", "f"})):
            continue
        hypothesis = _hypothesis(term, terms, kind, x, covariance)
        rank = int(np.linalg.matrix_rank(hypothesis @ covariance @ hypothesis.T))
        if rank == 0:
            continue
        estimate = hypothesis @ beta
        wald = float(estimate @ linalg.pinvh(hypothesis @ covariance @ hypothesis.T) @ estimate)
        if mixed and statistic == "f":
            result = contrast_test(model, hypothesis, method="kenward-roger")
            rows[term] = {"F": result.statistic, "Df": rank, "Df.res": result.den_df,
                          "Pr(>F)": result.pvalue}
        elif glm and statistic in {"lr", "f"}:
            higher = _relatives(term, terms) if kind == 2 else []
            exclude_higher = {j for name in higher for j in terms[name]}
            full_cols = [j for j in range(x.shape[1]) if j not in exclude_higher]
            reduced_cols = [j for j in full_cols if j not in terms[term]]
            if not reduced_cols:
                raise ValueError("A GLM LR comparison needs a nonempty reduced design")
            full = model if not exclude_higher else _glm_refit(model, full_cols)
            reduced = _glm_refit(model, reduced_cols)
            deviance = float(reduced.deviance - full.deviance)
            rank = round(reduced.df_resid - full.df_resid)
            if statistic == "lr":
                rows[term] = {"LR Chisq": deviance, "Df": rank,
                              "Pr(>Chisq)": float(stats.chi2.sf(deviance, rank))}
            else:
                f = deviance / rank / dispersion
                rows[term] = {"Sum Sq": deviance, "Df": rank, "F value": f,
                              "Pr(>F)": float(stats.f.sf(f, rank, model.df_resid))}
        elif statistic == "f":
            df = float(model.df_resid)
            rows[term] = {"Sum Sq": wald * float(model.scale), "Df": rank,
                          "F value": wald / rank,
                          "Pr(>F)": float(stats.f.sf(wald / rank, rank, df))}
        else:
            rows[term] = {"Chisq": wald, "Df": rank,
                          "Pr(>Chisq)": float(stats.chi2.sf(wald, rank))}
    table = pd.DataFrame.from_dict(rows, orient="index")
    if not mixed and not glm:
        table.loc["Residuals", "Sum Sq"] = float(model.ssr)
        table.loc["Residuals", "Df"] = float(model.df_resid)
    if glm and statistic == "f":
        table.loc["Residuals", "Sum Sq"] = dispersion * float(model.df_resid)
        table.loc["Residuals", "Df"] = float(model.df_resid)
    if glm and statistic == "f" and kind == 3:
        table = table.rename(columns={"F value": "F values"})
    if glm and statistic in {"wald", "chisq"}:
        table = table[["Df", "Chisq", "Pr(>Chisq)"]]
    table.attrs.update(type=kind, test_statistic=statistic)
    return table


def anova(*models: Any, type: int | str | None = None, ddf: str = "satterthwaite",
          refit: bool = True, adjust_sigma: bool = True) -> pd.DataFrame:
    """Return lmerTest F tests or ML likelihood-ratio comparisons.

    REML models are refitted by ML before a multi-model comparison because
    REML likelihoods with different fixed designs are not comparable.
    Fixed-effect hypotheses follow Kuznetsova et al. (2017); GLS dispatch uses
    the fitted covariance and the conventions of Pinheiro and Bates (2000).
    """
    if not models:
        raise ValueError("At least one model is required")
    from rparity.gls._fit import GLSResult

    if isinstance(models[0], GLSResult):
        if not all(isinstance(model, GLSResult) for model in models):
            raise ValueError("GLS comparisons require only GLS models")
        gls_type = "sequential" if type is None else type
        if gls_type in (1, "I"):
            gls_type = "sequential"
        elif gls_type in (2, 3, "II", "III"):
            gls_type = "marginal"
        return models[0].anova(*models[1:], type=str(gls_type), adjust_sigma=adjust_sigma)
    if len(models) > 1:
        return _comparison(models, refit)
    model = models[0]
    type = 3 if type is None else type
    if not hasattr(model, "cov_beta"):
        import statsmodels.api as sm
        return sm.stats.anova_lm(model, typ=_type(type))  # type: ignore[no-any-return]
    if getattr(model, "family", None) not in (None, "gaussian"):
        return Anova(model, type=type, test_statistic="Chisq")
    terms = _terms(model)
    x = np.asarray(model.X, dtype=float)
    numerical = {key for key, (kind, _) in model.fixed_spec.encoder_state.items()
                 if kind.value == "numerical"}
    numeric_factors = {
        "Intercept" if str(term) == "1" else str(term):
        {factor.expr for factor in term.factors if factor.expr in numerical}
        for term in model.fixed_spec.terms
    }
    marginal = _marginal_hypotheses(model) if _type(type) == 3 else None
    rows = {}
    for term in terms:
        if term == "Intercept":
            continue
        # SAS containment requires identical continuous factors. Adding a new
        # continuous predictor does not contain the original categorical term.
        relatives = [higher for higher in _relatives(term, terms)
                     if numeric_factors[term] == numeric_factors[higher]]
        if marginal is not None:
            hypothesis = marginal[term]
        else:
            if _type(type) == 2:
                hypothesis = _hypothesis(term, terms, 2, x, linalg.pinvh(x.T @ x),
                                         relatives=relatives)
            else:
                hypothesis = _sas_sequential(x, terms[term], list(range(x.shape[1])))
        result = contrast_test(model, hypothesis, method=ddf)
        sum_sq = result.statistic * result.num_df * float(model.sigma)**2 / result.scale
        rows[term] = {"Sum Sq": sum_sq, "Mean Sq": sum_sq / result.num_df,
                      "NumDF": result.num_df, "DenDF": result.den_df,
                      "F value": result.statistic, "Pr(>F)": result.pvalue}
    result_table = pd.DataFrame.from_dict(rows, orient="index")
    result_table.attrs.update(type=_type(type), ddf=ddf)
    return result_table


def _comparison(models: tuple[Any, ...], refit: bool) -> pd.DataFrame:
    fitted = []
    for model in models:
        if refit and getattr(model, "reml", False):
            warnings.warn("Refitting REML models with ML for likelihood comparison", UserWarning,
                          stacklevel=3)
            if hasattr(model, "refit"):
                model = model.refit(reml=False)
            else:
                from rparity.lmm.lmer import lmer
                model = lmer(model.formula, model.data, REML=False,
                             weights=getattr(model, "weights", None),
                             offset=getattr(model, "offset", None))
        fitted.append(model)
    if any(len(m.y) != len(fitted[0].y) for m in fitted):
        raise ValueError("Model comparisons require the same number of observations")
    if any(not np.allclose(m.y, fitted[0].y) for m in fitted):
        raise ValueError("Model comparisons require the same response observations")
    counts = [len(m.beta) + len(m.theta) + int(getattr(m, "family", None) in (None, "gaussian"))
              for m in fitted]
    order = np.argsort(counts, kind="stable")
    fitted = [fitted[i] for i in order]
    counts = [counts[i] for i in order]
    rows: list[dict[str, float]] = []
    for i, (model, count) in enumerate(zip(fitted, counts, strict=True)):
        ll = float(model.logLik())
        row = {"npar": count, "AIC": -2 * ll + 2 * count,
               "BIC": -2 * ll + np.log(len(model.y)) * count, "logLik": ll,
               "-2*log(L)": -2 * ll, "Chisq": np.nan, "Df": np.nan, "Pr(>Chisq)": np.nan}
        if i:
            statistic = max(0.0, 2 * (ll - rows[-1]["logLik"]))
            df = count - counts[i - 1]
            row.update(Chisq=statistic, Df=df,
                       **{"Pr(>Chisq)": float(stats.chi2.sf(statistic, df)) if df > 0 else np.nan})
        rows.append(row)
    return pd.DataFrame(rows, index=[m.formula for m in fitted])


__all__ = ["Anova", "anova"]
