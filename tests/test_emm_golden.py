"""Synthetic R emmeans black-box comparisons with stated field tolerances."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from rparity.emm import EmmGrid, contrast, emmeans, emtrends, joint_tests

FILES = sorted((Path(__file__).parent / "golden/emm").glob("*.json"))


def _fit(spec: dict[str, Any]) -> Any:
    data = pd.DataFrame(spec["data"])
    if spec["call"] == "lmer":
        from rparity.lmm import lmer

        return lmer(spec["formula"], data)
    if spec["call"] == "glmer":
        from rparity.lmm import glmer

        return glmer(spec["formula"], data, family=spec["family"], link=spec["link"])
    if spec["call"] == "gls":
        from rparity.gls import corAR1, gls

        return gls(spec["formula"], data, correlation=corAR1(form="~ t | g"))
    if spec["call"] == "lm":
        return smf.ols(spec["formula"], data).fit()
    if spec["family"] == "poisson":
        family = sm.families.Poisson()
    else:
        links = {"logit": sm.families.links.Logit, "probit": sm.families.links.Probit,
                 "cloglog": sm.families.links.CLogLog}
        family = sm.families.Binomial(link=links[spec["link"]]())
    return smf.glm(spec["formula"], data, family=family).fit(
        atol=1e-13, rtol=1e-12, maxiter=1000, wls_method="qr",
    )


@pytest.mark.golden
@pytest.mark.parametrize("path", FILES, ids=lambda path: path.stem)
def test_r_marginal_mean_golden(path: Path, record_property: Callable[[str, Any], None]) -> None:
    case = json.loads(path.read_text())
    spec = case["spec"]
    model = _fit(spec)
    args = spec["emm_args"]
    oracle = case["oracle"]["result"]
    assert "error" not in oracle, oracle
    log_likelihood = float(model.logLik() if spec['call'] in {'lmer', 'glmer', 'gls'} else model.llf)
    assert log_likelihood >= oracle['logLik'] - 1e-6
    record_property('better_optimum', log_likelihood > oracle['logLik'] + 1e-6)
    if "boundary" in case:
        # The saturated cell intercept tends to infinity. Covariance and EMMs
        # then depend on the IRLS stopping point, but the likelihood supremum
        # remains well defined. Keep both original tables in the fixture.
        optimum = case["fit_oracle"]["result"]["logLik"]
        assert model.llf >= optimum - 1e-6, case["boundary"]
        fitted = np.asarray(model.fittedvalues)
        numeric_boundary = np.any(fitted < 1e-10) or (
            spec["family"] == "binomial" and np.any(fitted > 1 - 1e-10)
        )
        if numeric_boundary:
            boundary_grid = emmeans(model, "a")
            assert isinstance(boundary_grid, EmmGrid)
            with pytest.warns(RuntimeWarning, match="Boundary GLM fit"):
                boundary_grid.summary()
        return
    if spec["operation"] == "joint_tests":
        actual = joint_tests(model, **args, lmer_df=spec["ddf"])
    else:
        function = emtrends if spec["operation"] == "emtrends" else emmeans
        grid = function(model, **args, lmer_df=spec["ddf"])
        assert isinstance(grid, EmmGrid)
        if "contrast" in spec:
            grid = contrast(grid, **spec["contrast"])
        actual = grid.summary(infer=(True, True), type=spec.get("response_type", "link"),
                              adjust=spec.get("adjust", "none"))
    expected = pd.DataFrame(oracle["values"])
    numeric = [name for name in expected if name not in {"a", "b", "contrast", "model term", "group"}]
    keys = [name for name in expected if name not in numeric]
    actual = actual.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    for key in keys:
        assert actual[key].tolist() == expected[key].tolist()
    for key in numeric:
        if key in {"df", "df2"}:
            # JSON null denotes the R infinity used for asymptotic inference.
            target = expected[key].fillna(np.inf).to_numpy(dtype=float)
        else:
            target = expected[key].to_numpy(dtype=float)
        if key == "p.value":
            np.testing.assert_allclose(actual[key], target, rtol=0, atol=1e-4)
        elif key in {"df", "df1", "df2"}:
            np.testing.assert_allclose(actual[key], target, rtol=1e-3, atol=1e-12)
        elif key == "SE":
            np.testing.assert_allclose(actual[key], target, rtol=1e-4, atol=1e-12)
        elif spec["operation"] == "joint_tests" and key == "F.ratio":
            # The public R table exports F.ratio rounded to three decimals.
            np.testing.assert_allclose(actual[key], target, rtol=0, atol=0.00050001)
        elif key in {"lower.CL", "upper.CL", "asymp.LCL", "asymp.UCL"}:
            # CI limits inherit both specified input tolerances: SE1e-4 and
            # denominator df1e-3. Propagate these, particularly for GLS where
            # the R apVar Hessian is itself a finite-difference approximation.
            degree = expected["df"].fillna(np.inf).to_numpy(dtype=float)
            se = expected["SE"].to_numpy(dtype=float)
            adjustment = actual.attrs.get("adjust", "none")
            per_family = len(expected) / (expected[spec["emm_args"]["by"]].nunique()
                                          if "by" in spec["emm_args"] else 1)

            def critical(df: np.ndarray, adjustment: str = adjustment,
                         per_family: float = per_family) -> np.ndarray:
                if adjustment == "tukey":
                    return np.asarray(stats.studentized_range.ppf(0.95, 3, df)) / np.sqrt(2)
                probability = 0.975
                if adjustment in {"bonferroni", "holm", "fdr"}:
                    probability = 1 - 0.05 / (2 * per_family)
                elif adjustment == "sidak":
                    probability = (1 + 0.95 ** (1 / per_family)) / 2
                return np.asarray(stats.t.ppf(probability, df))

            quantile = critical(degree)
            q_error = np.maximum(abs(critical(degree * 0.999) - quantile),
                                 abs(critical(degree * 1.001) - quantile))
            bound = abs(target) * 1e-4 + se * (q_error + quantile * 1e-4) + 1e-6
            assert np.all(abs(actual[key].to_numpy() - target) <= bound), key
        elif key in {"t.ratio", "z.ratio"}:
            # These are derived from the SE and degrees of freedom. The task
            # specifies direct parity tolerances for estimates, SE, df and p.
            np.testing.assert_allclose(actual[key], target, rtol=1e-4, atol=1e-6, err_msg=key)
        else:
            np.testing.assert_allclose(actual[key], target, rtol=1e-6, atol=1e-12, err_msg=key)
