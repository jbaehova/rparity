"""Analytical checks for means, weighting, contrasts and transformations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from rparity.emm import EmmGrid, contrast, emmeans, emtrends, joint_tests, pairs, ref_grid, regrid


def factorial_data() -> pd.DataFrame:
    rows = []
    for a, treatment in enumerate(["a", "b", "c"]):
        for b, sex in enumerate(["female", "male"]):
            count = 3 + a + 3 * b
            errors = np.linspace(-1, 1, count)
            for error in errors:
                rows.append({"y": 2 + 3 * a + 2 * b + a * b + error,
                             "treatment": treatment, "sex": sex})
    return pd.DataFrame(rows)


def test_factorial_marginal_means_and_cell_weights() -> None:
    data = factorial_data()
    model = smf.ols("y ~ treatment * sex", data).fit()
    equal = emmeans(model, "treatment")
    assert isinstance(equal, EmmGrid)
    np.testing.assert_allclose(equal.estimates, [3, 6.5, 10])
    cells = emmeans(model, "treatment", weights="cells")
    assert isinstance(cells, EmmGrid)
    expected = data.groupby("treatment")["y"].mean().to_numpy()
    np.testing.assert_allclose(cells.estimates, expected)
    proportional = emmeans(model, "treatment", weights="proportional")
    assert isinstance(proportional, EmmGrid)
    fraction = (data.sex == "male").mean()
    np.testing.assert_allclose(proportional.estimates, 2 + 3 * np.arange(3) + (2 + np.arange(3)) * fraction)


@pytest.mark.parametrize("adjust", ["none", "bonferroni", "holm", "sidak", "fdr", "tukey"])
def test_pairs_have_correct_linear_variance_and_adjustment(adjust: str) -> None:
    model = smf.ols("y ~ treatment * sex", factorial_data()).fit()
    means = emmeans(model, "treatment")
    assert isinstance(means, EmmGrid)
    comparisons = pairs(means, adjust=adjust)
    table = comparisons.summary()
    expected_matrix = np.array([[1, -1, 0], [1, 0, -1], [0, 1, -1]])
    np.testing.assert_allclose(comparisons.estimates, expected_matrix @ means.estimates)
    expected_se = np.sqrt(np.diag(expected_matrix @ means.covariance @ expected_matrix.T))
    np.testing.assert_allclose(table.SE, expected_se)
    raw = 2 * stats.t.sf(abs(table["t.ratio"]), model.df_resid)
    if adjust == "none":
        np.testing.assert_allclose(table["p.value"], raw)
    else:
        assert np.all(table["p.value"] >= raw - 1e-12)
    if adjust == "tukey":
        np.testing.assert_allclose(table["p.value"], stats.studentized_range.sf(
            abs(table["t.ratio"]) * np.sqrt(2), 3, model.df_resid,
        ))


def test_by_groups_and_formula_specs() -> None:
    model = smf.ols("y ~ treatment * sex", factorial_data()).fit()
    result = emmeans(model, "pairwise ~ treatment | sex")
    assert isinstance(result, dict)
    assert len(result["emmeans"]) == 6
    comparison = result["contrasts"]
    assert len(comparison) == 6
    female = comparison.summary().query("sex == 'female'")
    np.testing.assert_allclose(female.estimate, [-3, -6, -3])
    male = comparison.summary().query("sex == 'male'")
    np.testing.assert_allclose(male.estimate, [-4, -8, -4])


def test_contrast_families_and_custom_coefficients() -> None:
    model = smf.ols("y ~ treatment", factorial_data()).fit()
    means = emmeans(model, "treatment")
    assert isinstance(means, EmmGrid)
    np.testing.assert_allclose(contrast(means, "poly").estimates,
                               np.array([[-1, 0, 1], [1, -2, 1]]) @ means.estimates)
    np.testing.assert_allclose(contrast(means, "consec").estimates, np.diff(means.estimates))
    np.testing.assert_allclose(contrast(means, "trt.vs.ctrl").estimates,
                               means.estimates[1:] - means.estimates[0])
    np.testing.assert_allclose(contrast(means, {"average vs first": [-1, 0.5, 0.5]}).estimates,
                               [-means.estimates[0] + means.estimates[1:].mean()])
    np.testing.assert_allclose(pairs(means, reverse=True).estimates, -pairs(means).estimates)


def test_covariate_at_and_interaction_trends() -> None:
    rng = np.random.default_rng(42)
    data = pd.DataFrame({"x": np.tile(np.arange(12), 2), "group": np.repeat(["a", "b"], 12)})
    data["y"] = 1 + 2 * data.x + 3 * (data.group == "b") + data.x * (data.group == "b") + rng.normal(size=24)
    model = smf.ols("y ~ x * group", data).fit()
    trend = emtrends(model, "group", var="x")
    assert isinstance(trend, EmmGrid)
    np.testing.assert_allclose(trend.estimates,
                               [model.params["x"], model.params["x"] + model.params["x:group[T.b]"]])
    means = emmeans(model, "group", at={"x": 3})
    assert isinstance(means, EmmGrid)
    np.testing.assert_allclose(means.estimates,
                               model.predict(pd.DataFrame({"x": [3, 3], "group": ["a", "b"]})))


def test_glm_backtransform_delta_method_and_regrid() -> None:
    data = factorial_data()
    data["y"] = np.rint(np.exp(0.2 + 0.1 * data.y)).astype(int)
    model = smf.glm("y ~ treatment + sex", data, family=sm.families.Poisson()).fit()
    link_means = emmeans(model, "treatment")
    assert isinstance(link_means, EmmGrid)
    response = link_means.summary(type="response")
    np.testing.assert_allclose(response.rate, np.exp(link_means.estimates))
    np.testing.assert_allclose(response.SE, np.exp(link_means.estimates) * np.sqrt(np.diag(link_means.covariance)))
    assert np.all(np.isinf(response.df))
    ratio = pairs(link_means).summary(type="response")
    np.testing.assert_allclose(ratio.ratio, np.exp(pairs(link_means).estimates))
    transformed = regrid(ref_grid(model))
    arithmetic = emmeans(transformed, "treatment", type="response")
    assert isinstance(arithmetic, EmmGrid)
    assert np.all(arithmetic.estimates > response.rate)
    differences = pairs(regrid(link_means), adjust="none")
    assert differences.regridded
    np.testing.assert_allclose(differences.estimates, [-response.rate.iloc[1] + response.rate.iloc[0],
                                                     -response.rate.iloc[2] + response.rate.iloc[0],
                                                     -response.rate.iloc[2] + response.rate.iloc[1]])


def test_joint_tests_matches_balanced_type_three_wald_test() -> None:
    data = factorial_data()
    model = smf.ols("y ~ treatment * sex", data).fit()
    tests = joint_tests(model).set_index("model term")
    interaction_indices = [i for i, name in enumerate(model.params.index) if ":" in name]
    hypothesis = np.eye(len(model.params))[interaction_indices]
    expected = model.f_test(hypothesis)
    assert tests.loc["treatment:sex", "df1"] == 2
    np.testing.assert_allclose(tests.loc["treatment:sex", "F.ratio"], round(float(expected.fvalue), 3))
    np.testing.assert_allclose(tests.loc["treatment:sex", "p.value"], float(expected.pvalue))


def test_invalid_weights_and_unknown_variables_raise() -> None:
    model = smf.ols("y ~ treatment * sex", factorial_data()).fit()
    with pytest.raises(ValueError, match="Unknown"):
        emmeans(model, "missing")
    with pytest.raises(ValueError, match="weights"):
        emmeans(model, "treatment", weights=[0, 0])


def test_missing_rows_do_not_change_covariate_reference_value() -> None:
    data = pd.DataFrame({"y": [1, 2, 3, 5, np.nan], "x": [0, 1, 2, 3, 1000]})
    model = smf.ols("y ~ x", data).fit()
    means = emmeans(model, "1")
    assert isinstance(means, EmmGrid)
    np.testing.assert_allclose(means.estimates, model.predict(pd.DataFrame({"x": [1.5]})))


def test_transformed_ols_response_uses_inverse_transformation() -> None:
    data = factorial_data()
    data["y"] = np.exp(data.y / 10)
    model = smf.ols("np.log(y) ~ treatment", data).fit()
    means = emmeans(model, "treatment")
    assert isinstance(means, EmmGrid)
    np.testing.assert_allclose(means.summary(type="response").response, np.exp(means.estimates))
