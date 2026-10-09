"""Independent numerical checks for the GLS likelihood and covariance models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import linalg, stats

from rparity.gls import corAR1, corARMA, corCompSymm, corSymm, gls, varExp, varIdent, varPower


def data_frame() -> pd.DataFrame:
    rng = np.random.default_rng(945)
    return pd.DataFrame(
        {
            "y": 2 + rng.normal(size=30),
            "x": rng.normal(size=30),
            "t": np.tile([1, 2, 4], 10),
            "g": np.repeat(range(10), 3),
            "f": np.repeat(["a", "b"], 15),
        }
    )


@pytest.mark.parametrize("method", ["ML", "REML"])
def test_independent_gls_equals_gaussian_ols(method):
    frame = data_frame()
    model = gls("y ~ x", frame, method=method)
    X = np.column_stack([np.ones(len(frame)), frame.x])
    beta = linalg.lstsq(X, frame.y)[0]
    residual = frame.y - X @ beta
    df = len(frame) - 2 if method == "REML" else len(frame)
    sigma2 = residual @ residual / df
    expected = -0.5 * (df * (np.log(2 * np.pi * sigma2) + 1))
    if method == "REML":
        expected -= 0.5 * np.linalg.slogdet(X.T @ X)[1]
    assert_allclose(model.beta, beta, atol=1e-12)
    assert_allclose(model.logLik(), expected, atol=1e-12)
    assert_allclose(model.cov_beta, linalg.inv(X.T @ X) * (residual @ residual / (len(frame) - 2)))
    interval = model.intervals()["sigma"]
    assert_allclose(
        interval.loc["sigma", "lower"], np.sqrt(sigma2 * df / stats.chi2.ppf(0.975, df))
    )


@pytest.mark.parametrize(
    "structure",
    [
        corAR1(0.35, form="~t|g", fixed=True),
        corCompSymm(0.25, form="~1|g", fixed=True),
        corARMA([0.35], p=1, form="~t|g", fixed=True),
        corARMA([0.35], q=1, form="~t|g", fixed=True),
        corSymm([0.2, 0.1, 0.3], form="~1|g", fixed=True),
    ],
)
def test_fixed_covariance_matches_direct_gls(structure):
    frame = data_frame()
    model = gls("y ~ x", frame, correlation=structure)
    covariance = model._covariance
    full = np.zeros((len(frame), len(frame)))
    for block, matrix in zip(covariance.blocks, covariance.matrices(model._theta)):
        full[np.ix_(block, block)] = matrix
    precision = linalg.inv(full)
    information = model.X.T @ precision @ model.X
    beta = linalg.solve(information, model.X.T @ precision @ model.y)
    assert_allclose(model.beta, beta, atol=1e-12)
    residual = model.y - model.X @ beta
    sigma2 = residual @ precision @ residual / model.df_resid
    assert_allclose(model.vcov(), sigma2 * linalg.inv(information), atol=1e-12)
    expected = -0.5 * (
        model.df_resid * (np.log(2 * np.pi * sigma2) + 1)
        + np.linalg.slogdet(full)[1]
        + np.linalg.slogdet(information)[1]
    )
    assert_allclose(model.logLik(), expected, atol=1e-12)


@pytest.mark.parametrize(
    "structure",
    [
        varIdent({"b": 1.8}, form="~1|f", fixed={"b": 1.8}),
        varPower(0.4, form="~t", fixed=0.4),
        varExp(0.2, form="~t", fixed=0.2),
    ],
)
def test_fixed_variances_match_weighted_ols(structure):
    frame = data_frame()
    model = gls("y ~ x", frame, weights=structure)
    multiplier = model._covariance.sd_multipliers(model._covariance.expand(model._theta))
    X = model.X / multiplier[:, None]
    y = model.y / multiplier
    expected = linalg.lstsq(X, y)[0]
    assert_allclose(model.beta, expected, atol=1e-12)
    assert_allclose(model.predict(), model.fitted())


def test_prediction_and_anova_interfaces():
    frame = data_frame()
    model = gls("y ~ x + f", frame, correlation=corAR1(form="~ t | g"))
    predictions = model.predict(frame.iloc[:3], se_fit=True)
    assert list(predictions) == ["fit", "se.fit"]
    assert_allclose(predictions["fit"], model.fitted().iloc[:3])
    assert (predictions["se.fit"] > 0).all()
    sequential, marginal = model.anova(), model.anova(type="marginal")
    assert list(sequential) == ["numDF", "F-value", "p-value"]
    assert list(sequential.index) == ["(Intercept)", "x", "f"]
    assert_allclose(sequential.loc["f"], marginal.loc["f"])
    assert model.summary().startswith("Generalized least squares fit by REML")
    simple = gls("y ~ x", frame, method="ML")
    complex_model = gls("y ~ x + f", frame, method="ML")
    table = simple.anova(complex_model)
    assert table.loc[1, "L.Ratio"] >= 0
    assert_allclose(table.loc[1, "p-value"], stats.chi2.sf(table.loc[1, "L.Ratio"], 1))


def test_power_on_fitted_mean_converges():
    frame = data_frame()
    model = gls("y ~ x", frame, weights=varPower())
    assert model.converged
    assert_allclose(model._covariance.fitted, model.fitted(), rtol=1e-7)
    assert np.isfinite(model.intervals()["sigma"].to_numpy()).all()


def test_unbalanced_groups_and_original_row_order():
    frame = data_frame().drop(index=[2, 4, 6, 10]).sample(frac=1, random_state=4)
    model = gls("y ~ x", frame, correlation=corAR1(0.2, form="~ t | g", fixed=True))
    assert model.fitted().index.equals(frame.index)
    assert model.predict(frame).index.equals(frame.index)
    assert_allclose(model.predict(frame), model.fitted())


def test_distinct_integer_time_validation():
    frame = data_frame()
    frame.loc[0, "t"] = 2
    with pytest.raises(ValueError, match="unique"):
        gls("y ~ x", frame, correlation=corAR1(form="~ t | g"))
    frame["t"] = frame["t"].astype(float)
    frame.loc[0, "t"] = 1.5
    with pytest.raises(ValueError, match="integers"):
        gls("y ~ x", frame, correlation=corAR1(form="~ t | g"))


def test_stationarity_and_positive_definiteness_validation():
    with pytest.raises(ValueError, match="positive definite"):
        gls("y ~ x", data_frame(), correlation=corCompSymm(-0.8, form="~1|g"))
    with pytest.raises(ValueError, match="positive definite"):
        gls("y ~ x", data_frame(), correlation=corSymm([0.9, 0.9, -0.9], form="~1|g"))
    with pytest.raises(ValueError, match="stationary"):
        gls("y ~ x", data_frame(), correlation=corARMA([0.9, 0.9], p=2, form="~t|g"))


def test_complete_cases_are_retained():
    frame = data_frame()
    frame.loc[2, "y"] = np.nan
    model = gls("y ~ x", frame)
    assert model.nobs == 29
    assert 2 not in model.fitted().index
    with pytest.raises(ValueError, match="null"):
        gls("y ~ x", frame, na_action="raise")


@pytest.mark.parametrize("method", ["ML", "REML"])
def test_near_singular_symm_preserves_closed_form_likelihood(method):
    # Identical pairs have residuals only in the common-mode direction.
    # The covariance determinant is (1-rho)*(1+rho), even at the last
    # representable rho below one. Flooring a Cholesky diagonal changes
    # that likelihood rather than stabilizing the same statistical model.
    rho = np.nextafter(1.0, 0.0)
    pairs = np.linspace(-2.0, 2.0, 10)
    frame = pd.DataFrame({"y": np.repeat(pairs, 2), "g": np.repeat(range(10), 2)})
    with pytest.warns(RuntimeWarning, match="nearly singular"):
        model = gls(
            "y ~ 1", frame, method=method,
            correlation=corSymm([rho], form="~1|g", fixed=True),
        )
    count = len(frame) - (method == "REML")
    quadratic = 2 * np.sum((pairs - pairs.mean()) ** 2) / (1 + rho)
    sigma2 = quadratic / count
    logdet = len(pairs) * (np.log1p(-rho) + np.log1p(rho))
    information = len(frame) / (1 + rho)
    expected = -0.5 * (count * (np.log(2 * np.pi * sigma2) + 1) + logdet)
    if method == "REML":
        expected -= 0.5 * np.log(information)
    assert_allclose(model.beta, [pairs.mean()], atol=1e-14)
    assert_allclose(model.logLik(), expected, rtol=0, atol=1e-10)
    assert_allclose(model.sigma**2, sigma2, rtol=1e-12)
    assert_allclose(np.sum(model.residuals("normalized") ** 2), count, rtol=1e-12)
    assert np.isfinite(model.anova().to_numpy()).all()
    assert np.isfinite(model.intervals()["sigma"].to_numpy()).all()
