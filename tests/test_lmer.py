"""Independent Gaussian identities and model-structure regression tests."""
import numpy as np
import pandas as pd
import pytest
from scipy import optimize

from rparity.lmm.lmer import SingularFitWarning, lmer


def sample(seed=31):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(12), 6)
    x = np.tile(np.linspace(-1, 1, 6), 12)
    effects = rng.multivariate_normal([0, 0], [[1, .15], [.15, .2]], 12)
    y = 2 + 1.2*x + effects[g, 0] + effects[g, 1]*x + rng.normal(scale=.5, size=len(g))
    return pd.DataFrame({"y": y, "x": x, "g": g, "h": np.tile(np.arange(6), 12)})


@pytest.mark.parametrize("reml", [True, False])
def test_balanced_random_intercept_closed_form(reml):
    data = sample()
    model = lmer("y~x+(1|g)", data, reml=reml)
    _x, y = data.x.to_numpy(), data.y.to_numpy()
    beta = np.linalg.lstsq(model.X, y, rcond=None)[0]
    residual = y - model.X @ beta
    means = residual.reshape(12, 6).mean(axis=1)
    within = residual - np.repeat(means, 6)
    sigma2 = within @ within / (72-12-1 if reml else 72-12)
    variance = means @ means / (11 if reml else 12) - sigma2/6
    assert model.beta == pytest.approx(beta, rel=1e-9)
    assert model.sigma**2 == pytest.approx(sigma2, rel=1e-6)
    assert model.VarCorr()["g"].iloc[0, 0] == pytest.approx(variance, rel=1e-6)


@pytest.mark.parametrize("formula", ["y~x+(x|g)", "y~x+(x||g)", "y~x+(0+x|g)",
                                      "y~x+(1|g)+(1|h)", "y~x+(1|g/h)"])
def test_random_structures_and_conditional_predictions(formula):
    data = sample()
    model = lmer(formula, data)
    assert model.predict(data) == pytest.approx(model.fitted(), abs=1e-10)
    assert model.predict(data, re_form="NA") == pytest.approx(model.X @ model.beta)
    assert model.residuals() + model.fitted() == pytest.approx(model.y)
    assert np.linalg.eigvalsh(model.cov_beta).min() > 0
    for frame in model.ranef().values():
        assert frame.attrs["postVar"].shape == (frame.shape[1], frame.shape[1], len(frame))


def test_profile_gradient_and_conditional_covariance():
    model = lmer("y~x+(x|g)", sample())
    theta = np.array([.9, .3, .7])
    numerical = optimize.approx_fprime(theta, lambda t: model._evaluate(t)[0], 1e-6)
    assert model._objective_and_gradient(theta)[1] == pytest.approx(numerical, abs=5e-5)
    G = model._relative_G*model.sigma**2
    V = model.marginal_covariance_at(model.variance_params)
    expected = G - G @ model.Z.T @ np.linalg.solve(V, model.Z @ G)
    assert model._conditional_covariance == pytest.approx(expected, abs=1e-12)
    assert model.variance_objective(model.variance_params) == pytest.approx(model.deviance())


def test_weights_offset_and_information_criteria():
    data = sample()
    offset = np.linspace(-.3, .8, len(data))
    data.y += offset
    model = lmer("y~x+(1|g)", data, weights=np.linspace(.5, 2, len(data)), offset=offset)
    without = data.copy()
    without.y -= offset
    comparison = lmer("y~x+(1|g)", without, weights=model.weights)
    assert model.beta == pytest.approx(comparison.beta, abs=1e-8)
    assert model.predict(data, offset=offset) == pytest.approx(model.fitted())
    assert model.AIC() == pytest.approx(-2*model.logLik() + 2*model.df_model)
    assert model.BIC() == pytest.approx(-2*model.logLik() + np.log(len(data))*model.df_model)


def test_new_group_levels():
    model = lmer("y~x+(x|g)", sample())
    new = pd.DataFrame({"x": [.2, .4], "g": [101, 102]})
    with pytest.raises(ValueError, match="new grouping level"):
        model.predict(new)
    assert model.predict(new, allow_new_levels=True) == pytest.approx(
        model.predict(new, re_form="NA")
    )


def test_missing_observations_and_polars():
    import polars as pl
    data = sample()
    data.loc[2, "y"] = np.nan
    data.loc[6, "g"] = np.nan
    pandas_model = lmer("y~x+(1|g)", data)
    polars_model = lmer("y~x+(1|g)", pl.from_pandas(data))
    assert pandas_model.nobs == len(data)-2
    assert polars_model.beta == pytest.approx(pandas_model.beta)


def test_exact_zero_variance_boundary():
    g = np.repeat(np.arange(6), 4)
    data = pd.DataFrame({"g": g, "y": np.tile([-1, 1, 1, -1], 6)})
    with pytest.warns(SingularFitWarning):
        model = lmer("y~1+(1|g)", data)
    assert model.is_singular()
    assert model.theta[0] == pytest.approx(0, abs=1e-8)
    assert model.ranef()["g"].to_numpy() == pytest.approx(np.zeros((6, 1)))


def test_rank_deficiency_and_bad_weights():
    data = sample()
    data["duplicate"] = data.x
    with pytest.warns(UserWarning, match="rank deficient"):
        model = lmer("y~x+duplicate+(1|g)", data)
    assert model.p == 2
    assert model.predict(data) == pytest.approx(model.fitted())
    with pytest.raises(ValueError, match="strictly positive"):
        lmer("y~x+(1|g)", data, weights=np.zeros(len(data)))
