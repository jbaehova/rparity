"""Likelihood identities and GLMM behavior independent of an R installation."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, gammaln

from rparity.lmm._glmer_math import (
    conditional_mode,
    conditional_terms,
    numerical_hessian,
    observed_information,
)
from rparity.lmm.glmer import glmer


@pytest.fixture(scope="module")
def binomial_data():
    rng = np.random.default_rng(128)
    groups = np.repeat(np.arange(8), 12)
    x = rng.normal(size=len(groups))
    b = np.array([-1.2, -0.8, -0.5, -0.1, 0.3, 0.6, 1.0, 1.4])
    trials = np.full(len(groups), 8)
    k = rng.binomial(trials, expit(-0.3 + 0.7 * x + b[groups]))
    return pd.DataFrame(
        {
            "y": k / trials,
            "success": k,
            "failure": trials - k,
            "trials": trials,
            "x": x,
            "g": groups,
        }
    )


@pytest.fixture(scope="module")
def binomial_fit(binomial_data):
    return glmer("y ~ x + (1|g)", binomial_data, weights="trials")


def test_no_variance_laplace_is_exact_glm():
    eta = np.array([-0.5, 0.2, 0.8])
    y = np.array([0.0, 1.0, 1.0])
    expected = float(np.logaddexp(0, eta).sum() - y @ eta)
    value, modes, _, hessian = conditional_mode(
        eta, np.zeros((3, 2)), y, np.ones(3), np.ones(3), "binomial", "logit"
    )
    assert value == pytest.approx(expected, abs=1e-12)
    np.testing.assert_array_equal(modes, [0, 0])
    np.testing.assert_array_equal(hessian, np.eye(2))


@pytest.mark.parametrize("link", ["logit", "probit", "cloglog"])
def test_conditional_scores_are_likelihood_derivatives(link):
    eta = np.array([-0.8, 0.3, 1.1])
    y = np.array([0.2, 0.7, 0.9])
    n = np.full(3, 10.0)
    _, gradient, _ = conditional_terms(eta, y, n, np.ones(3), "binomial", link)
    actual = []
    for i in range(3):
        step = np.zeros(3)
        step[i] = 1e-6
        plus = conditional_terms(eta + step, y, n, np.ones(3), "binomial", link)[0]
        minus = conditional_terms(eta - step, y, n, np.ones(3), "binomial", link)[0]
        actual.append((plus - minus) / 2e-6)
    np.testing.assert_allclose(gradient, actual, atol=1e-7)


def test_poisson_constant():
    y = np.array([0.0, 2.0, 3.0])
    eta = np.array([0.1, 0.4, 0.7])
    value, _, _ = conditional_terms(eta, y, np.ones(3), np.ones(3), "poisson", "log")
    assert value == pytest.approx(np.sum(np.exp(eta) - y * eta + gammaln(y + 1)))


def test_grouped_response_matches_cbind(binomial_data, binomial_fit):
    other = glmer("cbind(success, failure) ~ x + (1|g)", binomial_data)
    np.testing.assert_allclose(other.beta, binomial_fit.beta, atol=1e-9)
    assert other.logLik() == pytest.approx(binomial_fit.logLik(), abs=1e-9)
    np.testing.assert_allclose(other.fitted(), binomial_fit.fitted(), atol=1e-9)


def test_prediction_and_random_modes(binomial_data, binomial_fit):
    fit = binomial_fit
    np.testing.assert_allclose(fit.predict(binomial_data), fit.fitted(), atol=1e-12)
    np.testing.assert_allclose(fit.predict(re_form="NA", type="link"), fit.X @ fit.beta)
    frame = binomial_data.iloc[:2].copy()
    frame["g"] = 100
    with pytest.raises(ValueError, match="New group"):
        fit.predict(frame)
    np.testing.assert_allclose(
        fit.predict(frame, allow_new_levels=True), fit.predict(frame, re_form="NA")
    )
    assert fit.ranef(cond_var=True)["g"].attrs["postVar"].shape == (1, 1, 8)
    assert np.min(np.linalg.eigvalsh(fit.cov_beta)) > 0
    assert fit.AIC() == pytest.approx(-2 * fit.logLik() + 6)


def test_hessian():
    matrix = np.array([[3.0, 1.0], [1.0, 2.0]])
    actual = numerical_hessian(lambda x: float(x @ matrix @ x / 2), np.array([0.2, -0.1]))
    np.testing.assert_allclose(actual, matrix, atol=1e-7)


def test_unsupported_options(binomial_data):
    with pytest.raises(NotImplementedError, match="nAGQ=1"):
        glmer("y~x+(1|g)", binomial_data, nAGQ=2)
    with pytest.raises(ValueError, match="families"):
        glmer("y~x+(1|g)", binomial_data, family="gamma")


@pytest.mark.parametrize("link", ["probit", "cloglog"])
def test_noncanonical_models(binomial_data, link):
    fit = glmer("cbind(success, failure)~x+(1|g)", binomial_data, link=link)
    assert np.isfinite(fit.logLik())
    assert fit.theta[0] > 0
    assert np.all((fit.fitted() > 0) & (fit.fitted() < 1))
    np.testing.assert_allclose(fit.predict(binomial_data), fit.fitted(), atol=1e-12)


def test_poisson_formula_offset_prediction():
    rng = np.random.default_rng(309)
    groups = np.repeat(np.arange(6), 12)
    exposure = rng.uniform(-0.7, 0.7, len(groups))
    x = rng.normal(size=len(groups))
    b = np.array([-1.2, -0.7, -0.3, 0.3, 0.7, 1.2])
    y = rng.poisson(np.exp(0.3 + 0.3 * x + exposure + b[groups]))
    frame = pd.DataFrame({"y": y, "x": x, "g": groups, "exposure": exposure})
    fit = glmer("y~x+offset(exposure)+(1|g)", frame, family="poisson")
    other = glmer("y~x+(1|g)", frame, family="poisson", offset="exposure")
    np.testing.assert_allclose(fit.beta, other.beta, atol=1e-10)
    np.testing.assert_allclose(fit.predict(frame), fit.fitted(), atol=1e-12)
    np.testing.assert_allclose(other.predict(frame), other.fitted(), atol=1e-12)
    assert fit.deviance() >= 0


def test_missing_unused_columns_do_not_drop_rows(binomial_data, binomial_fit):
    frame = binomial_data.copy()
    frame["unrelated"] = np.nan
    fit = glmer("y~x+(1|g)", frame, weights="trials")
    assert fit.nobs == binomial_fit.nobs
    np.testing.assert_allclose(fit.beta, binomial_fit.beta)


@pytest.mark.parametrize("link", ["probit", "cloglog"])
def test_observed_information_is_score_derivative(link):
    eta = np.array([-1.5, -0.2, 0.8, 2.0])
    response = np.array([0.0, 0.2, 0.6, 1.0])
    trials = np.full(4, 10.0)
    weights = np.ones(4)
    expected = observed_information(eta, response, trials, weights, "binomial", link)
    plus = conditional_terms(eta + 1e-6, response, trials, weights, "binomial", link)[1]
    minus = conditional_terms(eta - 1e-6, response, trials, weights, "binomial", link)[1]
    np.testing.assert_allclose(expected, (plus - minus) / 2e-6, atol=1e-7)


def test_cbind_expression_and_summary(binomial_data, binomial_fit):
    fit = glmer("cbind(success,trials-success)~x+(1|g)", binomial_data)
    np.testing.assert_allclose(fit.beta, binomial_fit.beta, atol=1e-9)
    text = fit.summary()
    sections = [
        "Scaled residuals:",
        "Random effects:",
        "Fixed effects:",
        "Correlation of Fixed Effects:",
    ]
    assert [text.index(section) for section in sections] == sorted(
        text.index(section) for section in sections
    )
    assert "Variance Std.Dev." in text
    assert "Pr(>|z|)" in text


def test_fractional_prior_weights_are_supported(binomial_data):
    frame = binomial_data.copy()
    frame["y"] = (frame["success"] > 3).astype(float)
    with pytest.warns(UserWarning, match="non-integer"):
        fit = glmer("y~x+(1|g)", frame, weights=1.5)
    assert np.isfinite(fit.logLik())
    np.testing.assert_array_equal(fit.y, frame["y"])
    np.testing.assert_allclose(fit.weights, 1.5)
    np.testing.assert_allclose(fit.trials, 1)
