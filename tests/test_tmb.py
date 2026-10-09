"""Normalized likelihood identities and distributional-model behavior."""

import numpy as np
import pandas as pd
import pytest
from scipy import linalg, special, stats

from rparity.lmm import glmer, lmer
from rparity.tmb import glmmTMB
from rparity.tmb._families import observation_terms


@pytest.mark.parametrize(
    "family", ["poisson", "nbinom1", "nbinom2", "binomial", "beta", "gaussian"]
)
@pytest.mark.parametrize("zero_inflation", [False, True])
def test_likelihood_predictor_derivatives(family, zero_inflation):
    eta = np.array([-0.8, 0.2, 0.9])
    delta = np.array([0.5, 0.8, 1.0])
    y = np.array([0.0, 2.0, 4.0])
    trials = np.ones(3)
    if family == "binomial":
        y, trials = np.array([0.0, 0.4, 0.8]), np.full(3, 5.0)
    elif family == "beta":
        y = np.array([0.0 if zero_inflation else 0.1, 0.3, 0.8])
    elif family == "gaussian":
        y = np.array([0.0, 0.3, 0.8])
    zeta = np.array([-0.5, 0.2, 0.5]) if zero_inflation else None
    weights = np.array([1.0, 0.7, 1.5])

    def terms(e=eta, d=delta, z=zeta):
        return observation_terms(e, d, z, y, trials, weights, family)

    actual = terms()
    h = 1e-5
    derivatives = {
        "score": (terms(e=eta + h).value - terms(e=eta - h).value) / (2 * h),
        "curvature": (terms(e=eta + h).score - terms(e=eta - h).score) / (2 * h),
        "third": (terms(e=eta + h).curvature - terms(e=eta - h).curvature) / (2 * h),
        "disp_score": (terms(d=delta + h).value - terms(d=delta - h).value) / (2 * h),
        "score_disp": (terms(d=delta + h).score - terms(d=delta - h).score) / (2 * h),
        "curvature_disp": (terms(d=delta + h).curvature - terms(d=delta - h).curvature) / (2 * h),
    }
    if zero_inflation:
        derivatives.update(
            {
                "zi_score": (terms(z=zeta + h).value - terms(z=zeta - h).value) / (2 * h),
                "score_zi": (terms(z=zeta + h).score - terms(z=zeta - h).score) / (2 * h),
                "curvature_zi": (terms(z=zeta + h).curvature - terms(z=zeta - h).curvature)
                / (2 * h),
            }
        )
    for name, expected in derivatives.items():
        np.testing.assert_allclose(getattr(actual, name), expected, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize(
    "family", ["poisson", "nbinom1", "nbinom2", "binomial", "beta", "gaussian"]
)
def test_densities_include_normalization(family):
    eta = np.array([-0.5, 0.4, 0.8])
    phi = np.array([1.2, 2.3, 3.5])
    trials = np.ones(3)
    y = np.array([0.0, 2.0, 3.0])
    if family in {"poisson", "nbinom1", "nbinom2"}:
        mu = np.exp(eta)
        if family == "poisson":
            expected = stats.poisson.logpmf(y, mu)
        else:
            size = phi if family == "nbinom2" else mu / phi
            expected = stats.nbinom.logpmf(y, size, size / (size + mu))
    elif family == "binomial":
        trials, y = np.full(3, 5.0), np.array([0.0, 0.4, 0.8])
        expected = stats.binom.logpmf(y * trials, trials, special.expit(eta))
    elif family == "beta":
        y = np.array([0.1, 0.3, 0.8])
        mu = special.expit(eta)
        expected = stats.beta.logpdf(y, mu * phi, (1 - mu) * phi)
    else:
        y = np.array([-0.3, 0.2, 0.9])
        expected = stats.norm.logpdf(y, eta, np.sqrt(phi))
    actual = observation_terms(eta, np.log(phi), None, y, trials, np.ones(3), family)
    np.testing.assert_allclose(actual.value, -expected, atol=1e-12)


@pytest.mark.parametrize(
    "family,dispersion",
    [("nbinom1", 1e-10), ("nbinom2", 1e10), ("nbinom1", 1e-300), ("nbinom2", 1e300)],
)
def test_negative_binomial_poisson_limit(family, dispersion):
    eta = np.array([-0.5, 0.4, 0.8])
    response = np.array([0.0, 2.0, 8.0])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        actual = observation_terms(
            eta,
            np.full(3, np.log(dispersion)),
            None,
            response,
            np.ones(3),
            np.ones(3),
            family,
        )
    expected = stats.poisson.logpmf(response, np.exp(eta))
    np.testing.assert_allclose(actual.value, -expected, atol=1e-8)
    np.testing.assert_allclose(actual.score, np.exp(eta) - response, atol=1e-8)
    np.testing.assert_allclose(actual.curvature, np.exp(eta), atol=1e-8)
    np.testing.assert_allclose(actual.third, np.exp(eta), atol=1e-8)


@pytest.fixture(scope="module")
def count_data():
    rng = np.random.default_rng(683)
    g = np.repeat(np.arange(8), 12)
    x = rng.normal(size=len(g))
    b = np.array([-0.7, -0.4, -0.2, -0.1, 0.2, 0.3, 0.4, 0.7])
    return pd.DataFrame({"y": rng.poisson(np.exp(0.6 + 0.3 * x + b[g])), "x": x, "g": g})


def test_poisson_laplace_matches_independent_glmer(count_data):
    fit = glmmTMB("y~x+(1|g)", count_data, family="poisson")
    reference = glmer("y~x+(1|g)", count_data, family="poisson")
    assert fit.logLik() == pytest.approx(reference.logLik(), abs=1e-8)
    np.testing.assert_allclose(fit.beta, reference.beta, atol=1e-6)
    np.testing.assert_allclose(fit.cov_beta, reference.cov_beta, atol=1e-6)
    np.testing.assert_allclose(fit.VarCorr()["cond"]["g"], reference.VarCorr()["g"], atol=1e-6)


def test_gaussian_ml_closed_form():
    rng = np.random.default_rng(731)
    x = rng.normal(size=75)
    frame = pd.DataFrame({"y": 1 + 0.4 * x + rng.normal(0, 0.5, len(x)), "x": x})
    fit = glmmTMB("y~x", frame)
    expected = linalg.lstsq(fit.X, fit.y)[0]
    variance = np.mean((fit.y - fit.X @ expected) ** 2)
    np.testing.assert_allclose(fit.beta, expected, atol=1e-8)
    assert fit.sigma() == pytest.approx(np.sqrt(variance), abs=1e-8)
    assert fit.fixef()["disp"].iloc[0] == pytest.approx(np.log(variance) / 2, abs=1e-8)
    np.testing.assert_allclose(fit.cov_beta, variance * linalg.inv(fit.X.T @ fit.X), atol=1e-8)
    assert fit.logLik() == pytest.approx(
        float(stats.norm.logpdf(fit.y, fit.X @ expected, np.sqrt(variance)).sum()), abs=1e-9
    )


def test_gaussian_laplace_is_exact_marginal_likelihood():
    rng = np.random.default_rng(832)
    g = np.repeat(np.arange(8), 10)
    x = rng.normal(size=len(g))
    y = 1 + 0.4 * x + rng.normal(0, 0.5, len(x)) + rng.normal(0, 0.7, 8)[g]
    frame = pd.DataFrame({"y": y, "x": x, "g": g})
    fit = glmmTMB("y~x+(1|g)", frame)
    reference = lmer("y~x+(1|g)", frame, REML=False)
    assert fit.logLik() == pytest.approx(reference.logLik(), abs=1e-7)
    np.testing.assert_allclose(fit.beta, reference.beta, atol=1e-6)


def test_gaussian_random_mode_uncertainty_has_closed_form():
    rng = np.random.default_rng(371)
    groups, per_group = 7, 12
    g = np.repeat(np.arange(groups), per_group)
    y = 0.8 + rng.normal(0, 0.7, groups)[g] + rng.normal(0, 0.4, len(g))
    frame = pd.DataFrame({"y": y, "g": g})
    fit = glmmTMB("y~1+(1|g)", frame)
    variance = fit.sigma() ** 2
    random_sd = fit.theta[0]
    shrinkage = per_group * random_sd**2 / (variance + per_group * random_sd**2)
    inner_variance = variance * random_sd**2 / (variance + per_group * random_sd**2)
    means = frame.groupby("g")["y"].mean().to_numpy()
    residuals = means - fit.beta[0]
    expected = []
    for residual in residuals:
        # Balanced Gaussian random-intercept posterior means are the group
        # residual times this shrinkage factor. Differentiate that closed
        # form with respect to beta, log residual SD and random SD.
        sensitivity = 2 * shrinkage * (1 - shrinkage) * residual
        jacobian = np.array([-shrinkage, -sensitivity, sensitivity / random_sd])
        expected.append(inner_variance + jacobian @ fit.joint_cov @ jacobian)
    actual = fit.ranef()["cond"]["g"].attrs["condVar"][0, 0]
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-12)
    assert np.all(actual > inner_variance)


def test_absent_gaussian_zero_component_has_no_spurious_wald_significance():
    rng = np.random.default_rng(451)
    x = rng.normal(size=60)
    frame = pd.DataFrame({"y": 1 + 0.3 * x + rng.normal(0, 0.4, len(x)), "x": x})
    with pytest.warns(RuntimeWarning):
        fit = glmmTMB("y~x", frame, ziformula="~1")
    assert np.max(fit.predict(type="zprob")) < 1e-9
    zero_table = fit.summary().split("Zero-inflation model:", 1)[1]
    assert "nan" in zero_table
    assert "Signif. codes:" not in zero_table


def test_zero_inflation_predictions_and_components():
    rng = np.random.default_rng(182)
    x = rng.uniform(-1, 1, 180)
    y = rng.poisson(np.exp(1 + 0.3 * x))
    y[rng.random(len(x)) < special.expit(-0.6 + 0.8 * x)] = 0
    frame = pd.DataFrame({"y": y, "x": x})
    fit = glmmTMB("y~x", frame, family="poisson", ziformula="~x")
    np.testing.assert_allclose(
        fit.predict(), fit.predict(type="conditional") * (1 - fit.predict(type="zprob"))
    )
    assert list(fit.fixef()) == ["cond", "zi", "disp"]
    assert fit.joint_fixed_covariance.shape == (4, 4)
    assert np.max(np.abs(fit.joint_fixed_covariance[:2, 2:])) > 0
    assert fit.AIC() == pytest.approx(-2 * fit.logLik() + 8)
    assert "Zero-inflation model:" in fit.summary()


def test_covariate_gaussian_dispersion():
    rng = np.random.default_rng(125)
    x = rng.uniform(-1, 1, 240)
    frame = pd.DataFrame(
        {"x": x, "y": 0.6 + 0.3 * x + rng.normal(size=len(x)) * np.exp(-0.5 + 0.7 * x)}
    )
    fit = glmmTMB("y~x", frame, dispformula="~x")
    assert fit.disp_beta[1] > 0.3
    np.testing.assert_allclose(fit.predict(type="disp"), np.exp(fit.disp_X @ fit.disp_beta))
    assert fit.component_data("disp")["link"] == "log"
    assert "Dispersion model:" in fit.summary()


def test_prediction_new_levels_and_offsets(count_data):
    frame = count_data.copy()
    frame["o"] = np.linspace(-0.2, 0.2, len(frame))
    fit = glmmTMB("y~x+offset(o)+(1|g)", frame, family="poisson")
    np.testing.assert_allclose(fit.predict(frame), fit.fitted(), atol=1e-12)
    new = frame.iloc[:2].copy()
    new["g"] = 500
    with pytest.raises(ValueError, match="New group"):
        fit.predict(new)
    np.testing.assert_allclose(
        fit.predict(new, allow_new_levels=True), fit.predict(new, re_form="NA")
    )
    assert fit.ranef()["cond"]["g"].attrs["condVar"].shape == (1, 1, 8)


def test_binomial_proportion_and_cbind():
    rng = np.random.default_rng(658)
    x = rng.normal(size=120)
    n = np.full(len(x), 6)
    k = rng.binomial(n, special.expit(0.2 + 0.5 * x))
    frame = pd.DataFrame({"y": k / n, "x": x, "n": n, "k": k})
    a = glmmTMB("y~x", frame, family="binomial", weights="n")
    b = glmmTMB("cbind(k,n-k)~x", frame, family="binomial")
    np.testing.assert_allclose(a.beta, b.beta, atol=1e-10)
    assert a.logLik() == pytest.approx(b.logLik(), abs=1e-10)


def test_reject_invalid_beta_and_control(count_data):
    with pytest.raises(ValueError, match="Beta response"):
        glmmTMB("y~x", count_data, family="beta")
    with pytest.raises(NotImplementedError, match="requires"):
        glmmTMB("y~x", count_data, family="poisson", link="identity")
    with pytest.raises(NotImplementedError, match="fixed effects"):
        glmmTMB("y~x", count_data, family="poisson", ziformula="~(1|g)")
    with pytest.raises(ValueError, match="control"):
        glmmTMB("y~x", count_data, control={"magic": True})
