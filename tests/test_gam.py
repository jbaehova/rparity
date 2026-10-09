"""Independent identities for penalized fitting, covariance, and prediction."""

import json
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from scipy import linalg, optimize

from rparity.gam import gam


def sample(seed=984, n=120):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    return pd.DataFrame({"x": x, "y": 1 + np.sin(3*x) + rng.normal(scale=.3, size=n),
                         "z": rng.uniform(-1, 1, n), "g": np.repeat(np.arange(12), n//12)})


@pytest.mark.parametrize("method", ["ML", "REML", "GCV.Cp"])
def test_unpenalized_gaussian_matches_ols_and_criterion(method):
    frame = sample()
    fit = gam("y ~ x", frame, method=method)
    X = np.column_stack([np.ones(len(frame)), frame.x])
    beta = linalg.lstsq(X, frame.y)[0]
    residual = frame.y - X @ beta
    rss = float(residual @ residual)
    assert_allclose(fit.beta, beta, atol=1e-13)
    assert_allclose(fit.edf, np.ones(2), atol=1e-13)
    assert_allclose(fit.scale, rss/(len(frame)-2), atol=1e-13)
    assert_allclose(fit.cov_beta, fit.scale*linalg.inv(X.T@X), atol=1e-13)
    if method == "REML":
        expected = .5*((len(frame)-2)*(np.log(2*np.pi*rss/(len(frame)-2))+1)
                        + np.linalg.slogdet(X.T@X)[1])
    elif method == "ML":
        expected = .5*len(frame)*(np.log(2*np.pi*rss/len(frame))+1)
    else:
        expected = len(frame)*rss/(len(frame)-2)**2
    assert_allclose(fit.objective, expected, atol=1e-11)


@pytest.mark.parametrize("family", ["gaussian", "binomial", "poisson", "Gamma"])
def test_unpenalized_family_matches_independent_glm(family):
    frame = sample()
    rng = np.random.default_rng(382)
    eta = .3 + .4*frame.x.to_numpy()
    if family == "binomial":
        frame.y = rng.binomial(1, 1/(1+np.exp(-eta)))
        reference_family = sm.families.Binomial()
    elif family == "poisson":
        frame.y = rng.poisson(np.exp(eta))
        reference_family = sm.families.Poisson()
    elif family == "Gamma":
        frame.y = rng.gamma(5, np.exp(eta)/5)
        reference_family = sm.families.Gamma(link=sm.families.links.Log())
    else:
        reference_family = sm.families.Gaussian()
    fit = gam("y ~ x", frame, family=family, link="log" if family == "Gamma" else None)
    reference = sm.GLM(frame.y, sm.add_constant(frame.x), family=reference_family).fit(tol=1e-14, maxiter=1000)
    assert_allclose(fit.beta, reference.params, rtol=1e-9, atol=1e-11)
    assert_allclose(fit.fitted(), reference.fittedvalues, rtol=1e-9, atol=1e-11)


def test_fixed_penalty_matches_direct_normal_equations_and_se():
    frame = sample()
    fit = gam('y ~ s(x, bs="cr", k=8)', frame, sp=[2.5], method="REML")
    X, S = fit.X, fit.design.S[0]*2.5
    H = X.T@X+S
    expected = linalg.solve(H, X.T@fit.y)
    assert_allclose(fit.beta, expected, atol=1e-12)
    assert_allclose(fit.edf, np.diag(linalg.solve(H, X.T@X)), atol=1e-12)
    predicted = fit.predict(frame.iloc[:10], se_fit=True)
    xnew = fit.predict(frame.iloc[:10], type="lpmatrix")
    assert_allclose(predicted["fit"], xnew@expected, atol=1e-12)
    assert_allclose(predicted["se.fit"], np.sqrt(np.einsum("ij,jk,ik->i", xnew, fit.cov_beta, xnew)), atol=1e-12)
    assert_allclose(fit.vcov(freq=True), fit.scale*linalg.solve(H, X.T@X)@linalg.inv(H), atol=1e-12)


@pytest.mark.parametrize("method", ["ML", "REML", "GCV.Cp"])
def test_smoothing_selection_improves_independent_fixed_penalty_grid(method):
    frame = sample()
    formula = 'y ~ s(x, bs="cr", k=8)'
    fit = gam(formula, frame, method=method)
    grid = [gam(formula, frame, method=method, sp=[value]).objective for value in np.logspace(-3, 5, 12)]
    assert fit.objective <= min(grid)+1e-7
    assert 1 < fit.smooth_edf.iloc[0] < 7
    assert fit.converged


def test_gaussian_random_effect_reml_matches_direct_marginal_likelihood():
    frame = sample()
    effects = np.random.default_rng(96).normal(scale=1.2, size=12)
    frame.y += effects[frame.g.to_numpy()]
    frame.g = pd.Categorical(frame.g)
    fit = gam('y ~ x + s(g, bs="re")', frame, method="REML")
    fixed = fit.X[:, :2]
    random = fit.X[:, 2:]

    def objective(log_sp):
        lam = np.exp(float(log_sp))
        V = np.eye(len(frame))+random@random.T/lam
        precision = linalg.inv(V)
        information = fixed.T@precision@fixed
        beta = linalg.solve(information, fixed.T@precision@fit.y)
        residual = fit.y-fixed@beta
        rss = float(residual@precision@residual)
        df = len(frame)-2
        log_v = 2*np.log(np.diag(linalg.cholesky(V))).sum()
        log_information = 2*np.log(np.diag(linalg.cholesky(information))).sum()
        return .5*(df*(np.log(2*np.pi*rss/df)+1)+log_v+log_information)

    optimum = optimize.minimize_scalar(objective, bounds=(-12, 20), method="bounded")
    assert_allclose(fit.objective, optimum.fun, atol=1e-7)
    assert_allclose(fit.sp, np.exp(optimum.x), rtol=3e-4)


def test_response_prediction_delta_method_and_terms_sum():
    frame = sample()
    rng = np.random.default_rng(716)
    frame.y = rng.poisson(np.exp(.5+.4*np.sin(3*frame.x)))
    fit = gam('y ~ z + s(x, bs="cr", k=7)', frame, family="poisson", method="REML")
    link = fit.predict(frame.iloc[:5], se_fit=True)
    response = fit.predict(frame.iloc[:5], type="response", se_fit=True)
    assert_allclose(response["fit"], np.exp(link["fit"]))
    assert_allclose(response["se.fit"], np.exp(link["fit"])*link["se.fit"])
    terms = fit.predict(type="terms")
    assert_allclose(terms.sum(axis=1)+terms.attrs["constant"], fit.linear_predictor, atol=1e-12)


def test_weighted_binomial_cbind_equals_proportion_response():
    frame = sample()
    rng = np.random.default_rng(253)
    frame["n"] = rng.integers(2, 15, len(frame))
    frame["success"] = rng.binomial(frame.n, .5)
    frame["failure"] = frame.n-frame.success
    frame["proportion"] = frame.success/frame.n
    counted = gam('cbind(success, failure) ~ s(x, bs="cr", k=6)', frame, family="binomial")
    proportion = gam('proportion ~ s(x, bs="cr", k=6)', frame, family="binomial", weights="n")
    assert_allclose(counted.beta, proportion.beta, atol=1e-9)
    assert_allclose(counted.logLik(), proportion.logLik(), atol=1e-9)


def test_offset_and_partial_effect_data():
    frame = sample()
    frame["off"] = .2*frame.z
    with_offset = gam('y ~ s(x, bs="cr", k=6) + offset(off)', frame, sp=[2])
    adjusted = frame.copy()
    adjusted.y -= adjusted.off
    without = gam('y ~ s(x, bs="cr", k=6)', adjusted, sp=[2])
    assert_allclose(with_offset.beta, without.beta, atol=1e-11)
    predicted = with_offset.predict(frame.iloc[:4])
    assert_allclose(predicted, without.predict(frame.iloc[:4])+frame.off.iloc[:4])
    partial = with_offset.partial_effects("s(x)", n=20)
    assert len(partial) == 20
    assert {"x", "fit", "se"} <= set(partial)
    assert partial.x.is_monotonic_increasing


def test_criterion_coordinate_measure_conversion():
    fit = gam('y ~ s(x, bs="cr", k=6)', sample(), method="REML")
    transform = np.diag(np.linspace(.5, 2, len(fit.beta)))
    penalty = linalg.inv(transform).T@fit._penalty@linalg.inv(transform)
    values = linalg.eigvalsh(penalty)
    prior_values = linalg.eigvalsh(fit._penalty)
    rank = np.linalg.matrix_rank(fit._penalty)
    shift = -np.linalg.slogdet(transform)[1]-.5*(np.log(values[-rank:]).sum()-np.log(prior_values[-rank:]).sum())
    assert_allclose(fit.criterion_in_basis(transform), fit.objective+shift, atol=1e-8)


def test_rank_deficient_reml_coordinate_slice_measure():
    frame = sample()
    frame["w"] = np.random.default_rng(930).uniform(.5, 1.5, len(frame))
    fit = gam('y ~ w + s(x, by=w, bs="cr", k=6)', frame, method="REML")
    joint_null = linalg.null_space(np.vstack([fit.X, fit._penalty]), rcond=1e-12)
    assert joint_null.shape[1] == 1
    identity = np.eye(len(fit.beta))
    normal = identity[:, 1:2]
    quotient_score = fit.criterion_in_basis(identity, constraint=joint_null)
    slice_score = fit.criterion_in_basis(identity, constraint=normal)
    shift = np.log(abs((normal.T@joint_null).item()))
    assert_allclose(slice_score, quotient_score+shift, atol=1e-8)
    with pytest.raises(ValueError, match="joint design/penalty nullspace"):
        fit.criterion_in_basis(identity, constraint=identity[:, :1])


@pytest.mark.parametrize("method", ["ML", "REML"])
def test_weighted_gamma_profiles_precision_likelihood(method):
    from scipy.stats import gamma as gamma_distribution

    frame = sample()
    rng = np.random.default_rng(298)
    weights = rng.uniform(.4, 2.5, len(frame))
    mean = np.exp(.4+.3*frame.x)
    frame.y = rng.gamma(weights/.3, mean*.3/weights)
    fit = gam('y ~ s(x, bs="cr", k=6)', frame, family="Gamma", link="log",
              method=method, weights=weights, sp=[.8])
    values, vectors = linalg.eigh(fit._penalty)
    penalized = values > np.max(values)*1e-10
    random = vectors[:, penalized]
    matrix = fit._inner.observed+fit._penalty
    determinant = np.linalg.slogdet(matrix if method == "REML" else random.T@matrix@random)[1]
    prior_det = np.log(values[penalized]).sum()
    null_dim = len(values)-sum(penalized) if method == "REML" else 0

    def likelihood(log_phi):
        phi = np.exp(log_phi)
        ll = gamma_distribution.logpdf(fit.y, a=weights/phi, scale=fit.fitted_values*phi/weights).sum()
        return -ll+fit._inner.penalty/(2*phi)+.5*(determinant-prior_det-null_dim*np.log(2*np.pi*phi))

    independent = optimize.minimize_scalar(likelihood, bounds=(-8, 2), method="bounded")
    assert_allclose(np.log(fit._likelihood_scale), independent.x, atol=2e-6)
    assert_allclose(fit.objective, independent.fun, atol=1e-8)
    assert_allclose(fit.criterion_in_basis(np.eye(len(fit.beta))), fit.objective, atol=1e-8)


def test_qr_stationarity_accepts_exact_solution_and_rejects_wrong_mean():
    from rparity.gam._fit import _Family, _irls, _penalized_stationary

    frame = sample()
    X = np.column_stack([np.ones(len(frame)), frame.x, frame.z])
    root = np.diag([0., 3., 1e6])
    weights = np.ones(len(frame))
    family = _Family("gaussian", "identity")
    inner = _irls(X, frame.y.to_numpy(), weights, np.zeros(len(frame)), family,
                  root.T@root, 100, 1e-11, root)
    assert _penalized_stationary(X, frame.y.to_numpy(), weights, family, inner, root)
    wrong_beta = inner.beta+np.array([.01, 0, 0])
    wrong_mean = X@wrong_beta
    wrong = replace(inner, beta=wrong_beta, eta=wrong_mean, mu=wrong_mean,
                    deviance=float(np.sum((frame.y.to_numpy()-wrong_mean)**2)))
    assert not _penalized_stationary(X, frame.y.to_numpy(), weights, family, wrong, root)


def test_outer_stationarity_requires_score_and_positive_curvature():
    from rparity.gam._fit import _criterion_stationary

    assert _criterion_stationary(np.zeros(2), np.diag([0., 2.]), 10.)
    assert not _criterion_stationary(np.array([1e-3, 0.]), np.eye(2), 10.)
    assert not _criterion_stationary(np.zeros(2), np.diag([-1e-3, 2.]), 10.)
    assert not _criterion_stationary(np.array([np.nan]), np.ones((1, 1)), 10.)


def test_stationary_fit_retains_failed_optimizer_status(monkeypatch):
    minimize = optimize.minimize

    def stalled_status(*args, **kwargs):
        result = minimize(*args, **kwargs)
        result.success = False
        result.message = "Simulated line-search stagnation after reaching the optimum"
        return result

    reference = gam('y ~ s(x, bs="cr", k=6)', sample(), method="REML")
    monkeypatch.setattr(optimize, "minimize", stalled_status)
    with warnings.catch_warnings(record=True) as caught:
        fit = gam('y ~ s(x, bs="cr", k=6)', sample(), method="REML")
    assert fit.converged
    assert not fit._outer_result.success
    assert "Simulated" in fit._outer_result.message
    assert not any(issubclass(item.category, RuntimeWarning) for item in caught)
    assert_allclose(fit.beta, reference.beta, rtol=1e-8, atol=1e-11)
    assert_allclose(fit.objective, reference.objective, atol=1e-11)


@pytest.mark.parametrize("case_id", [406, 601, 820, 835])
def test_machine_stationary_regressions_without_r_runtime(case_id):
    # Only seeded input data are used. No R observations or optimizer results
    # supply a convergence label or a fitting value to this regression test.
    fixture = Path(__file__).parent/f"golden/gam/gam_{case_id:04d}.json"
    spec = json.loads(fixture.read_text())["spec"]
    frame = pd.DataFrame(spec["data"])
    for name in spec.get("factors", []):
        frame[name] = pd.Categorical(frame[name])
    with warnings.catch_warnings(record=True) as caught:
        fit = gam(spec["formula"], frame, family=spec["family"], link=spec.get("link"),
                  **spec.get("args", {}))
    assert fit.converged
    assert not any(issubclass(item.category, RuntimeWarning) for item in caught)
    gradient, hessian = fit.smoothing_derivatives()
    assert np.max(np.abs(gradient)) < 1e-8*max(1, abs(fit.objective))
    assert np.min(linalg.eigvalsh(hessian)) >= -1e-8*max(1, np.linalg.norm(hessian, 2))


@pytest.mark.parametrize("argument", [{"method":"invalid"}, {"family":"beta"}, {"weights":-1}, {"sp":[1,2]}, {"gamma":0}])
def test_invalid_fit_arguments(argument):
    with pytest.raises(ValueError):
        gam('y ~ s(x, bs="cr", k=6)', sample(), **argument)
