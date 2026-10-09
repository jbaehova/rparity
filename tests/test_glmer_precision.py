"""Validate Laplace derivatives independently of optimizer endpoints."""

import importlib
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import optimize

from rparity.lmm._glmer_math import score_hessian


@pytest.mark.parametrize(
    "family,link,response_kind",
    [
        ("poisson", "log", "count"),
        ("binomial", "logit", "binary"),
        ("binomial", "probit", "binary"),
        ("binomial", "cloglog", "binary"),
        ("binomial", "probit", "proportion"),
        ("binomial", "cloglog", "rounded"),
    ],
)
def test_laplace_score_matches_independent_finite_difference(
    monkeypatch, family, link, response_kind
):
    """Check determinant and implicit-mode terms away from the fitted optimum."""
    module = importlib.import_module("rparity.lmm.glmer")
    rng = np.random.default_rng(692)
    n = 48
    y = rng.poisson(2, n) if family == "poisson" else rng.binomial(1, 0.5, n)
    weights = None
    if response_kind in {"proportion", "rounded"}:
        trials = rng.integers(3, 9, n)
        y = rng.binomial(trials, 0.5) / trials
        weights = trials.astype(float)
        if response_kind == "rounded":
            weights += 0.2
    data = pd.DataFrame({"y": y, "x": rng.normal(size=n), "g": np.repeat(range(8), 6)})
    original = optimize.minimize
    checked = []

    def checked_minimize(function, start, *args, **kwargs):
        jac = kwargs.get("jac")
        if callable(jac):
            point = np.asarray(start).copy()
            point[0] = 0.7
            point[1] = -0.1
            point[2] = 0.4
            h = 3e-4
            directions = np.eye(len(point)) * h
            numerical = np.array(
                [
                    (
                        -function(point + 2 * direction)
                        + 8 * function(point + direction)
                        - 8 * function(point - direction)
                        + function(point - 2 * direction)
                    )
                    / (12 * h)
                    for direction in directions
                ]
            )
            np.testing.assert_allclose(jac(point), numerical, rtol=1e-6, atol=2e-7)
            checked.append(True)
        return original(function, start, *args, **kwargs)

    monkeypatch.setattr(module.optimize, "minimize", checked_minimize)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module.glmer("y ~ x + (1 | g)", data, family=family, link=link, weights=weights)
    assert checked


def test_score_hessian_preserves_small_cross_curvature():
    """A tiny cross term survives even when the likelihood has a large constant."""
    curvature = np.array([[2.0, 3e-8], [3e-8, 4.0]])

    def gradient(x):
        return curvature @ x + 0.2 * x**3

    point = np.array([0.3, -0.4])
    expected = curvature + np.diag(0.6 * point**2)
    np.testing.assert_allclose(score_hessian(gradient, point), expected, rtol=1e-6, atol=1e-12)
