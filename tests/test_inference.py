"""Analytic small-sample limits and mixed-model inference invariants."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from rparity.inference import adjusted_covariance, coefficient_tests, contrast_df, contrast_test
from rparity.lmm.lmer import lmer


def gaussian_linear_model() -> SimpleNamespace:
    n = 24
    x = np.column_stack((np.ones(n), np.linspace(-1, 1, n), np.sin(np.arange(n))))
    sigma = 1.7
    df = n - x.shape[1]
    inv_xtx = np.linalg.inv(x.T @ x)
    return SimpleNamespace(
        X=x, beta=np.array([1.0, .7, -.2]), cov_beta=sigma**2 * inv_xtx,
        reml=True, variance_params=np.array([sigma]), coef_names=['Intercept', 'x', 'z'],
        marginal_covariance_at=lambda t: np.eye(n) * t[0]**2,
        covariance_at=lambda t: inv_xtx * t[0]**2,
        covariance_components=lambda: [np.eye(n)],
        variance_objective=lambda t: 2 * df * np.log(t[0]) + df * sigma**2 / t[0]**2,
    )


def test_kr_and_satterthwaite_reproduce_exact_linear_model_df() -> None:
    m = gaussian_linear_model()
    for contrast in ([0, 1, 0], [[0, 1, 0], [0, 0, 1]]):
        assert contrast_df(m, contrast) == pytest.approx(21, rel=2e-6)
        assert contrast_df(m, contrast, 'kenward-roger') == pytest.approx(21, rel=2e-10)
    np.testing.assert_allclose(adjusted_covariance(m), m.cov_beta, atol=1e-12)
    result = contrast_test(m, [[0, 1, 0], [0, 0, 1]], method='kenward-roger')
    assert result.scale == pytest.approx(1)
    assert result.pvalue == pytest.approx(stats.f.sf(result.statistic, 2, 21))


def test_balanced_random_intercept_within_group_slope_exact_df() -> None:
    rng = np.random.default_rng(46)
    g = np.repeat(np.arange(12), 5)
    x = np.tile(np.arange(5.0), 12)
    data = pd.DataFrame({'y': 1 + 2*x + rng.normal(0, 1, 12)[g]
                         + rng.normal(0, .6, 60), 'x': x, 'g': g})
    m = lmer('y ~ x + (1|g)', data)
    assert contrast_df(m, [0, 1]) == pytest.approx(47, rel=1e-5)
    assert contrast_df(m, [0, 1], 'kenward-roger') == pytest.approx(47, rel=1e-8)
    assert list(coefficient_tests(m).columns) == ['Estimate', 'Std. Error', 'df',
                                                't value', 'Pr(>|t|)']


def test_kr_changes_unbalanced_mixed_model_covariance() -> None:
    rng = np.random.default_rng(822)
    g = np.repeat(np.arange(10), np.arange(3, 13))
    x = rng.normal(size=len(g))
    data = pd.DataFrame({'y': .3*x + rng.normal(size=10)[g] + rng.normal(size=len(g)),
                         'x': x, 'g': g})
    m = lmer('y ~ x + (1|g)', data)
    correction = adjusted_covariance(m) - m.cov_beta
    assert np.trace(correction) > 0
    assert np.linalg.eigvalsh(correction).min() > -1e-10
    assert np.isfinite(contrast_df(m, [0, 1], 'kenward-roger'))
    with pytest.raises(ValueError, match='REML'):
        adjusted_covariance(m.refit(reml=False))


def test_invalid_df_method_and_zero_contrast() -> None:
    m = gaussian_linear_model()
    with pytest.raises(ValueError, match='Unknown'):
        contrast_df(m, [0, 1, 0], 'unsupported')
    with pytest.raises(ValueError, match='zero'):
        contrast_test(m, [0, 0, 0])
    assert contrast_df(m, [0, 1, 0], 'asymptotic') == np.inf
