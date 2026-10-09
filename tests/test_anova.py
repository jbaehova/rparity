"""Term hypothesis invariants, statsmodels parity and ML refit semantics."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from rparity.anova import Anova, anova
from rparity.lmm.lmer import lmer


def factorial_data() -> pd.DataFrame:
    rng = np.random.default_rng(518)
    a = np.repeat(['a', 'b', 'c'], [23, 31, 18])
    x = rng.normal(size=len(a))
    y = .4*x + (a == 'b') * .8 + (a == 'c') * .3*x + rng.normal(size=len(a))
    return pd.DataFrame({'y': y, 'a': a, 'x': x, 'g': np.arange(len(a)) % 12})


@pytest.mark.parametrize('kind', [2, 3])
def test_ols_term_tests_match_statsmodels(kind: int) -> None:
    model = smf.ols('y ~ C(a, Sum)*x', factorial_data()).fit()
    actual = Anova(model, type=kind)
    expected = sm.stats.anova_lm(model, typ=kind)
    for term in expected.index[:-1]:
        assert actual.loc[term, 'F value'] == pytest.approx(expected.loc[term, 'F'], rel=1e-10)
        assert actual.loc[term, 'Sum Sq'] == pytest.approx(expected.loc[term, 'sum_sq'], rel=1e-10)
        assert actual.loc[term, 'Pr(>F)'] == pytest.approx(expected.loc[term, 'PR(>F)'], abs=1e-12)


def test_glm_likelihood_ratio_and_wald() -> None:
    data = factorial_data()
    rng = np.random.default_rng(99)
    data['y'] = rng.poisson(np.exp(.5 + .2*data.x))
    model = smf.glm('y ~ x', data, family=sm.families.Poisson()).fit()
    reduced = smf.glm('y ~ 1', data, family=sm.families.Poisson()).fit()
    table = Anova(model, test_statistic='LR')
    assert table.loc['x', 'LR Chisq'] == pytest.approx(reduced.deviance - model.deviance)
    wald = Anova(model, test_statistic='Wald')
    assert wald.loc['x', 'Chisq'] == pytest.approx(model.tvalues['x']**2)
    f = Anova(model, test_statistic='F')
    assert f.loc['x', 'F value'] == pytest.approx((reduced.deviance-model.deviance)
                                           / (model.pearson_chi2/model.df_resid))


def test_lmer_type_i_ii_iii_and_comparison_ml_refit() -> None:
    data = factorial_data()
    data['y'] += np.random.default_rng(90).normal(scale=2, size=12)[data.g]
    model = lmer('y ~ C(a, Sum)*x + (1|g)', data)
    for kind in (1, 2, 3):
        table = anova(model, type=kind)
        assert set(table.index) == {'C(a, Sum)', 'x', 'C(a, Sum):x'}
        assert np.all(table['NumDF'] > 0)
        assert np.all(table['DenDF'] > 0)
    table = Anova(model, type=3, test_statistic='F')
    assert 'Intercept' in table.index
    reduced = lmer('y ~ x + (1|g)', data)
    with pytest.warns(UserWarning, match='Refitting REML'):
        comparison = anova(reduced, model)
    a, b = reduced.refit(reml=False), model.refit(reml=False)
    assert comparison.iloc[-1]['Chisq'] == pytest.approx(2*(b.logLik()-a.logLik()), rel=1e-7)
    assert comparison.iloc[-1]['Df'] == 4


def test_invalid_type() -> None:
    with pytest.raises(ValueError, match='Type II'):
        Anova(smf.ols('y~x', factorial_data()).fit(), type=1)


def test_lmertest_type_iii_is_invariant_to_factor_coding() -> None:
    data = factorial_data()
    data['y'] += np.random.default_rng(90).normal(scale=2, size=12)[data.g]
    treatment = lmer('y ~ a*x + (1|g)', data)
    sums = lmer('y ~ C(a, Sum)*x + (1|g)', data)
    left, right = anova(treatment, type=3), anova(sums, type=3)
    np.testing.assert_allclose(left['F value'], right['F value'], rtol=1e-6)
    np.testing.assert_allclose(left['DenDF'], right['DenDF'], rtol=1e-5)
    # car Type III keeps the user's coefficient coding, so the x hypothesis differs.
    assert not np.isclose(Anova(treatment, type=3).loc['x', 'Chisq'],
                          Anova(sums, type=3).loc['x', 'Chisq'])


def test_public_anova_dispatches_gls_method() -> None:
    from rparity.gls import gls

    model = gls('y~x', factorial_data())
    pd.testing.assert_frame_equal(anova(model), model.anova())
    pd.testing.assert_frame_equal(anova(model, type='marginal'), model.anova(type='marginal'))
    pd.testing.assert_frame_equal(anova(model, type=3), model.anova(type='marginal'))
    other = gls('y~a+x', factorial_data())
    with pytest.warns(UserWarning, match='REML likelihoods'):
        actual = anova(model, other)
    with pytest.warns(UserWarning, match='REML likelihoods'):
        expected = model.anova(other)
    pd.testing.assert_frame_equal(actual, expected)


def test_anova_negative_contracts() -> None:
    model = smf.ols('y~x', factorial_data()).fit()
    with pytest.raises(ValueError, match='At least one model'):
        anova()
    with pytest.raises(ValueError, match='type must'):
        Anova(model, type=4)
    with pytest.raises(ValueError, match='test_statistic'):
        Anova(model, test_statistic='z')
    data = factorial_data()
    mixed = lmer('y~x+(1|g)', data)
    with pytest.raises(ValueError, match='Term-wise LR'):
        Anova(mixed, test_statistic='LR')
    changed = data.copy()
    changed['y'] += 1
    second = lmer('y~x+(1|g)', changed)
    with pytest.raises(ValueError, match='same response'):
        anova(mixed, second, refit=False)
    aliased = smf.ols('y~x+I(2*x)', data).fit()
    with pytest.raises(ValueError, match='non-aliased'):
        Anova(aliased, type=3)


def test_glmer_kr_f_is_rejected_like_car() -> None:
    from rparity.lmm.glmer import glmer

    data = factorial_data()
    data['y'] = np.random.default_rng(234).poisson(np.exp(.2+.1*data['x']))
    model = glmer('y~x+(1|g)', data, family='poisson')
    with pytest.raises(ValueError, match='Gaussian linear mixed model'):
        Anova(model, test_statistic='F')
