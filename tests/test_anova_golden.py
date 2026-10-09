"""Committed car ANOVA tables with the Stage 1 numerical contract."""
from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from rparity.anova import Anova, anova
from rparity.lmm.glmer import glmer
from rparity.lmm.lmer import lmer

FIXTURES = sorted((Path(__file__).parent / 'golden/anova').glob('*.json'))


def _table(output: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame({key: np.atleast_1d(value) for key, value in output['values'].items()},
                        index=np.atleast_1d(output['rows']))


def _formula(formula: str, sums: bool) -> str:
    return formula.replace('a', 'C(a, Sum)') if sums else formula


def _name(name: str) -> str:
    return name.replace('C(a, Sum)', 'a').replace('(Intercept)', 'Intercept')


@pytest.mark.golden
@pytest.mark.parametrize('path', FIXTURES, ids=lambda p: p.stem)
def test_anova_golden(path: Path, record_property: Callable[[str, Any], None]) -> None:
    artifact = json.loads(path.read_text())
    spec, oracle = artifact['spec'], artifact['oracle']
    result = oracle['result']
    assert 'error' not in result, result.get('error')
    data = pd.DataFrame(spec['data'])
    for name in spec.get('factors', []):
        data[name] = pd.Categorical(data[name])
    formula = _formula(spec['formula'], spec['contrasts'] == 'sum')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if spec['operation'] == 'compare':
            fits = [lmer(_formula(item['formula'], spec['contrasts'] == 'sum'), data,
                         REML=item['args']['REML']) for item in spec['models']]
            actual = anova(*fits, refit=spec['refit'])
            expected = _table(result)
            assert np.all(actual['logLik'].to_numpy() >= expected['logLik'].to_numpy()-1e-6)
            assert list(actual.columns) == list(expected.columns)
            record_property('better_optimum', bool(np.any(actual['logLik'].to_numpy() > expected['logLik'].to_numpy()+1e-6)))
            np.testing.assert_allclose(actual.to_numpy(float), expected.to_numpy(float), rtol=1e-4, atol=1e-6, equal_nan=True)
            if spec['models'][0]['args']['REML']:
                assert any('Refitting REML' in str(w.message) for w in caught)
            return
        if spec['call'] == 'lm':
            model = smf.ols(formula, data).fit()
        elif spec['call'] == 'glm':
            family = sm.families.Poisson() if spec['family'] == 'poisson' else sm.families.Binomial()
            model = smf.glm(formula, data, family=family).fit(maxiter=1000, tol=1e-12)
        elif spec['call'] == 'lmer':
            model = lmer(formula, data, REML=spec['args']['REML'])
        else:
            model = glmer(formula, data, family=spec['family'])
        actual = Anova(model, type=spec['type'], test_statistic=spec['test'])
    ll = float(model.logLik() if spec['call'] in ('lmer', 'glmer') else model.llf)
    assert ll >= result['logLik'] - 1e-6
    record_property('better_optimum', ll > result['logLik'] + 1e-6)
    oracle_warnings = np.atleast_1d(oracle.get('warnings', []))
    singular = any('singular' in str(w).lower() for w in oracle_warnings)
    convergence = any('converg' in str(w).lower() or 'hessian' in str(w).lower()
                      for w in oracle_warnings)
    if singular:
        assert any('singular' in str(w.message).lower() for w in caught)
    if convergence:
        assert any('converg' in str(w.message).lower() for w in caught)
    if singular or convergence:
        return
    expected = _table(result)
    actual.index = [_name(str(name)) for name in actual.index]
    expected.index = [_name(str(name)) for name in expected.index]
    assert set(actual.index) == set(expected.index)
    assert list(actual.columns) == list(expected.columns)
    actual = actual.loc[expected.index]
    for column in expected.columns:
        if column.startswith('Pr('):
            np.testing.assert_allclose(actual[column], expected[column].astype(float), atol=1e-4, rtol=0, equal_nan=True)
        else:
            np.testing.assert_allclose(actual[column], expected[column].astype(float), rtol=1e-3 if column == 'Df.res' else 1e-4, atol=1e-10, equal_nan=True)
