"""Representative public R inference examples, without committed package data.

Every R observation and dataset is fetched through the approved JSON oracle.
Only the ignored ``oracle/cache/acceptance-*`` files contain built-in data.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from rparity import Anova, anova, corAR1, emmeans, emtrends, gls, lmer, pairs
from rparity.emm import EmmGrid
from rparity.inference import coefficient_tests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'oracle' / 'cache'
pytestmark = [
    pytest.mark.needs_r,
    pytest.mark.skipif(shutil.which('Rscript') is None, reason='R is a development-only oracle'),
]


def _oracle(label: str, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    CACHE.mkdir(parents=True, exist_ok=True)
    source = CACHE / f'acceptance-{label}-input.json'
    destination = CACHE / f'acceptance-{label}-output.json'
    source.write_text(json.dumps({'cases': specs}))
    completed = subprocess.run(
        ['Rscript', 'oracle/run_case.R', str(source), str(destination)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if completed.returncode and "there is no package called" in completed.stderr:
        pytest.skip(completed.stderr.strip())
    assert completed.returncode == 0, completed.stderr
    outputs = json.loads(destination.read_text())
    for output in outputs:
        error = output['result'].get('error', '')
        if "there is no package called" in error or "package or namespace load failed" in error:
            pytest.skip(error)
        assert not error, error
        assert not output['warnings'], output['warnings']
    return [output['result'] for output in outputs]


def _dataset(name: str, package: str) -> pd.DataFrame:
    result = _oracle(name + '-data', [{
        'id': 'acceptance-' + name, 'operation': 'data', 'dataset': name, 'package': package,
    }])[0]
    return pd.DataFrame(result['data'])


def _table(result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame({name: np.atleast_1d(value) for name, value in result['values'].items()},
                        index=np.atleast_1d(result['rows']))


@pytest.fixture(scope='module')
def sleepstudy() -> tuple[Any, list[dict[str, Any]]]:
    data = _dataset('sleepstudy', 'lme4')
    base = {'dataset': 'sleepstudy', 'package': 'lme4', 'call': 'lmer',
            'formula': 'Reaction ~ Days + (Days | Subject)'}
    specs = [dict(base, id='acceptance-sleepstudy-fit')]
    for method in ['Satterthwaite', 'Kenward-Roger']:
        specs.append(dict(base, id='acceptance-sleepstudy-' + method,
                          operation='anova', type=3, ddf=method))
    for method in ['satterthwaite', 'kenward-roger', 'asymptotic']:
        specs.append(dict(base, id='acceptance-sleepstudy-emtrends-' + method,
                          operation='emtrends', ddf=method,
                          emm_args={'specs': '1', 'var': 'Days'}))
    return lmer(base['formula'], data), _oracle('sleepstudy-inference', specs)


def test_lmertest_sleepstudy_coefficients(sleepstudy: tuple[Any, list[dict[str, Any]]]) -> None:
    model, results = sleepstudy
    expected = _table(results[0]['coefficients'])
    actual = coefficient_tests(model)
    np.testing.assert_allclose(actual['Estimate'], expected['Estimate'], rtol=1e-5)
    np.testing.assert_allclose(actual['Std. Error'], expected['Std. Error'], rtol=1e-4)
    np.testing.assert_allclose(actual['df'], expected['df'], rtol=1e-3)
    np.testing.assert_allclose(actual['Pr(>|t|)'], expected['Pr(>|t|)'], rtol=0, atol=1e-4)


@pytest.mark.parametrize('method,index', [('satterthwaite', 1), ('kenward-roger', 2)])
def test_lmertest_sleepstudy_anova(
    sleepstudy: tuple[Any, list[dict[str, Any]]], method: str, index: int,
) -> None:
    model, results = sleepstudy
    expected = _table(results[index])
    actual = anova(model, type=3, ddf=method)
    assert actual.index.tolist() == expected.index.tolist()
    np.testing.assert_allclose(actual['F value'], expected['F value'].to_numpy(dtype=float), rtol=1e-4)
    np.testing.assert_allclose(actual['DenDF'], expected['DenDF'], rtol=1e-3)
    np.testing.assert_allclose(actual['Pr(>F)'], expected['Pr(>F)'].to_numpy(dtype=float), rtol=0, atol=1e-4)


@pytest.mark.parametrize('method,index', [
    ('satterthwaite', 3), ('kenward-roger', 4), ('asymptotic', 5),
])
def test_emmeans_sleepstudy_trends(
    sleepstudy: tuple[Any, list[dict[str, Any]]], method: str, index: int,
) -> None:
    model, results = sleepstudy
    expected = _table(results[index])
    trend = emtrends(model, '1', var='Days', lmer_df=method)
    assert isinstance(trend, EmmGrid)
    actual = trend.summary(infer=(True, True), adjust='none')
    np.testing.assert_allclose(actual['Days.trend'], expected['Days.trend'], rtol=1e-6)
    np.testing.assert_allclose(actual['SE'], expected['SE'], rtol=1e-4)
    np.testing.assert_allclose(actual['df'], expected['df'].fillna(np.inf).to_numpy(dtype=float), rtol=1e-3)
    np.testing.assert_allclose(actual['p.value'], expected['p.value'], rtol=0, atol=1e-4)


@pytest.mark.parametrize('kind', [2, 3])
def test_car_duncan_anova(kind: int) -> None:
    data = _dataset('Duncan', 'carData')
    formula = 'prestige ~ income + education'
    expected = _table(_oracle('Duncan-Anova-' + str(kind), [{
        'id': 'acceptance-Duncan-Anova', 'dataset': 'Duncan', 'package': 'carData',
        'call': 'lm', 'formula': formula, 'operation': 'Anova', 'type': kind, 'test': 'F',
    }])[0])
    actual = Anova(smf.ols(formula, data).fit(), type=kind)
    # R uses (Intercept), whereas statsmodels exposes Intercept.
    actual.index = actual.index.str.replace('Intercept', '(Intercept)', regex=False)
    assert actual.index.tolist() == expected.index.tolist()
    np.testing.assert_allclose(actual['F value'], expected['F value'].to_numpy(dtype=float), rtol=1e-4)
    np.testing.assert_allclose(actual['Sum Sq'], expected['Sum Sq'], rtol=1e-4)
    np.testing.assert_allclose(actual['Pr(>F)'], expected['Pr(>F)'].to_numpy(dtype=float), rtol=0, atol=1e-4)


@pytest.mark.parametrize('operation', ['emmeans', 'pairs'])
def test_emmeans_warpbreaks_tukey(operation: str) -> None:
    data = _dataset('warpbreaks', 'datasets')
    for name in ['wool', 'tension']:
        data[name] = pd.Categorical(data[name], categories=list(data[name].unique()))
    spec: dict[str, Any] = {
        'id': 'acceptance-warpbreaks', 'dataset': 'warpbreaks', 'package': 'datasets',
        'call': 'lm', 'formula': 'breaks ~ wool * tension', 'operation': 'emmeans',
        'emm_args': {'specs': 'tension', 'by': 'wool'},
    }
    model = smf.ols(spec['formula'], data).fit()
    grid = emmeans(model, 'tension', by='wool')
    assert isinstance(grid, EmmGrid)
    if operation == 'pairs':
        spec['contrast'] = {'method': 'pairwise', 'adjust': 'tukey'}
        grid = pairs(grid, adjust='tukey')
    expected = _table(_oracle('warpbreaks-' + operation, [spec])[0]).reset_index(drop=True)
    actual = grid.summary(infer=(True, True), adjust='tukey' if operation == 'pairs' else 'none')
    keys = ['wool', 'contrast' if operation == 'pairs' else 'tension']
    actual = actual.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    assert actual[keys].equals(expected[keys])
    estimate = 'estimate' if operation == 'pairs' else 'emmean'
    np.testing.assert_allclose(actual[estimate], expected[estimate], rtol=1e-6)
    np.testing.assert_allclose(actual['SE'], expected['SE'], rtol=1e-4)
    np.testing.assert_allclose(actual['df'], expected['df'], rtol=1e-3)
    np.testing.assert_allclose(actual['p.value'], expected['p.value'], rtol=0, atol=1e-4)


def test_nlme_orthodont_inference() -> None:
    data = _dataset('Orthodont', 'nlme')
    base = {'dataset': 'Orthodont', 'package': 'nlme', 'call': 'gls',
            'formula': 'distance ~ age',
            'correlation': {'name': 'corAR1', 'args': {'form': '~ 1 | Subject'}},
            'gls_control': {'.relStep': 1e-3}}
    results = _oracle('Orthodont-inference', [
        dict(base, id='acceptance-Orthodont-fit'),
        dict(base, id='acceptance-Orthodont-anova', operation='anova', type='marginal'),
    ])
    model = gls(base['formula'], data, correlation=corAR1(form='~ 1 | Subject'))
    actual, expected = model.anova(type='marginal'), _table(results[1])
    assert actual.index.tolist() == expected.index.tolist()
    np.testing.assert_allclose(actual['F-value'], expected['F-value'], rtol=1e-4)
    np.testing.assert_allclose(actual['p-value'], expected['p-value'], rtol=0, atol=1e-4)
    intervals = model.intervals()
    np.testing.assert_allclose(intervals['coef'], results[0]['intervals']['coef'], rtol=1e-4)
    np.testing.assert_allclose(intervals['sigma'], np.atleast_2d(results[0]['intervals']['sigma']),
                               rtol=1e-4)
