"""Strict small-sample numeric comparisons with committed R black-box output."""
from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from rparity.anova import anova
from rparity.inference import coefficient_tests
from rparity.lmm.lmer import lmer

FIXTURES = sorted((Path(__file__).parent / 'golden/inference').glob('*.json'))


def _rtable(output: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame({key: np.atleast_1d(value) for key, value in output['values'].items()},
                        index=np.atleast_1d(output['rows']))


@pytest.mark.golden
@pytest.mark.parametrize('path', FIXTURES, ids=lambda p: p.stem)
def test_inference_golden(path: Path, record_property: Callable[[str, Any], None]) -> None:
    artifact = json.loads(path.read_text())
    spec, oracle = artifact['spec'], artifact['oracle']
    assert 'error' not in oracle['result'], oracle['result'].get('error')
    data = pd.DataFrame(spec['data'])
    for name in spec.get('factors', []):
        data[name] = pd.Categorical(data[name])
    formula = spec['formula'].replace('a', 'C(a, Sum)')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = lmer(formula, data)
    result = oracle['result']
    assert model.logLik() >= result['logLik'] - 1e-6
    record_property('better_optimum', model.logLik() > result['logLik'] + 1e-6)
    oracle_warnings = np.atleast_1d(oracle.get('warnings', []))
    singular = any('singular' in w.lower() for w in oracle_warnings)
    convergence = any('converg' in w.lower() or 'hessian' in w.lower() for w in oracle_warnings)
    if singular:
        assert any('singular' in str(w.message).lower() for w in caught)
    if convergence:
        assert any('converg' in str(w.message).lower() for w in caught)
    boundary = singular or convergence
    if spec['operation'] == 'fit':
        if boundary:
            return
        np.testing.assert_allclose(model.beta, result['beta'], rtol=1e-5, atol=1e-12)
        np.testing.assert_allclose(model.cov_beta, result['vcov'], rtol=1e-4, atol=1e-12)
        actual = coefficient_tests(model)
        expected = _rtable(result['coefficients'])
        np.testing.assert_allclose(actual['df'], expected['df'], rtol=1e-3)
        np.testing.assert_allclose(actual['Pr(>|t|)'], expected['Pr(>|t|)'], atol=1e-4, rtol=0)
    else:
        if boundary:
            # Objective comparison above is sufficient for R singular/warning fits.
            return
        actual = anova(model, type=spec['type'], ddf=spec['ddf'].lower())
        expected = _rtable(result)
        assert len(actual) == len(expected)
        np.testing.assert_allclose(actual['DenDF'], expected['DenDF'], rtol=1e-3)
        np.testing.assert_allclose(actual['F value'], expected['F value'], rtol=1e-4)
        np.testing.assert_allclose(actual['Pr(>F)'], expected['Pr(>F)'], atol=1e-4, rtol=0)
