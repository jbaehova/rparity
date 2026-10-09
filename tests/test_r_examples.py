"""Representative built-in datasets stay in an ignored development cache."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rparity import corAR1, glmer, gls, lmer

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'oracle' / 'cache'


@pytest.mark.needs_r
@pytest.mark.skipif(shutil.which('Rscript') is None, reason='R is a development-only oracle')
@pytest.mark.parametrize('dataset,package,call,formula', [
    ('sleepstudy', 'lme4', 'lmer', 'Reaction ~ Days + (Days | Subject)'),
    ('cbpp', 'lme4', 'glmer', 'cbind(incidence, size-incidence) ~ period + (1 | herd)'),
    ('Orthodont', 'nlme', 'gls', 'distance ~ age'),
])
def test_builtin_example(dataset: str, package: str, call: str, formula: str) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    source = {'id':dataset+'_data', 'operation':'data', 'dataset':dataset, 'package':package,
              'cache_csv': str(Path('oracle/cache') / (dataset + '.csv'))}
    input_path = CACHE / (dataset+'-data-spec.json')
    output_path = CACHE / (dataset+'-data-result.json')
    input_path.write_text(json.dumps(source))
    subprocess.run(['Rscript', 'oracle/run_case.R', str(input_path), str(output_path)], cwd=ROOT, check=True, capture_output=True)
    observation = json.loads(output_path.read_text())[0]['result']
    assert 'error' not in observation, observation.get('error')
    frame = pd.DataFrame(observation['data'])
    if dataset == 'cbpp':
        frame['failure'] = frame['size'] - frame['incidence']
        frame['period'] = pd.Categorical(frame['period'], categories=sorted(frame['period'].unique()))
        frame.to_csv(CACHE / 'cbpp.csv', index=False)
    spec = {'id':dataset+'_fit', 'call':call, 'formula':formula, 'csv':f'oracle/cache/{dataset}.csv'}
    if call=='gls':
        spec['correlation'] = {'name':'corAR1', 'args':{'form':'~ 1 | Subject'}}
    if call=='glmer':
        spec['family']='binomial'
        spec['factors']=['period']
    input_path = CACHE / (dataset+'-fit-spec.json')
    output_path = CACHE / (dataset+'-fit-result.json')
    input_path.write_text(json.dumps(spec))
    subprocess.run(['Rscript', 'oracle/run_case.R', str(input_path), str(output_path)], cwd=ROOT, check=True, capture_output=True)
    result = json.loads(output_path.read_text())[0]['result']
    assert 'error' not in result, result.get('error')
    if call=='lmer':
        model = lmer(formula, frame)
    elif call=='glmer':
        model = glmer(formula, frame)
    else:
        model = gls(formula, frame, correlation=corAR1(form='~ 1 | Subject'))
    assert model.logLik() >= result['logLik'] - 1e-6
    np.testing.assert_allclose(model.beta, result['beta'], rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(model.cov_beta, result['vcov'], rtol=1e-4, atol=1e-10)
    np.testing.assert_allclose(model.fitted(), result['fitted'], rtol=1e-6, atol=1e-10)
