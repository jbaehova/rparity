"""Fit-time benchmark against the same-machine, development-only R oracle."""
from __future__ import annotations

import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd

from rparity import corAR1, glmer, gls, lmer

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / 'development/oracle/cache'


def fit(spec: dict[str, Any]) -> Any:
    data = pd.DataFrame(spec['data'])
    for column in spec.get('factors', []):
        data[column] = pd.Categorical(data[column])
    if spec['call'] == 'lmer':
        return lmer(spec['formula'],data,contrasts=spec.get('contrasts','treatment'),
                    weights=spec.get('weights'),offset=spec.get('offset'),**spec.get('args',{}))
    if spec['call']=='glmer':
        return glmer(spec['formula'],data,family=spec['family'],link=spec['link'],
                     weights=spec.get('weights'),offset=spec.get('offset'))
    return gls(spec['formula'],data,correlation=corAR1(**spec['correlation']['args']),**spec['args'])


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    selected=[]
    candidates = [('lmer','lmer_0000'),('lmer','lmer_0001'),('glmer','glmer_0000'),
                  ('glmer','glmer_0003'),('gls','gls_0001')]
    for module,name in candidates:
        case=json.loads((ROOT/f'tests/golden/{module}/{name}.json').read_text())
        selected.append((case['option'],case['spec']))
    specs=[]
    for i,(_,spec) in enumerate(selected):
        specs.append(dict(spec,id=f'warmup_{i}',benchmark=True))
    for repeat in range(3):
        for i,(_,spec) in enumerate(selected):
            specs.append(dict(spec,id=f'benchmark_{i}_{repeat}',benchmark=True))
    input_file=CACHE/'benchmark-input.json'
    output_file=CACHE/'benchmark-output.json'
    input_file.write_text(json.dumps({'cases':specs}))
    subprocess.run(['Rscript','development/oracle/run_case.R',str(input_file),str(output_file)],cwd=ROOT,check=True)
    results={result['id']:result for result in json.loads(output_file.read_text())}
    report=[]
    for i,(option,spec) in enumerate(selected):
        fit(spec)
        times=[]
        for _ in range(3):
            start=time.perf_counter()
            fit(spec)
            times.append(time.perf_counter()-start)
        r_times=[]
        for repeat in range(3):
            observation=results[f'benchmark_{i}_{repeat}']['result']
            if 'error' in observation:
                raise RuntimeError(observation['error'])
            r_times.append(observation['fit_seconds'])
        report.append({'option':option,'observations':len(next(iter(spec['data'].values()))),
                       'python_seconds':statistics.median(times),'r_seconds':statistics.median(r_times)})
    output=ROOT/'development/reports/benchmark.json'
    output.write_text(json.dumps({'repetitions':3,'statistic':'median',
        'scope':'warm fit calls on the same machine, excluding R process startup',
        'models':report},indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
