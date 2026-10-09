"""Seeded car ANOVA and likelihood-ratio black-box fixtures."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def case_spec(index: int) -> dict[str, Any]:
    """Generate unbalanced interactions across model classes and contrast codings."""
    rng = np.random.default_rng(915000 + index)
    group = index // 50
    ng = int(rng.integers(9, 14))
    g = np.repeat(np.arange(ng), rng.integers(6, 11, size=ng))
    n = len(g)
    x, z = rng.normal(size=(2, n))
    a = rng.choice(['a', 'b', 'c'], size=n, p=[.25, .45, .30])
    eta = .3 + .25*x + .18*z + .3*(a == 'b') - .15*(a == 'c') + .15*x*(a == 'c')
    call = 'lm' if group < 4 else 'glm' if group < 8 else 'lmer' if group < 10 else 'glmer' if group in (10, 13) else 'lmer'
    family = 'binomial' if group in (6, 7) or (group in (10, 13) and index % 2) else 'poisson'
    if call in ('lmer', 'glmer'):
        eta += rng.normal(scale=.7, size=ng)[g]
    if call in ('lm', 'lmer'):
        y = eta + rng.normal(scale=.7, size=n)
    elif family == 'poisson':
        y = rng.poisson(np.exp(eta))
    else:
        y = rng.binomial(1, 1/(1+np.exp(-eta)))
    fixed = 'a*x+z'
    formula = 'y~'+fixed+ ('+(1|g)' if call in ('lmer', 'glmer') else '')
    spec: dict[str, Any] = {
        'id': f'anova_{index:04}', 'call': call, 'formula': formula,
        'data': {'y': y.tolist(), 'x': x.tolist(), 'z': z.tolist(), 'a': a.tolist(), 'g': g.tolist()},
        'factors': ['a', 'g'], 'contrasts': 'sum' if index % 2 else 'treatment',
        'operation': 'Anova', 'type': 2 if group % 2 == 0 else 3,
        'test': 'F' if call == 'lm' else ['LR', 'Wald', 'F'][index % 3] if call == 'glm' else 'F' if group == 9 else 'Chisq',
        'family': family, 'args': {'REML': True} if call == 'lmer' else {},
    }
    if group == 12:
        spec['type'] = 2 if index % 2 else 3
        spec['test'] = 'F' if index % 2 else 'Chisq'
    if group == 11:
        spec['operation'] = 'compare'
        spec['models'] = [{'call': 'lmer', 'formula': 'y~x+z+(1|g)', 'args': {'REML': index % 2 == 0}},
                          {'call': 'lmer', 'formula': formula, 'args': {'REML': index % 2 == 0}}]
        spec['refit'] = True
    return spec


def generate(count: int = 700, batch_size: int = 25, start: int = 0) -> None:
    """Persist public oracle output without filtering warnings or unsuccessful fits."""
    destination = ROOT / 'tests/golden/anova'
    destination.mkdir(parents=True, exist_ok=True)
    cache = ROOT / 'development/oracle/cache'
    cache.mkdir(parents=True, exist_ok=True)
    for begin in range(start, count, batch_size):
        specs = [case_spec(i) for i in range(begin, min(begin + batch_size, count))]
        request, response = cache / 'anova_request.json', cache / 'anova_response.json'
        request.write_text(json.dumps({'cases': specs}))
        subprocess.run(['Rscript', str(ROOT / 'development/oracle/run_case.R'), str(request), str(response)], check=True)
        outputs = json.loads(response.read_text())
        for spec, oracle in zip(specs, outputs, strict=True):
            option = f"car::Anova / {spec['call']} / Type {spec['type']} / {spec['test']} / {spec['contrasts']}"
            if spec['operation'] == 'compare':
                option = 'anova / lmer / ML model comparison'
            artifact = {'id': spec['id'], 'module': 'anova', 'option': option, 'spec': spec, 'oracle': oracle}
            (destination / f"{spec['id']}.json").write_text(json.dumps(artifact, indent=2)+'\n')
        print(f'anova: persisted {min(begin + batch_size, count)}/{count}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=700)
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--start', type=int, default=0)
    args = parser.parse_args()
    generate(args.count, args.batch_size, args.start)
