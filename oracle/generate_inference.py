"""Generate independent seeded numeric Satterthwaite and KR black-box fixtures."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def case_spec(index: int) -> dict[str, Any]:
    """Build a varying mixed-model design and an explicit inference request."""
    rng = np.random.default_rng(711000 + index // 3)
    variant = index // 3 % 6
    ng = int(rng.integers(8, 14))
    sizes = rng.integers(4, 8, size=ng)
    g = np.repeat(np.arange(ng), sizes)
    n = len(g)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    a = np.array(['a', 'b', 'c'])[np.arange(n) % 3]
    h = np.arange(n) % 7
    intercepts = rng.normal(scale=1.0, size=ng)
    slopes = .25 * intercepts + rng.normal(scale=.35, size=ng)
    y = 1.0 + .35*x + .2*z + (a == 'b')*.45 + (a == 'c')*.25*x
    y += intercepts[g] + rng.normal(scale=.7, size=n)
    random = '(1|g)'
    if variant in (1, 2):
        random = '(x|g)' if variant == 1 else '(x||g)'
        y += slopes[g]*x
    if variant == 3:
        random = '(1|g)+(1|h)'
        y += rng.normal(scale=.5, size=7)[h]
    if variant == 4:
        random = '(0+x|g)'
        y -= intercepts[g]
        y += slopes[g]*x
    fixed = 'a*x+z' if variant in (0, 1, 2, 3) else 'x+z'
    options = index % 3
    operation = 'fit' if options == 0 else 'anova'
    spec: dict[str, Any] = {
        'id': f'inference_{index:04}', 'call': 'lmer',
        'formula': f'y~{fixed}+{random}',
        'data': {'y': y.tolist(), 'x': x.tolist(), 'z': z.tolist(),
                 'a': a.tolist(), 'g': g.tolist(), 'h': h.tolist()},
        'factors': ['a', 'g', 'h'], 'contrasts': 'sum', 'operation': operation,
        'args': {'REML': True}, 'type': (index // 18) % 3 + 1,
        'ddf': 'Kenward-Roger' if options == 2 else 'Satterthwaite',
    }
    return spec


def generate(count: int = 600, batch_size: int = 50) -> None:
    """Run only the approved R oracle and preserve every output, including errors."""
    destination = ROOT / 'tests/golden/inference'
    destination.mkdir(parents=True, exist_ok=True)
    cache = ROOT / 'oracle/cache'
    cache.mkdir(parents=True, exist_ok=True)
    for begin in range(0, count, batch_size):
        specs = [case_spec(i) for i in range(begin, min(begin + batch_size, count))]
        request = cache / 'inference_request.json'
        response = cache / 'inference_response.json'
        request.write_text(json.dumps({'cases': specs}))
        subprocess.run(['Rscript', str(ROOT / 'oracle/run_case.R'), str(request), str(response)],
                       check=True)
        outputs = json.loads(response.read_text())
        for spec, oracle in zip(specs, outputs, strict=True):
            option = f"lmerTest / {spec['ddf']} / {spec['operation']} / Type {spec['type']}"
            artifact = {'id': spec['id'], 'module': 'inference', 'option': option,
                        'spec': spec, 'oracle': oracle}
            (destination / f"{spec['id']}.json").write_text(json.dumps(artifact, indent=2)+'\n')
        print(f'inference: persisted {min(begin + batch_size, count)}/{count}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=600)
    parser.add_argument('--batch-size', type=int, default=50)
    args = parser.parse_args()
    generate(args.count, args.batch_size)
