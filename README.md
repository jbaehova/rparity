![Status: Stage 1 complete (v0.1.0)](https://img.shields.io/badge/Status-Stage%201%20complete%20%28v0.1.0%29-green)

# rparity

R-grade mixed models, marginal means, GAMs and meta-analysis in pure Python, verified against R, no R required.

Stage 1 is complete in v0.1.0. The synthetic corpus passes 3,796/3,856 cases (98.44%); 60 documented differences remain counted as failures. See the [Stage 1 report](reports/STAGE_1_REPORT.md), [validation coverage](docs/coverage.md), and [documentation site](https://jbaehova.github.io/rparity/).

## Installation

Python 3.11 or later is required. Download the wheel from the [GitHub release](https://github.com/jbaehova/rparity/releases/tag/v0.1.0), or build from this checkout with `uv build`,
then install its wheel into your Python environment:

```sh
python -m pip install dist/rparity-0.1.0-py3-none-any.whl
```

NumPy and SciPy handle numerical work. pandas, formulaic and statsmodels provide data and model support.
Polars input is supported when Polars is already available. No R or rpy2 is
required for fitting, inference, or prediction.

## Quick start

```python
import numpy as np
import pandas as pd

from rparity import Anova, emmeans, lmer, pairs

rng = np.random.default_rng(42)
subject = np.repeat(np.arange(12), 6)
days = np.tile(np.arange(6), 12)
intercepts = rng.normal(0, 12, 12)
slopes = rng.normal(0, 2, 12)
data = pd.DataFrame({
    "Reaction": 250 + 8 * days + intercepts[subject]
                + slopes[subject] * days + rng.normal(0, 5, len(days)),
    "Days": days,
    "Subject": subject.astype(str),
})
model = lmer("Reaction ~ Days + (Days | Subject)", data=data)
print(model.summary())
print(Anova(model, type=3))
means = emmeans(model, "Days", at={"Days": [0, 5]})
print(pairs(means, adjust="tukey").summary())
```

The same example is available in `examples/quickstart.py`. The generated data
is synthetic and does not require a package example dataset.

## Validation

R is used as a black-box development oracle through `oracle/run_case.R`.
Synthetic inputs and their numerical observations are stored in `tests/golden/`.
The checks apply separate tolerances to likelihoods, coefficients, covariance,
degrees of freedom, marginal means, and p-values. Boundary or convergence
warnings have an explicit likelihood-only policy and warning checks.

See [option-level coverage](docs/coverage.md), [clean-room records](CLEANROOM.md),
and [algorithm references](REFERENCES.md). Built-in example data stays in an
ignored development cache; its tests are marked `needs_r`.

## R migration

- [lme4 and lmerTest](docs/migration/lme4.md)
- [emmeans](docs/migration/emmeans.md)
- [car](docs/migration/car.md)
- [nlme](docs/migration/nlme.md)

Type III tests depend on contrasts. Use sum contrasts for main effects in
interaction models, including when fitting statsmodels formulas.

## Development

```sh
uv sync
uv run pytest
uv run ruff check
uv run mypy src
uv run python -m rparity._coverage
uv run mkdocs build --strict
uv build
```

CI runs the tests without R on Linux and macOS. Documentation deploys through
GitHub Pages when `main` is pushed. PyPI publication is performed by a person.
