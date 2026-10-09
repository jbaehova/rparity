![Status: Stage 2 complete](https://img.shields.io/badge/Status-Stage%202%20complete-brightgreen)

# rparity

R-style mixed models, GAMs and marginal means in Python, verified against R,
with no runtime R dependency.

Stage 2 is complete in v0.2.0. Its fixed synthetic corpus passes 2,356/2,400
cases (98.17%): 1,184/1,200 GAMs and 1,172/1,200 extended mixed models.
The 16 GAM and 28 mixed-model differences remain counted as failures. See the
[Stage 2 report](reports/STAGE_2_REPORT.md) and
[case-level failure evidence](reports/STAGE_2_FAILURES.json).

Stage 1 retains 3,799/3,856 passes (98.52%) and 57 documented differences,
with no new regression. Three precision fixes are described in the
[refinement report](reports/PARITY_REFINEMENT.md). The immutable v0.1.0 release
retains its original 3,796 passes and 60 differences. See the
[Stage 1 report](reports/STAGE_1_REPORT.md),
[validation coverage](docs/coverage.md), and
[documentation site](https://jbaehova.github.io/rparity/).

## Installation

Python 3.11 or later is required. Download a compatible platform wheel from the
[GitHub release](https://github.com/jbaehova/rparity/releases/tag/v0.2.0),
or build from this checkout with `uv build`,
then install its wheel into your Python environment:

```sh
python -m pip install dist/rparity-0.2.0-*.whl
```

NumPy and SciPy handle numerical work. pandas, formulaic and statsmodels provide data and model support.
Polars input is supported when Polars is already available. No R or rpy2 is
required for fitting, inference, or prediction.

Platform wheels bundle the fixed OpenBLAS backend used for cubic shrinkage
penalties. It is a build dependency and does not add a sixth Python runtime
dependency. rparity's own code is MIT licensed; bundled native libraries retain
the notices in [LICENSES](LICENSES/README.md). Clean wheel execution has been
verified on macOS ARM64. Other platforms require their own execution check.

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

## GAM and distributional mixed models

```python
import numpy as np
import pandas as pd

from rparity import gam, glmmTMB

rng = np.random.default_rng(17)
group = np.repeat(np.arange(12), 10)
x = rng.uniform(-1, 1, len(group))
intercepts = rng.normal(0, 0.4, 12)
data = pd.DataFrame({
    "x": x,
    "group": group.astype(str),
    "y": np.sin(3 * x) + rng.normal(0, 0.2, len(x)),
    "count": rng.poisson(np.exp(0.4 + 0.6 * x + intercepts[group])),
})
smooth = gam("y ~ s(x,bs='cr',k=8)", data, method="REML")
print(smooth.summary_tables()["smooth"])
print(smooth.predict(data.iloc[:5], type="response", se_fit=True))

mixed = glmmTMB("count ~ x+(1|group)", data, family="poisson")
print(mixed.fixef()["cond"])
print(mixed.predict(data.iloc[:5], type="response", re_form="NA"))
```

GAMs support thin plate, cubic, shrinkage, P-spline and random-effect smooths,
with tensor products and `by` terms. Extended mixed models add negative
binomial and beta families, zero inflation, and dispersion formulas. The
[mgcv guide](docs/migration/mgcv.md) and
[glmmTMB guide](docs/migration/glmmTMB.md) describe the supported syntax,
results and numerical limitations.

## Validation

R is used as a black-box development oracle through `oracle/run_case.R`.
Synthetic inputs and their numerical observations are stored in `tests/golden/`.
The checks apply separate tolerances to likelihoods, coefficients, covariance,
degrees of freedom, marginal means, and p-values. Boundary or convergence
warnings have explicit case policies and warning checks. Stage 2 identified
boundary limits additionally verify numerical score, curvature and error
bounds. Original R attempts remain recorded, and every unresolved comparison
counts as a failed corpus case.

See [option-level coverage](docs/coverage.md), [clean-room records](CLEANROOM.md),
and [algorithm references](REFERENCES.md). Built-in example data stays in an
ignored development cache; its tests are marked `needs_r`.

## R migration

- [lme4 and lmerTest](docs/migration/lme4.md)
- [emmeans](docs/migration/emmeans.md)
- [car](docs/migration/car.md)
- [nlme](docs/migration/nlme.md)
- [mgcv](docs/migration/mgcv.md)
- [glmmTMB](docs/migration/glmmTMB.md)

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

Run tests locally. The Linux/macOS no-R CI matrix is available only through
manual dispatch when explicitly requested. GitHub Pages deploys only when
documentation content, navigation, or validation evidence changes on `main`.
Release assets are hosted on GitHub.
