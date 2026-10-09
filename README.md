<div align="center">
  <h1>rparity</h1>
  <p><strong>Mixed models, GAMs and marginal means in Python.</strong></p>
  <p>Use familiar R-style formulas with your pandas DataFrames.<br>
  Fit, test, compare and predict without running R.</p>
  <p><a href="https://jbaehova.github.io/rparity/">Documentation</a> | <a href="https://jbaehova.github.io/rparity/getting-started/quickstart/">Quickstart</a> | <a href="examples/">Examples</a></p>
</div>

## From data to analysis

Given a pandas DataFrame `df` with a numeric `score`, categorical `treatment`,
numeric `time` and a `participant` identifier:

```python
from rparity import Anova, emmeans, lmer, pairs

model = lmer("score ~ treatment + time + (1 | participant)", data=df)
print(model.summary())
print(Anova(model, type=2))

means = emmeans(model, "treatment")
print(pairs(means, adjust="tukey").summary())
print(model.predict(df.iloc[:5], re_form="NA"))
```

The [quickstart](https://jbaehova.github.io/rparity/getting-started/quickstart/)
includes a complete dataset you can generate and run. Models also accept
Polars DataFrames when Polars is installed.

## Install

Download the wheel from [GitHub Releases](https://github.com/jbaehova/rparity/releases/latest),
then install it in your Python environment:

```sh
python -m pip install ./rparity-*.whl
```

Python 3.11 or newer is required. The published wheel is verified on
**macOS ARM64**. See [installation](https://jbaehova.github.io/rparity/getting-started/installation/)
for building from source on another platform.

## Choose your workflow

| You want to… | Use | Start here |
| --- | --- | --- |
| Analyze repeated measurements or grouped observations | `lmer`, `glmer` | [Mixed models](https://jbaehova.github.io/rparity/guides/mixed-models/) |
| Fit smooth nonlinear effects | `gam`, `gam_check` | [Generalized additive models](https://jbaehova.github.io/rparity/guides/generalized-additive-models/) |
| Model zero-inflated counts or varying dispersion | `glmmTMB` | [Zero-inflated models](https://jbaehova.github.io/rparity/guides/zero-inflated-models/) |
| Compare adjusted means, trends or treatment contrasts | `emmeans`, `emtrends`, `pairs`, `contrast` | [Marginal means](https://jbaehova.github.io/rparity/guides/marginal-means/) |
| Model correlated or unequal residual variance | `gls` | [GLS and residual structures](docs/migration/nlme.md) |
| Test model terms or compare nested models | `Anova`, `anova` | [API reference](https://jbaehova.github.io/rparity/reference/) |

Coming from R? See [lme4, mgcv, glmmTMB and emmeans in Python](https://jbaehova.github.io/rparity/guides/r-packages-in-python/)
and the [R-to-Python migration guides](https://jbaehova.github.io/rparity/migration/lme4/).

## Run an example

Clone the repository, install rparity and run an analysis from
[`examples/`](examples/README.md). Each script creates its own data and
uses the public Python API.

```sh
python examples/quickstart.py
python examples/generalized_additive_models.py
python examples/zero_inflated_counts.py
python examples/marginal_means.py
```

## Find your way around

| Directory | What it provides |
| --- | --- |
| [`src/rparity/`](src/rparity/) | The installable Python library |
| [`examples/`](examples/) | Complete, runnable analyses |
| [`docs/`](docs/) | Installation, workflow guides and API documentation |
| [`development/`](development/) | Maintainer tools and numerical validation records |

R comparisons and known numerical limits are documented in
[validation](https://jbaehova.github.io/rparity/validation/).
To contribute, see [CONTRIBUTING.md](CONTRIBUTING.md).

rparity's implementation is [MIT licensed](LICENSE). Bundled numerical
libraries retain their [third-party notices](LICENSES/README.md).
