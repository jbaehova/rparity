# Python examples

Use these scripts as starting points for an analysis with `rparity`. Each one
creates a small synthetic pandas DataFrame and fits a model entirely in Python.
Replace the DataFrame with your observations to use the same workflow on your data.

| What you want to do | Example | Python API |
| --- | --- | --- |
| Estimate a longitudinal trend and predict individual trajectories | [quickstart.py](quickstart.py) | `lmer`, `Anova`, `emmeans`, `pairs`, `predict` |
| Fit a nonlinear effect and predict with standard errors | [generalized_additive_models.py](generalized_additive_models.py) | `gam`, `summary`, `predict`, `partial_effects` |
| Analyze overdispersed counts with extra zeros and clustered observations | [zero_inflated_counts.py](zero_inflated_counts.py) | `glmmTMB`, component `Anova`, component predictions |
| Compare adjusted treatment means and a prespecified contrast | [marginal_means.py](marginal_means.py) | `lmer`, `emmeans`, `pairs`, `contrast` |

Install `rparity` using the [installation guide](https://jbaehova.github.io/rparity/getting-started/installation/).
From a checkout of this repository, run:

```bash
python examples/quickstart.py
python examples/generalized_additive_models.py
python examples/zero_inflated_counts.py
python examples/marginal_means.py
```

For an editable checkout with dependencies managed by uv:

```bash
uv sync
uv run python examples/quickstart.py
```

The scripts require no R installation, no external datasets, and no plotting
dependencies. Results are printed as summaries or pandas tables that you can
export, join to other data, or pass to your preferred plotting library.

For formulas and supported options, see the
[Python documentation](https://jbaehova.github.io/rparity/).
