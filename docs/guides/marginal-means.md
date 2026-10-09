---
title: "Estimated marginal means and treatment contrasts in Python"
description: Compute adjusted group means, pairwise comparisons, custom contrasts and trends in Python with emmeans-style inference. Control reference grids and p-value adjustments.
---

# Marginal means and contrasts

Estimated marginal means summarize model predictions over a defined reference
grid. Use them to compare treatments while controlling the covariate values
and factor weighting used in the comparison. `emmeans()` returns an `EmmGrid`;
its summary and confidence-interval methods return pandas data frames.

The snippets below assume a fitted `model` with categorical `treatment` and
numeric `time` predictors. Start with the [quickstart](../getting-started/quickstart.md)
or run the [complete marginal-means example](https://github.com/jbaehova/rparity/blob/main/examples/marginal_means.py)
to create the data and model.

## Adjusted means and pairwise comparisons

```python
from rparity import emmeans, pairs

means = emmeans(model, "treatment", at={"time": [2.5]},
                lmer_df="satterthwaite")
print(means.summary())
print(means.confint(level=0.95))
print(pairs(means, adjust="tukey").summary())
```

Here the comparison uses `time=2.5`. By default, other numeric covariates are
reduced to their means and factor combinations are weighted equally.
Binary numeric covariates can be retained as factor-like variables by the
reference-grid rules. Pass `at` to control covariate values explicitly.
`by="site"` computes comparisons separately within site.

## Choose weights and custom contrasts

```python
from rparity import contrast

observed_mix = emmeans(model, "treatment", weights="proportional")
print(observed_mix.summary())

# With treatment levels ordered as control, low, high:
comparison = contrast(means, {"high vs control": [-1, 0, 1]},
                      adjust="none")
print(comparison.summary(infer=(True, True)))
```

Custom contrast coefficients follow the row order of the marginal means.
Inspect `means.summary()` before specifying them. Supported adjustments
include Tukey for pairwise contrasts, Bonferroni, Holm, Sidak and FDR. Control
contrasts use an explicitly supported adjustment; R's default `dunnettx`
approximation is unavailable.

## Trends and response-scale estimates

```python
from rparity import emtrends

trends = emtrends(model, "treatment", var="time")
print(trends.summary())
```

For a model with a non-identity link, `emmeans(model, "treatment",
type="response")` back-transforms the averaged link-scale estimate.
`regrid="response"` instead transforms grid cells before averaging; these
answer different questions. Trends use finite differences with respect to
the requested covariate.

For linear mixed models, choose `lmer_df="kenward-roger"`,
`"satterthwaite"` or `"asymptotic"`. Kenward-Roger is the default.
Distributional models support `component="cond"`, `"zi"` and `"disp"`;
see the [count-model guide](zero-inflated-models.md).

The inference adapter also supports selected statsmodels OLS and GLM
results. See the [API reference](../reference/index.md) for result methods
and the [emmeans migration guide](../migration/emmeans.md) for all supported
contrast families and weighting options.
