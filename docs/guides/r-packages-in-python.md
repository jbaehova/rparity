---
title: "R statistical models in Python: lme4, mgcv, glmmTMB and emmeans"
description: Move R mixed-model, GAM and marginal-means workflows to Python with rparity. Find equivalent lmer, mgcv, glmmTMB, emmeans, car and nlme calls without an R runtime.
---

# Bring your R analysis to Python

rparity provides familiar formula-based APIs for selected `lme4`, `mgcv`,
`glmmTMB` and `emmeans` workflows. Fit models and carry their results into
prediction and inference without an R installation or `rpy2`.
The numerical implementation runs directly in Python.

If you are starting a new analysis, use the [quickstart](../getting-started/quickstart.md)
and [analysis guides](mixed-models.md). If you already have R code, the
migration pages below map its calls to the Python API.

## Find the Python equivalent

| Your R workflow | Python entry point | Guide |
| --- | --- | --- |
| `lme4::lmer`, `lmerTest` inference | `lmer()` | [Linear mixed models](../migration/lme4.md) |
| `lme4::glmer` | `glmer()` | [Binomial and Poisson mixed models](../migration/lme4.md) |
| `mgcv::gam` | `gam()`, `gam_check()` | [Smooths, REML and GAM prediction](../migration/mgcv.md) |
| `glmmTMB::glmmTMB` | `glmmTMB()` | [Zero inflation and dispersion](../migration/glmmTMB.md) |
| `emmeans`, `emtrends`, pairwise contrasts | `emmeans()`, `emtrends()`, `pairs()` | [Marginal means and contrasts](../migration/emmeans.md) |
| `car::Anova`, `lmerTest::anova` | `Anova()`, `anova()` | [Term tests and model comparison](../migration/car.md) |
| `nlme::gls` | `gls()`, correlation and variance structures | [Residual covariance models](../migration/nlme.md) |

Formula arguments become Python strings. Random effects retain syntax such
as `"y ~ x + (1 | group)"`, and GAM smooths use `"y ~ s(x, k=8)"`.
Python keyword arguments use underscores: for example, `re_form`,
`allow_new_levels` and `lmer_df`.

## Fit, then extract results

With an existing pandas data frame `data` containing `y`, `x` and `group`:

```python
from rparity import emmeans, lmer, pairs

model = lmer("y ~ x + (1 | group)", data)
coefficients = model.fixef()
population = model.predict(data, re_form="NA")
means = emmeans(model, "x", at={"x": [0, 1]})
comparisons = pairs(means).summary()
```

`fixef()` returns a named pandas series. Marginal-mean and contrast summaries
return data frames, so you can export them, join them with other tables or
use them in your existing Python reporting workflow. Consult the
[API reference](../reference/index.md) for other result methods.

## Preserve the question your analysis answers

The Python APIs cover the supported options documented in each migration
guide. Set the family, fitting method, contrast coding and prediction scale
explicitly when reproducing an R analysis. For interaction models, sum
contrasts are appropriate for the usual Type III main-effect hypotheses.
For nonlinear models, setting random effects to zero differs from averaging
over their distribution.

Check [numerical accuracy and limitations](../validation.md) before relying
on a migrated fit, especially with singular random effects, extreme
smoothing or nearly absent zero-inflation and dispersion components.
