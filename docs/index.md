---
title: "rparity: mixed models, GAMs and marginal means in Python"
description: Fit linear mixed models, generalized additive models and zero-inflated regression in Python. Use R-style formulas, prediction and estimated marginal means without R.
---

# Statistical modeling in Python, with familiar R formulas

rparity fits mixed models, smooth effects and zero-inflated responses directly
in Python. Start with a pandas data frame, write a formula, and carry the fitted
model into prediction, hypothesis tests and estimated marginal means.
Fitting and inference require no R installation or `rpy2`.

[Install rparity](getting-started/installation.md)
| [Run your first analysis](getting-started/quickstart.md)
| [API reference](reference/index.md)
| [Source and examples](https://github.com/jbaehova/rparity)

## From your data to an answer

Suppose `data` contains repeated measurements with columns `score`, `time`
and `participant`:

```python
from rparity import Anova, emmeans, lmer, pairs

model = lmer("score ~ time + (1 | participant)", data)
print(model.summary())
print(Anova(model, type=2))

means = emmeans(model, "time", at={"time": [0, 5]})
print(pairs(means).summary())
predictions = model.predict(data, re_form="NA")
```

The [quickstart](getting-started/quickstart.md) includes a complete dataset
and a runnable version of this workflow. Result tables are pandas data frames
or named series; predictions can be used in your existing Python analysis.

## Choose your analysis

| You want to | Start here | Python API |
| --- | --- | --- |
| Model repeated measurements or grouped observations | [Mixed models](guides/mixed-models.md) | `lmer`, `glmer` |
| Estimate nonlinear effects without fixing their shape | [Generalized additive models](guides/generalized-additive-models.md) | `gam`, `gam_check` |
| Model overdispersed counts and excess zeros | [Zero-inflated models](guides/zero-inflated-models.md) | `glmmTMB` |
| Compare adjusted group means and treatment effects | [Marginal means and contrasts](guides/marginal-means.md) | `emmeans`, `pairs`, `contrast` |
| Account for correlated or unequal residual variation | [GLS and residual covariance](migration/nlme.md) | `gls`, `corAR1`, `varIdent` |

## Bringing an R analysis to Python?

The [R-to-Python guide](guides/r-packages-in-python.md) maps common
`lme4`, `mgcv`, `glmmTMB`, `emmeans`, `car` and `nlme` workflows to rparity.
Detailed migration pages compare formula syntax, method arguments and result
extraction side by side.

Numerical agreement is checked against recorded R results. Read
[numerical accuracy and limitations](validation.md) for the supported scope
and known differences. The Python implementation is MIT licensed.
