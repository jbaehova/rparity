---
title: "Linear and generalized mixed models in Python"
description: Fit repeated-measures and grouped data with lmer and glmer in Python. Specify random intercepts and slopes, predict, and test fixed effects using rparity.
---

# Mixed models for repeated measurements

Use a mixed model when observations share a participant, site or other
grouping unit. `lmer()` fits a continuous Gaussian response with ML or REML.
`glmer()` fits binomial or Poisson responses with a Laplace approximation.

The snippets below assume a pandas data frame `data` with `score`, `time`,
`treatment` and `participant` columns. Set `treatment` to a categorical column
to make its levels and ordering explicit. For a complete runnable dataset,
see the [quickstart](../getting-started/quickstart.md) and
[mixed-model example](https://github.com/jbaehova/rparity/blob/main/examples/quickstart.py).

## Fit a random intercept or slope

```python
from rparity import lmer

model = lmer("score ~ treatment + time + (1 | participant)", data)
print(model.summary())
print(model.fixef())
print(model.VarCorr())

slopes = lmer("score ~ treatment + time + (time | participant)", data)
```

`(1 | participant)` allows each participant a different intercept.
`(time | participant)` adds a participant-specific time slope and estimates
its correlation with the intercept. Use `(time || participant)` for
independent intercept and slope components. REML is the default; set
`reml=False` for maximum likelihood.

## Predict for existing or new participants

```python
conditional = model.predict(data)
population = model.predict(data, re_form="NA")
new_predictions = model.predict(newdata, allow_new_levels=True)
```

Here `newdata` is a data frame containing the predictors in the fitted
formula. Default predictions include the fitted random effects for known
participants. `re_form="NA"` sets those effects to zero. With
`allow_new_levels=True`, a new participant contributes a zero random effect.

## Test effects and compare models

```python
from rparity import Anova, anova, emmeans, pairs

print(Anova(model, type=2))
means = emmeans(model, "treatment", lmer_df="satterthwaite")
print(pairs(means, adjust="tukey").summary())

reduced = lmer("score ~ time + (1 | participant)", data)
print(anova(reduced, model))
```

`Anova()` tests fixed-effect terms. `anova(reduced, model)` compares
likelihoods and refits REML fits using ML by default. Use models fitted to
the same observations when comparing their likelihoods. For Type III
main-effect tests involving interactions, fit with `contrasts="sum"`.

## Binary and count responses

If `data` instead contains a binary `success` column or a nonnegative integer
`count` column, choose the corresponding family:

```python
from rparity import glmer

binary = glmer("success ~ time + (1 | participant)", data,
               family="binomial")
counts = glmer("count ~ time + (1 | participant)", data,
               family="poisson")
```

Use [`glmmTMB()`](zero-inflated-models.md) for negative-binomial counts,
zero inflation or modeled dispersion. Supported `glmer()` links and further
formula examples appear in the [lme4 migration guide](../migration/lme4.md).
Review singular-fit and convergence warnings before interpreting uncertainty;
see [numerical limitations](../validation.md).
