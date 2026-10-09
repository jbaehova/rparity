---
title: "Zero-inflated negative binomial mixed models in Python"
description: Fit excess-zero and overdispersed count data in Python with glmmTMB-style formulas. Model random effects, zero inflation and dispersion, then predict each component.
---

# Zero-inflated and overdispersed counts

Use `glmmTMB()` when your count model needs negative-binomial variation,
structural zeros or predictors for dispersion. A single fit combines a
conditional response model with optional zero-inflation and dispersion
formulas. Conditional random effects describe grouped observations.

This guide assumes a pandas data frame `data` with integer `count`, numeric
`x`, categorical `treatment` and grouping column `site`. The
[complete count-model example](https://github.com/jbaehova/rparity/blob/main/examples/zero_inflated_counts.py)
generates suitable data and runs the analysis without R.

## Give each component its own formula

```python
from rparity import glmmTMB

model = glmmTMB(
    "count ~ treatment + x + (1 | site)",
    data,
    family="nbinom2",
    ziformula="~ treatment",
    dispformula="~ treatment",
)
print(model.summary())
print(model.fixef())
```

The conditional formula models the count mean on a log scale. `ziformula`
models the probability of a structural zero on a logit scale. `dispformula`
models dispersion on a log scale. Use `ziformula="~0"` to omit zero inflation
and `dispformula="~1"` for constant dispersion; both are defaults.

| Family | Conditional variance or response scope |
| --- | --- |
| `poisson` | Variance equals the mean |
| `nbinom1` | `mu * (1 + phi)` |
| `nbinom2` | `mu + mu**2 / phi` |
| `binomial` | Binary or success/failure responses |
| `gaussian` | Continuous response with modeled standard deviation |
| `beta` | Response between zero and one; optional structural zeros |

NB1 and NB2 use different dispersion conventions. Increasing `phi` increases
NB1 overdispersion and decreases NB2 overdispersion. All families use their
default links in this API.

## Predict the quantity you need

```python
overall = model.predict(data, type="response", re_form="NA")
conditional = model.predict(data, type="conditional", re_form="NA")
zero_probability = model.predict(data, type="zprob")
dispersion = model.predict(data, type="disp")
```

The overall response mean is `conditional * (1 - zero_probability)`.
`re_form="NA"` sets fitted random effects to zero. It does not integrate
the inverse link over the random-effect distribution. Default predictions
include fitted random effects for known groups.

## Compare treatments in a specific component

```python
from rparity import Anova, emmeans

print(Anova(model, type=2, component="cond"))
conditional_means = emmeans(model, "treatment", component="cond",
                            type="response")
zero_probabilities = emmeans(model, "treatment", component="zi",
                             type="response")
print(conditional_means.summary())
print(zero_probabilities.summary())
```

Conditional marginal means describe the conditional count process.
Zero-inflation marginal means describe structural-zero probabilities.
`component="disp"` selects dispersion. The [marginal-means guide](marginal-means.md)
explains averaging and contrast adjustments.

Fits use ML with a normalized Laplace approximation for conditional random
effects. Dense calculations suit small and medium grouped models. Random
effects in zero-inflation or dispersion formulas, hurdle models and prediction
standard errors are unavailable. Very weak components and singular random
effects can make Wald inference unstable. Read the
[glmmTMB migration guide](../migration/glmmTMB.md) for full component semantics
and [numerical limitations](../validation.md) for known differences.
