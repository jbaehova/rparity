---
title: "Python mixed-model quickstart: fit, predict and compare means"
description: Run a complete rparity analysis in Python with synthetic repeated measurements. Fit a random-slope model, test a time effect and compare estimated marginal means.
---

# Your first model

This complete example creates repeated measurements for twelve subjects.
It estimates an overall time trend while allowing each subject a different
intercept and slope. [Install rparity](installation.md), then paste the code
into a Python script or notebook:

```python
import numpy as np
import pandas as pd
from rparity import Anova, emmeans, lmer, pairs

rng = np.random.default_rng(42)
subject = np.repeat(np.arange(12), 6)
day = np.tile(np.arange(6), 12)
intercepts = rng.normal(0, 12, 12)
slopes = rng.normal(0, 2, 12)
data = pd.DataFrame({
    "reaction": 250 + 8 * day + intercepts[subject]
                + slopes[subject] * day + rng.normal(0, 5, len(day)),
    "day": day,
    "subject": subject.astype(str),
})

model = lmer("reaction ~ day + (day | subject)", data)
print(model.summary())
print(Anova(model, type=2))

means = emmeans(model, "day", at={"day": [0, 5]})
print(pairs(means).summary(infer=True).to_string(index=False))

future = pd.DataFrame({"day": [0, 3, 5], "subject": ["0", "0", "0"]})
future["population_prediction"] = model.predict(future, re_form="NA")
future["subject_prediction"] = model.predict(future)
print(future.to_string(index=False))
```

`model.summary()` reports fixed effects and variance components. The `day`
coefficient describes the population trend; `(day | subject)` captures
correlated subject-specific intercepts and slopes. `Anova()` tests the fixed
time effect, and `pairs()` reports the fitted difference between day zero
and day five with its uncertainty.

Population predictions set random effects to zero. Subject predictions
include the fitted random effects for subject `"0"`. Replace the generated
`data` with your own data frame and update the formula to start an analysis.
Keep grouping labels consistent between training data and prediction data.

## Choose your next step

- [Mixed models](../guides/mixed-models.md) for random-effect formulas,
  binary outcomes and counts.
- [Generalized additive models](../guides/generalized-additive-models.md)
  for nonlinear effects and smooth predictions.
- [Zero-inflated counts](../guides/zero-inflated-models.md) for
  negative-binomial responses, structural zeros and dispersion.
- [Marginal means](../guides/marginal-means.md) for treatment comparisons
  and custom contrasts.

The repository includes [four runnable examples](https://github.com/jbaehova/rparity/tree/main/examples)
with complete data generation. Use the [API reference](../reference/index.md)
for call arguments and result methods, and [numerical limitations](../validation.md)
when reproducing a published or existing R analysis.
