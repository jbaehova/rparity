---
title: "Generalized additive models in Python with mgcv-style smooths"
description: Fit GAMs in Python using s, te and ti smooths with rparity. Estimate nonlinear effects, choose REML smoothing, predict with standard errors and check basis size.
---

# Generalized additive models

A GAM estimates nonlinear relationships without specifying a single
polynomial shape. Write smooth terms inside the formula and let `gam()`
estimate their penalties. Supported response families are Gaussian, binomial,
Poisson and Gamma.

The example below assumes a pandas data frame `data` with a continuous
response `y` and predictor `x`. A complete synthetic dataset is in the
[runnable GAM example](https://github.com/jbaehova/rparity/blob/main/examples/generalized_additive_models.py).

## Fit and inspect a smooth

```python
from rparity import gam

model = gam("y ~ s(x, bs='cr', k=8)", data, method="REML")
print(model.summary())
print(model.summary_tables()["smooth"])
```

`k` sets the available basis dimension. The fitted effective degrees of
freedom describe the smooth's complexity after penalization. `method="REML"`
selects the penalty by restricted likelihood; `"ML"` and `"GCV.Cp"` are
also available. The default `"GCV.Cp"` uses GCV or UBRE according to whether
the response scale is estimated or known.

| Formula term | Purpose |
| --- | --- |
| `s(x, bs='tp', k=8)` | Thin plate regression spline |
| `s(x, bs='cr', k=8)` | Natural cubic regression spline |
| `s(x, bs='cs', k=8)` | Cubic spline with shrinkage |
| `s(x, bs='ps', k=8)` | P-spline |
| `s(group, bs='re')` | Random-effect smooth |
| `te(x, z, k=[5, 5])` | Tensor product smooth |
| `ti(x, z, k=[5, 5])` | Tensor interaction with marginal constraints |
| `s(x, by=treatment, k=8)` | Factor-specific smooths |

For `group` or `treatment`, use a pandas categorical column. Numeric `by`
variables are also supported. The [mgcv migration guide](../migration/mgcv.md)
describes marginal bases and tensor options.

## Predict and extract effects

```python
predictions = model.predict(data.iloc[:5], type="response", se_fit=True)
print(predictions["fit"])
print(predictions["se.fit"])

terms = model.predict(data.iloc[:5], type="terms", se_fit=True)
effect = model.partial_effects(n=50)
```

Prediction standard errors condition on the estimated smoothing parameters.
They describe uncertainty in the estimated mean, rather than new-observation
prediction intervals. `type="terms"` returns term contributions on the link
scale, excluding the intercept. `partial_effects()` creates numerical effect
tables that you can plot using your preferred Python plotting library.

## Check whether the basis is large enough

```python
from rparity import gam_check

checks = gam_check(model, k_rep=200, seed=42)
print(checks["k.check"])
```

The basis check reports effective degrees of freedom, a residual-variance
index and a permutation p-value where available. A low index with a low
p-value can motivate a larger `k` and a refit. Interpret it alongside the
residual pattern and convergence diagnostics.

This implementation uses dense matrices and suits modest basis dimensions.
Large-data `bam()` workflows and uncertainty from smoothing selection are
unavailable. Smooth coefficient coordinates can differ from R even when
fitted functions agree. See [numerical limitations](../validation.md) and
the [mgcv migration guide](../migration/mgcv.md) when reproducing an analysis.
