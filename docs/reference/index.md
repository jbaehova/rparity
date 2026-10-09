---
title: "rparity Python API reference: models, predictions and inference"
description: Reference the public rparity Python functions and result methods for lmer, glmer, gam, glmmTMB, GLS, ANOVA and estimated marginal means.
---

# Python API reference

Import public functions directly from `rparity`. Formula arguments are
strings, and `data` is a pandas data frame or a Polars frame converted to
pandas internally. This page lists the main call arguments and the result
methods you use in an analysis. Detailed option comparisons are in the
[migration guides](../guides/r-packages-in-python.md).

## Mixed models

```python
lmer(formula, data, reml=True, weights=None, offset=None,
     contrasts="treatment", control=None)

glmer(formula, data, family="binomial", *, link=None, weights=None,
      offset=None, nAGQ=1, contrasts="treatment", control=None)
```

`lmer()` fits Gaussian mixed models; `reml=False` selects ML. `glmer()`
supports binomial and Poisson families with Laplace integration (`nAGQ=1`).
Binomial links are logit, probit and cloglog. Weights and offsets can be
vectors or data-column names. `contrasts` accepts `"treatment"` or `"sum"`.

| Result method | Returns or purpose |
| --- | --- |
| `fixef()` | Named fixed-effect coefficient series |
| `vcov()` | Fixed-effect covariance data frame |
| `ranef(cond_var=True)` | Random-effect data frames by group |
| `VarCorr()` | Random-effect covariance data frames by group |
| `summary()` | Printable model summary |
| `predict(newdata=None, re_form=None, allow_new_levels=False, offset=None)` | Conditional or population prediction array |
| `fitted()`, `residuals(type="response")` | Fitted values and residuals |
| `logLik()`, `AIC()`, `BIC()`, `deviance()` | Fit criteria |
| `is_singular(tol=1e-4)` | Random-effect singularity check |

For `lmer()`, `summary(ddf="satterthwaite")` chooses the coefficient
degrees-of-freedom method. `refit(reml=False)` refits the model with ML.
For both fit types, `re_form="NA"` predicts with random effects set to zero.
See [mixed models](../guides/mixed-models.md) for examples.

## Generalized additive models

```python
gam(formula, data, *, family="gaussian", link=None, method="GCV.Cp",
    weights=None, offset=None, sp=None, scale=0, gamma=1, control=None)

gam_check(model, *, k_rep=200, k_sample=5000, seed=0)
```

Supported families are Gaussian, binomial, Poisson and Gamma. Smoothing
methods are `"GCV.Cp"`, `"ML"` and `"REML"`. Formula smooths use `s()`,
`te()` or `ti()`. Negative entries in `sp` request estimated penalties;
nonnegative entries fix the corresponding penalty.

`gam()` returns a `GamResult`. `gam_check()` returns a dictionary of
convergence and residual diagnostics, including the `"k.check"` table.

| Result method or attribute | Returns or purpose |
| --- | --- |
| `coef()` or `coefficients` | Named coefficient series |
| `vcov(freq=False)` | Bayesian covariance by default; frequentist covariance with `freq=True` |
| `smooth_edf` | Effective degrees of freedom by smooth |
| `summary_tables()` | Dictionary including `"parametric"` and `"smooth"` tables |
| `summary()` | Printable summary |
| `predict(newdata=None, type="link", se_fit=False, terms=None, exclude=None, offset=None)` | Predictions, term effects or linear-predictor matrix |
| `partial_effects(term=None, n=100, newdata=None)` | Numerical effect tables |
| `check(k_rep=200, k_sample=5000, seed=0)` | Equivalent model diagnostics |
| `fitted()`, `residuals(type="deviance")` | Fitted values and residuals |
| `logLik()`, `AIC()`, `deviance()` | Fit criteria |
| `converged` | Fit convergence status |

Prediction types are `"link"`, `"response"`, `"terms"`, `"iterms"` and
`"lpmatrix"`. With `se_fit=True`, ordinary or term predictions return
`{"fit": ..., "se.fit": ...}`. Standard errors condition on the smoothing
parameters. See the [GAM guide](../guides/generalized-additive-models.md).

## Distributional models

```python
glmmTMB(formula, data, family="gaussian", *, ziformula="~0",
        dispformula="~1", link=None, weights=None, offset=None,
        contrasts="treatment", control=None)
```

Supported families are `"gaussian"`, `"poisson"`, `"binomial"`,
`"nbinom1"`, `"nbinom2"` and `"beta"`, with their default links.
Conditional random effects use the main formula. Zero-inflation and
dispersion formulas contain fixed predictors. Fitting uses ML.

`glmmTMB()` returns a `GlmmTMBResult`:

| Result method or attribute | Returns or purpose |
| --- | --- |
| `fixef()` | Coefficient series keyed by `"cond"`, `"zi"`, `"disp"` |
| `vcov(component="cond", full=False)` | Component covariance or full parameter covariance |
| `ranef(cond_var=True)`, `VarCorr()` | Component dictionaries of group results |
| `predict(newdata=None, type="response", re_form=None, allow_new_levels=False, offset=None)` | Response or component prediction array |
| `model_matrix(component="cond", newdata=None)` | Component design matrix |
| `sigma()` | Intercept-only dispersion summary |
| `summary()` | Printable model and component summaries |
| `fitted()`, `residuals(type="response")` | Fitted response means and residuals |
| `logLik()`, `AIC()`, `BIC()` | Fit criteria |
| `is_singular(tol=1e-4)`, `converged` | Fit diagnostics |

Prediction types include `"response"`, `"conditional"` (or `"cond"`),
`"link"`, `"zprob"`, `"zlink"` and `"disp"`. Response means combine the
conditional mean and zero inflation. See the
[zero-inflated-model guide](../guides/zero-inflated-models.md).

## Generalized least squares

```python
gls(formula, data, *, correlation=None, weights=None, method="REML",
    REML=None, control=None, na_action="omit")
```

Choose `method="ML"` for maximum likelihood. `weights` is a variance
structure here, rather than a vector of observation weights.

| Structure | Main arguments |
| --- | --- |
| `corAR1` | `value=0.0, form="~1", fixed=False` |
| `corARMA` | `value=None, form="~1", p=0, q=0, fixed=False` |
| `corCompSymm` | `value=0.0, form="~1", fixed=False` |
| `corSymm` | `value=None, form="~1", fixed=False` |
| `varIdent` | `value=None, form="~1", fixed=None` |
| `varPower` | `value=0.0, form="~fitted(.)", fixed=None` |
| `varExp` | `value=0.0, form="~fitted(.)", fixed=None` |

For example, `corAR1(form="~time | subject")` specifies within-subject
AR1 correlation. GLS results provide `coef()`, `fixef()`, `vcov()`,
`summary()`, `predict(newdata=None, se_fit=False)`, `fitted()` and
`residuals(type="response")`. They also provide `logLik()`, `AIC()`, `BIC()`,
`anova(type="sequential")` and `intervals(level=0.95, which="all")`.
See the [nlme guide](../migration/nlme.md).

## Marginal means and contrasts

```python
emmeans(model, specs, *, by=None, at=None, weights="equal", type="link",
        pairwise=False, adjust=None, lmer_df="kenward-roger", **kwargs)

emtrends(model, specs, *, var, delta_var=None, at=None, by=None,
         weights="equal", type="link", lmer_df="kenward-roger", **kwargs)

contrast(object, method="pairwise", *, by=None, adjust=None, ref=1,
         max_degree=None)
pairs(object, *, reverse=False, **kwargs)
joint_tests(model, *, by=None, at=None, weights="equal",
            lmer_df="kenward-roger", **kwargs)
```

`specs` names the predictors to retain in the result. `at` sets reference
values, for example `{"time": [0, 5]}`. `weights` accepts `"equal"`,
`"proportional"`, `"cells"`, `"outer"`, `"flat"` or explicit weights.
`lmer_df` accepts `"kenward-roger"`, `"satterthwaite"` or `"asymptotic"`.
For distributional models, pass `component="cond"`, `"zi"` or `"disp"`
through `kwargs`.

`emmeans()` and `emtrends()` return an `EmmGrid`, or a dictionary with
`"emmeans"` and `"contrasts"` when pairwise output is requested. `contrast()`
and `pairs()` return an `EmmGrid`; `joint_tests()` returns a data frame.

- `grid.summary(type=None, level=0.95, adjust=None, infer=None, null=0.0)`
  returns estimates, intervals and optional tests. `infer=(True, True)`
  requests confidence intervals and tests.
- `grid.confint(level=0.95)` returns confidence intervals.
- `grid.grid` exposes reference-grid rows.

For explicit reference-grid control, use
`ref_grid(model, at=None, cov_reduce=..., cov_keep=2, lmer_df="kenward-roger",
component="cond", ...)`. `regrid(grid, transform="response")` transforms
grid cells before averaging. See [marginal means](../guides/marginal-means.md)
for the distinction from ordinary response-scale summaries.

## Hypothesis tests and model comparison

```python
Anova(model, type=2, test_statistic=None, error_estimate="pearson",
      component="cond")

anova(*models, type=None, ddf="satterthwaite", refit=True, adjust_sigma=True)
```

Both return pandas data frames. `Anova()` performs Type II or III term tests.
Its statistic depends on the fitted model; distributional models use
component-specific Wald chi-squared tests. For supported linear mixed models,
`anova(model, type=...)` accepts `1`, `2` or `3` for sequential or marginal tests; `ddf`
selects the degree-of-freedom method. `anova(model1, model2)` compares fitted
likelihoods, refitting REML models with ML unless `refit=False`.

Type III main-effect hypotheses in interaction models require suitable
contrast coding; use `contrasts="sum"` when that hypothesis is intended.
See the [car migration guide](../migration/car.md) and
[numerical limitations](../validation.md).
