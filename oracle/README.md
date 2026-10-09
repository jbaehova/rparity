# Black-box oracle

All numerical R observations pass through `run_case.R`. This script accepts a
single JSON case or `{ "cases": [...] }`. Each case specifies a public model
function, formula, data columns (or CSV), arguments, and an extraction operation.
It writes a JSON array of public results and captured warnings.

```sh
Rscript oracle/run_case.R oracle/cache/input.json oracle/cache/output.json
```

Supported fit calls are `lmer`, `glmer`, `gls`, `lm`, `glm`, `gam`, and
`glmmTMB`. Operations
include `fit`, `anova`, `Anova`, `emmeans`, `emtrends`, `joint_tests`, `predict`,
`partial_effects`, `gam.check`, `data`, and `versions`. Covariance constructors
and distribution families are explicitly allowlisted.
R's contrasts, mixed-model optimizer, GLS optimizer, and marginal-means degrees
of freedom are explicitly configured.

## Stage 2 case schema

GAM cases use `call: "gam"`, the ordinary R smooth formula, and a top-level
`family` of `gaussian`, `binomial`, `poisson`, or `Gamma`. An optional `link`
selects the public family constructor's link. Always supply `args.method` as
`REML`, `ML`, or `GCV.Cp`; the oracle otherwise explicitly uses `GCV.Cp`.
`args.sp` and `args.knots` accept numerical smoothing parameters and knot maps.
Fitting uses `gam.control(epsilon = 1e-10, maxit = 1000)` unless a case supplies
a `gam_control` override.

The GAM fit result includes the coefficient vector and Bayesian covariance,
per-coefficient `edf` and `edf1`, smoothing parameters, scale, likelihood,
selection criterion, parametric coefficient table, and approximate smooth-test
table. `X` is the public `predict(type = "lpmatrix")` result. Each `smooths`
entry records numerical penalty matrices `S`, penalty scaling `S_scale`, rank,
null-space dimension, and parameter indices. Those indices remain R's
one-based inclusive indices. These are fitted-object numeric data, not function
bodies or package source.

For GAM prediction, supply `operation: "predict"`, `newdata`,
`prediction_type` (`link`, `response`, `terms`, or `lpmatrix`) and optionally
`se_fit: true`. `unconditional: true` requests smoothing-uncertainty covariance
when available. Results contain `prediction`, optional `se_fit`, and matrix
column names. `operation: "partial_effects"` returns term fits and standard
errors on supplied plotting-grid `newdata`, or on the fitting data when no
grid is supplied. An optional `terms` list selects terms.

`operation: "gam.check"`, or `gam_check: true` on a fit case, returns the public
`mgcv::k.check()` table and numerical convergence, rank, scale, deviance, and
residual degrees of freedom. `diagnostic_seed` defaults to 1, `k_rep` to 400,
and `k_subsample` to 5000. The simulated k-check p-value depends on the RNG
implementation as well as the seed; its simulation settings are explicit.
`gam_prediction: true` adds new-data link and term predictions (training rows when newdata is absent) with standard
errors to the fit result.

Extended mixed-model cases use `call: "glmmTMB"` and a top-level `family` of
`poisson`, `nbinom1`, `nbinom2`, `binomial`, `beta`, or `gaussian`. The public
`beta_family()` constructor implements `beta`. `ziformula` defaults explicitly
to `~ 0`, and `dispformula` to `~ 1`. The oracle uses one evaluation thread and
`nlminb` controls `iter.max = 10000`, `eval.max = 10000`, and `rel.tol = 1e-10`;
`tmb_control` can supply explicit overrides. Top-level `weights` and `offset`
name data columns, as in Stage 1 cases.

The extended-model fit result retains conditional `beta`, `coef_names`, and
`vcov` fields and adds all-component `fixef`, component covariance matrices,
full covariance and names, coefficient tables, conditional random modes and
their variances, random-effect covariance matrices, and optimizer diagnostics.
`predictions` contains fitting-data `response`, `cond`, `zprob`, and `disp`
predictions. Explicit prediction accepts `prediction_type: "cond"` as an
alias for R's `conditional` and supports `se_fit`, `population`, and
`allow_new_levels`. Existing `Anova` and marginal-means operations call their
public methods for this model class too.

Synthetic golden files include their full inputs and observations. Built-in R
example data may only be saved beneath ignored `oracle/cache/`. Regeneration
requires development packages listed in `versions.json`, when available.
