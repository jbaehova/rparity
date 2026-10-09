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
`partial_effects`, `gam.check`, `data`, and `versions`. Numerical diagnostic
operations also include `rng`, `lanczos`, `quadratic_tail`, `smooth_basis`, and
`matrix_eigen`. Covariance
constructors and distribution families are explicitly allowlisted.
R's contrasts, mixed-model optimizer, GLS optimizer, and marginal-means degrees
of freedom are explicitly configured.

R is required only for oracle regeneration and development checks. The
installed Python library requires Python 3.11 or newer and the five runtime
dependencies declared in `pyproject.toml`; it does not require R or `rpy2`.
Wheel builds vendor the OpenBLAS backend and its native dependencies, with
their license notices. The provider is a build-time dependency. The current
clean-wheel execution check covers macOS ARM64.

## Stage 2 case schema

GAM cases use `call: "gam"`, the ordinary R smooth formula, and a top-level
`family` of `gaussian`, `binomial`, `poisson`, or `Gamma`. An optional `link`
selects the public family constructor's link. Always supply `args.method` as
`REML`, `ML`, or `GCV.Cp`; the oracle otherwise explicitly uses `GCV.Cp`.
`args.sp` and `args.knots` accept numerical smoothing parameters and knot maps.
Fitting uses `gam.control(epsilon = 1e-12, maxit = 1000, mgcv.tol = 1e-10,
newton = list(conv.tol = 1e-10))` unless a case supplies a `gam_control`
override. These controls separately tighten inner IRLS and outer smoothness
selection.
`args.in.out` supplies numerical smoothing and scale initialization through
the documented public `gam()` interface. An optional `gam_initial_sp` uses
the public two-part fit interface with an initialized setup object's numeric
`sp` field. Neither option fixes the smoothing parameter unless `args.sp`
explicitly requests that behavior.

The GAM fit result includes the coefficient vector and Bayesian covariance,
per-coefficient `edf` and `edf1`, smoothing parameters, scale, likelihood,
selection criterion, parametric coefficient table, and approximate smooth-test
table. `X` is the public `predict(type = "lpmatrix")` result. Each `smooths`
entry records numerical penalty matrices `S`, penalty scaling `S_scale`, rank,
null-space dimension, and parameter indices. Those indices remain R's
one-based inclusive indices. These are fitted-object numeric data, not function
bodies or package source.

`optimizer_diagnostics: true` adds public numerical prior and working weights,
Bayesian and frequentist covariance, factor matrices, scale, and residual
degrees of freedom to the outer gradient/Hessian diagnostics when available.
`smooth_basis_diagnostics: true` adds available numerical `F`, `xp`, `UZ`,
`Xu`, `shift`, and `cmX` fields. They are used for independent mathematical
basis verification, never as Python runtime fitting inputs.

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
`gam_prediction: true` adds new-data link and term predictions with standard
errors to the fit result, using training rows when `newdata` is absent.

Extended mixed-model cases use `call: "glmmTMB"` and a top-level `family` of
`poisson`, `nbinom1`, `nbinom2`, `binomial`, `beta`, or `gaussian`. The public
`beta_family()` constructor implements `beta`. `ziformula` defaults explicitly
to `~ 0`, and `dispformula` to `~ 1`. The oracle uses one evaluation thread and
`nlminb` controls `iter.max = 10000`, `eval.max = 10000`, `rel.tol = 1e-14`,
`x.tol = 1e-12`, `sing.tol = 1e-16`, and `xf.tol = 1e-16`. `tmb_control` can
supply explicit overrides. Top-level `weights` and `offset` name data columns,
as in Stage 1 cases. JSON arrays under `args.start` are converted to numeric
vectors for the public `beta`, `betazi`, `betadisp`, and `theta` start fields.
Their parameterization follows the
[public fit specification](https://glmmTMB.github.io/glmmTMB/reference/glmmTMB.html)
and [unstructured-covariance mapping](https://glmmtmb.github.io/glmmTMB/articles/covstruct.html#unstructured-1).
For the recorded glmmTMB 1.1.15.2 version, Gaussian dispersion coefficients
model log standard deviation.

`score_polish: true` installs an independent optimizer callback after `nlminb`
through the documented
[glmmTMBControl interface](https://glmmTMB.github.io/glmmTMB/reference/glmmTMBControl.html).
It evaluates only the supplied objective and gradient. Up to six Newton
updates use public `stats::optimHess()` with an explicit `1e-4` score increment.
Only small steps in stable curvature directions are considered; backtracking
accepts a step when the objective is unchanged within `1e-10` and the score
norm decreases. A score below `1e-8` establishes stationary-score convergence.
This oracle refinement is distinct from the Python covariance convention's
`1e-3` increment.

The extended-model fit result retains conditional `beta`, `coef_names`, and
`vcov` fields and adds all-component `fixef`, component covariance matrices,
full covariance and names, coefficient tables, conditional random modes and
their variances, random-effect covariance matrices, and optimizer diagnostics.
`predictions` contains fitting-data `response`, `cond`, `zprob`, and `disp`
predictions. Explicit prediction accepts `prediction_type: "cond"` as an
alias for R's `conditional` and supports `se_fit`, `population`, and
`allow_new_levels`. Existing `Anova` and marginal-means operations call their
public methods for this model class too.

`tmb_covariance_diagnostics: true` adds the public fitted native parameter
vector, `sdr$cov.fixed`, and an independently evaluated public score Hessian.
`tmb_hessian_step` defaults to `1e-3` and controls that diagnostic score
increment. The result retains finite-value checks, eigenvalues, and the
fitted score. An optional `tmb_inner_control` changes documented
`TMB::newtonOption()` controls through glmmTMB's public modular fitting
interface. These diagnostic options do not supply Python runtime fitting
inputs.

## Reproducible Stage 2 references

The generators enumerate seeded designs before fitting. Run them locally with
the development R packages recorded in `versions.json`:

```sh
uv run python oracle/generate_gam.py --count 1200
uv run python oracle/generate_tmb.py --count 1200
```

Use `--start` and `--count` for an indexed range, or `--specs-only` to write
the input manifest without fitting. Generation writes each completed fixture
atomically. Every indexed case remains in the corpus, including errors and
boundary fits.

The TMB generator uniformly enables stationary-score polishing. Both
generators apply the case-specific starts and control overrides recorded in
`stage2_overrides.json`. Independent public R refits confirm a better
likelihood, a better smoothing criterion, or improved stationarity before a
reference refinement is accepted. The full fitting specification is retained
and can be reproduced.
When a specification changes, the previous specification and complete oracle
observation remain in `oracle_attempts`, including original warnings. Beta
draws that round to an endpoint are projected into the open interval before
adding structural zeros; the original endpoint attempt is retained too.

Reference refinement does not discard a failed input or change comparison
tolerances. Current required-field differences continue to count as failures
in the recorded numerical coverage. A warning difference receives a separate
case-level numerical explanation rather than a fabricated runtime warning.

Some boundary cases have a separately recorded identified-limit reference.
For example, a public reduced-rank R fit can provide finite covariance on the
identifiable directions of a rank-one random-effect model. Dynamic checks
verify the score, curvature, likelihood, and physical covariance before such
a reference is used. Original observations remain intact. Auxiliary models
do not replace the original model's parameter count, AIC, or BIC. Each
exception records its fields and mathematical evidence in the Stage 2
reports and golden policies.

Stage 2 is complete in v0.2.0. The fixed 2,400-case corpus has 2,356 passes
(98.17%) and 44 retained failures: 16 GAM and 28 extended-model cases.
Stage 1 retains its 3,799/3,856 baseline with 57 differences and no new
regression. See the [Stage 2 report](../reports/STAGE_2_REPORT.md) and
[case-level failures](../reports/STAGE_2_FAILURES.json). All tests run locally;
GitHub Actions tests require an explicit request for manual dispatch.

Synthetic golden files include their full inputs and observations. Built-in R
example data may only be saved beneath ignored `oracle/cache/`. Regeneration
requires development packages listed in `versions.json`, when available.
