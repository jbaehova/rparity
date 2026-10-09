# Stage 1 report: v0.1.0

Stage 1 implements the mixed-model analysis stack in Python without an R runtime.
The final committed corpus contains 3,856 synthetic cases: 3,796 pass
and 60 retain documented differences (98.44% passing).

This is the v0.1.0 release baseline. Later checkout improvements are recorded in
[the refinement report](PARITY_REFINEMENT.md); current coverage is regenerated from
the latest validation. The published release tag and artifacts are unchanged.

## Implemented functions and options

- `lmer`: ML/REML, all six required random-effect forms, weights, offsets, conditional modes and covariance, model summaries, fixed/population predictions and new groups.
- `glmer`: Laplace ML, logit/probit/cloglog binomial and log Poisson, binary/proportional/count responses, prior weights and offsets.
- Inference: Satterthwaite coefficient tests, Type I/II/III F tests, Kenward-Roger covariance and F adjustments.
- `Anova` and `anova`: OLS/GLM Type II/III F/LR/Wald, mixed-model Wald and Gaussian KR tests and ML-refitted likelihood comparisons.
- Marginal means: all five adapters, grids and weights, required contrasts and six corrections, response transformations, trends and joint tests.
- `gls`: ML/REML, AR1/compound symmetry/unstructured/ARMA correlations, identity/power/exponential variance functions, summary, ANOVA, intervals and prediction.

## Coverage

| Module | Synthetic cases | Passed | Failed | Pass rate |
| --- | ---: | ---: | ---: | ---: |
| anova | 700 | 697 | 3 | 99.57% |
| emm | 753 | 752 | 1 | 99.87% |
| glmer | 603 | 583 | 20 | 96.68% |
| gls | 600 | 587 | 13 | 97.83% |
| inference | 600 | 594 | 6 | 99.00% |
| lmer | 600 | 583 | 17 | 97.17% |

The completion threshold is 98% across the entire Stage 1 corpus.
Each module has at least 300 cases. Per-option coverage is generated in
[coverage](../docs/coverage.md); release outcomes are in
[the tagged validation](https://github.com/jbaehova/rparity/blob/v0.1.0/reports/validation.json).

Fourteen development-only representative R examples pass, including sleepstudy,
cbpp, Duncan, warpbreaks and Orthodont. Built-in datasets remain outside the repository.

## Numerical policy and retained failures

Original fixtures are retained. Strict expected-failure markers identify reviewed
differences so unrelated regressions still fail pytest. Fifty-seven markers are strict.
Four reviewed environment-dependent IDs (`glmer_0107`, `lmer_0059`, `lmer_0585`, `lmer_0364`)
permit genuine passes on supported platforms; their numerical thresholds stay unchanged.
Runtime errors are never expected failures.
These expected failures count as failures in every corpus pass-rate calculation.

Tests use the requested field-specific relative tolerances, with a 1e-12 absolute
floating-point floor for zero-valued coefficients, covariance and predictions.
R convergence/singularity warnings use likelihood-only comparison and warning-kind
checks. Sixty individually documented saturated GLM cases lack a finite intercept
MLE; their likelihood supremum and boundary warning are tested instead of divergent
coefficients. Original EMM observations and supplemental fit observations are retained.

All GLS cases record the same public R BFGS controls. All GLMER cases record
`tolPwrss=1e-14` and disable the initial nAGQ=0 step, which otherwise left
observable stopping error in difficult Poisson fits. Previous GLMER precision attempts
are retained for audit. These controls are chosen uniformly, without selecting outputs
based on Python agreement. See [decisions](../DECISIONS.md).

| Case | Classification | Cause |
| --- | --- | --- |
| anova_0337 | boundary_hypothesis_rank | Quasi-separation makes the R binomial Type II Wald covariance and hypothesis ranks unstable. |
| anova_0476 | optimizer_warning | R bobyqa emits a convergence warning that does not apply to the successful Python optimization. |
| anova_0652 | near_zero_coefficient | Near-zero GLMM intercept differs by about 1.4e-7 despite log likelihood agreement within 3.3e-9. |
| emm_0689 | near_zero_contrast | GLS marginal contrast near zero differs by 3.89e-8, exceeding the strict 1e-6 relative estimate criterion. |
| glmer_0002 | coefficient_precision | Laplace objectives meet the 1e-6 criterion, but R/Python coefficient precision exceeds the specified strict relative threshold. |
| glmer_0040 | coefficient_precision | Laplace objectives meet the 1e-6 criterion, but R/Python coefficient precision exceeds the specified strict relative threshold. |
| glmer_0102 | coefficient_precision | Laplace objectives meet the 1e-6 criterion, but R/Python coefficient precision exceeds the specified strict relative threshold. |
| glmer_0107 | fixed_covariance_precision | Laplace objectives meet the 1e-6 criterion, but R/Python fixed covariance precision exceeds the specified strict relative threshold. |
| glmer_0128 | coefficient_precision | Laplace objectives meet the 1e-6 criterion, but R/Python coefficient precision exceeds the specified strict relative threshold. |
| glmer_0152 | fixed_covariance_precision | Laplace objectives meet the 1e-6 criterion, but R/Python fixed covariance precision exceeds the specified strict relative threshold. |
| glmer_0157 | coefficient_precision | Laplace objectives meet the 1e-6 criterion, but R/Python coefficient precision exceeds the specified strict relative threshold. |
| glmer_0269 | fixed_covariance_precision | Laplace objectives meet the 1e-6 criterion, but R/Python fixed covariance precision exceeds the specified strict relative threshold. |
| glmer_0312 | better_optimum_warning_divergence | Python improves log likelihood by 7.21e-5 at SD 0.0172; R returns a singular SD near 4e-7. The fitted Python model does not warrant the R singular warning. |
| glmer_0338 | coefficient_precision | Laplace objectives meet the 1e-6 criterion, but R/Python coefficient precision exceeds the specified strict relative threshold. |
| glmer_0361 | prediction_precision | Laplace objectives meet the 1e-6 criterion, but R/Python prediction precision exceeds the specified strict relative threshold. |
| glmer_0368 | coefficient_precision | Laplace objectives meet the 1e-6 criterion, but R/Python coefficient precision exceeds the specified strict relative threshold. |
| glmer_0372 | fixed_covariance_precision | Laplace objectives meet the 1e-6 criterion, but R/Python fixed covariance precision exceeds the specified strict relative threshold. |
| glmer_0379 | coefficient_precision | Laplace objectives meet the 1e-6 criterion, but R/Python coefficient precision exceeds the specified strict relative threshold. |
| glmer_0418 | fixed_covariance_precision | Laplace objectives meet the 1e-6 criterion, but R/Python fixed covariance precision exceeds the specified strict relative threshold. |
| glmer_0502 | fixed_covariance_precision | Laplace objectives meet the 1e-6 criterion, but R/Python fixed covariance precision exceeds the specified strict relative threshold. |
| glmer_0517 | coefficient_precision | Laplace objectives meet the 1e-6 criterion, but R/Python coefficient precision exceeds the specified strict relative threshold. |
| glmer_0528 | fixed_covariance_precision | Laplace objectives meet the 1e-6 criterion, but R/Python fixed covariance precision exceeds the specified strict relative threshold. |
| glmer_0539 | coefficient_precision | Laplace objectives meet the 1e-6 criterion, but R/Python coefficient precision exceeds the specified strict relative threshold. |
| glmer_0581 | coefficient_precision | Laplace objectives meet the 1e-6 criterion, but R/Python coefficient precision exceeds the specified strict relative threshold. |
| gls_0037 | small fitted prediction relative precision | Strict R/Python small fitted prediction relative precision mismatch; original inputs and numerical thresholds are retained. |
| gls_0054 | near-unit ARMA correlation | Strict R/Python near-unit ARMA correlation mismatch; original inputs and numerical thresholds are retained. |
| gls_0133 | small fitted prediction relative precision | Strict R/Python small fitted prediction relative precision mismatch; original inputs and numerical thresholds are retained. |
| gls_0173 | boundary corSymm optimizer/coefficients | Strict R/Python boundary corSymm optimizer/coefficients mismatch; original inputs and numerical thresholds are retained. |
| gls_0177 | small fitted prediction relative precision | Strict R/Python small fitted prediction relative precision mismatch; original inputs and numerical thresholds are retained. |
| gls_0229 | small fitted prediction relative precision | Strict R/Python small fitted prediction relative precision mismatch; original inputs and numerical thresholds are retained. |
| gls_0273 | boundary corSymm optimizer/coefficients | Strict R/Python boundary corSymm optimizer/coefficients mismatch; original inputs and numerical thresholds are retained. |
| gls_0319 | small fitted prediction relative precision | Strict R/Python small fitted prediction relative precision mismatch; original inputs and numerical thresholds are retained. |
| gls_0323 | small fitted prediction relative precision | Strict R/Python small fitted prediction relative precision mismatch; original inputs and numerical thresholds are retained. |
| gls_0333 | boundary corSymm optimizer/coefficients | Strict R/Python boundary corSymm optimizer/coefficients mismatch; original inputs and numerical thresholds are retained. |
| gls_0339 | small fitted prediction relative precision | Strict R/Python small fitted prediction relative precision mismatch; original inputs and numerical thresholds are retained. |
| gls_0357 | near-zero variance coefficient | Strict R/Python near-zero variance coefficient mismatch; original inputs and numerical thresholds are retained. |
| gls_0429 | small fitted prediction relative precision | Strict R/Python small fitted prediction relative precision mismatch; original inputs and numerical thresholds are retained. |
| inference_0084 | optimizer_warning | R bobyqa reports convergence code 3; Python converges with matching objective and applicable singular warnings. |
| inference_0085 | optimizer_warning | R bobyqa reports convergence code 3; Python converges with matching objective and applicable singular warnings. |
| inference_0086 | optimizer_warning | R bobyqa reports convergence code 3; Python converges with matching objective and applicable singular warnings. |
| inference_0462 | optimizer_warning | R bobyqa reports convergence code 3; Python converges with matching objective and applicable singular warnings. |
| inference_0463 | optimizer_warning | R bobyqa reports convergence code 3; Python converges with matching objective and applicable singular warnings. |
| inference_0464 | optimizer_warning | R bobyqa reports convergence code 3; Python converges with matching objective and applicable singular warnings. |
| lmer_0003 | near_zero_prediction | R optimizer roundoff exceeds the strict relative prediction criterion near zero; all objective checks pass. |
| lmer_0059 | near_zero_prediction | R optimizer roundoff exceeds the strict relative prediction criterion near zero; all objective checks pass. |
| lmer_0160 | optimizer_warning | R bobyqa reports trust-region convergence code 3; Python converges with matching objective and applicable singular warning. |
| lmer_0186 | near_zero_prediction | R optimizer roundoff exceeds the strict relative prediction criterion near zero; all objective checks pass. |
| lmer_0214 | near_zero_prediction | R optimizer roundoff exceeds the strict relative prediction criterion near zero; all objective checks pass. |
| lmer_0239 | optimizer_warning | R bobyqa reports trust-region convergence code 3; Python converges with matching objective and applicable singular warning. |
| lmer_0257 | near_zero_prediction | R optimizer roundoff exceeds the strict relative prediction criterion near zero; all objective checks pass. |
| lmer_0270 | optimizer_warning | R bobyqa reports trust-region convergence code 3; Python converges with matching objective and applicable singular warning. |
| lmer_0288 | near_zero_prediction | R optimizer roundoff exceeds the strict relative prediction criterion near zero; all objective checks pass. |
| lmer_0325 | near_zero_prediction | R optimizer roundoff exceeds the strict relative prediction criterion near zero; all objective checks pass. |
| lmer_0337 | near_zero_prediction | R optimizer roundoff exceeds the strict relative prediction criterion near zero; all objective checks pass. |
| lmer_0372 | near_zero_prediction | R optimizer roundoff exceeds the strict relative prediction criterion near zero; all objective checks pass. |
| lmer_0406 | near_zero_prediction | R optimizer roundoff exceeds the strict relative prediction criterion near zero; all objective checks pass. |
| lmer_0488 | near_zero_prediction | R optimizer roundoff exceeds the strict relative prediction criterion near zero; all objective checks pass. |
| lmer_0550 | near_zero_prediction | R optimizer roundoff exceeds the strict relative prediction criterion near zero; all objective checks pass. |
| lmer_0561 | near_zero_prediction | R optimizer roundoff exceeds the strict relative prediction criterion near zero; all objective checks pass. |
| lmer_0585 | near_zero_prediction | R optimizer roundoff exceeds the strict relative prediction criterion near zero; all objective checks pass. |

`better_optimum`: 5 cases have Python objective improvement exceeding 1e-6 (likelihood or REML criterion).
An improved objective does not override a required warning-kind mismatch.

## Environment-specific additional observation

The local reference passes `lmer_0364`. A later Ubuntu Python 3.13 CI run
exceeds the unchanged 1e-4 relative conditional-mode threshold: the R value is
-1.444224020531799e-5, the Python value is -1.444383197701793e-5, and the
absolute difference is 1.59e-9. This remains a numerical failure on that
environment rather than a tolerance waiver. The reviewed union of failures
across environments contains 61 IDs; the reference coverage table above counts
its 60 actual failures. Every recorded environment stays above 98% passing.

## Validation

- Full pytest: 3920 passed, 60 expected failures; no unexpected failures or skipped tests.
- `uv run ruff check` and `uv run mypy src` pass.
- `uv run mkdocs build --strict` and `uv build` pass.
- A fresh Python 3.12 environment installs the release wheel with pip and runs the README quick start unchanged, with R excluded from PATH.
- Tests run locally by default. The optional manual Linux/macOS CI matrix covers
  Python 3.11, 3.12 and 3.13. Automatic test runs are disabled at the user's request.
- The final remote run was canceled after four jobs passed; it is not claimed as
  a complete six-environment validation. Stage 1 acceptance uses the complete
  local pytest run and the isolated local Python 3.11 typing check.
  The [recorded run](https://github.com/jbaehova/rparity/actions/runs/37885827966)
  tested commit `1f69afe`; subsequent release changes affect documentation and
  workflow triggers only. The validation source fingerprint matches the wheel.

## Same-machine speed comparison

Median of three warm fit calls after warm-up. R process startup and data serialization
are excluded. Small R timings have clock-resolution limits; these are representative
development measurements rather than a performance guarantee.

| Model | Rows | Python seconds | R seconds | Python / R |
| --- | ---: | ---: | ---: | ---: |
| lmer / REML / (1\|g) / boundary | 48 | 0.006878 | 0.009000 | 0.76 |
| lmer / ML / (x\|g) / treatment | 42 | 0.005317 | 0.012000 | 0.44 |
| glmer / binomial-logit | 48 | 0.022744 | 0.015000 | 1.52 |
| glmer / poisson-log | 99 | 0.046105 | 0.016000 | 2.88 |
| gls / corAR1 / REML | 57 | 0.039564 | 0.002000 | 19.78 |

Machine: macOS on Apple silicon. Measurements use the recorded R/package versions
in [oracle versions](../oracle/versions.json) and the locked Python environment.

## Extensions and limitations

No optional Stage 1 extension is claimed. Profile/bootstrap intervals, adaptive
quadrature beyond nAGQ=1, Gamma/negative-binomial GLMMs, multivariate-t corrections,
CLD and spatial GLS correlations are deferred.

Dense mixed-model matrices limit large-data scalability. GLMER requires a full-rank
fixed design and supports all/no random-effect prediction rather than partial selection.
R optimizer warnings can differ when Python finds a valid or better optimum.
Kenward-Roger F inference applies to Gaussian mixed models; GLMM requests are rejected.
Control contrasts require an explicit supported adjustment: the R `dunnettx` default
is unavailable and Python defaults to `none`. Type III main-effect tests require
sum contrasts for the documented interpretation.

Future stages must preserve this corpus and improve near-zero relative precision,
boundary GLS optimization and warning equivalence. Stage 2 has not been started.

The package is MIT licensed. R was used only as a numerical oracle; target package
source and function bodies were not inspected. Algorithms and behavioral references
are recorded in [REFERENCES.md](../REFERENCES.md) and [CLEANROOM.md](../CLEANROOM.md).
