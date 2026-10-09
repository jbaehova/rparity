# Stage 2 validation report

Version 0.2.0 completes Stage 2, adding clean-room GAMs and distributional mixed models to the same
`rparity` package. R is a development oracle; fitting and inference require no
R process or rpy2. Numerical validation uses all 2,400 seeded designs, preserves
all original reference attempts, and counts every remaining discrepancy as a
failure.

## Implemented scope

| Module | Required behavior |
| --- | --- |
| GAM smooths | `s` with tp/cr/cs/ps/re, `te`, `ti`, factor and numeric `by`, explicit `k` |
| GAM selection and families | REML, ML, GCV/UBRE; Gaussian, binomial, Poisson, Gamma |
| GAM results | Coefficients, coefficient and smooth EDF, approximate smooth tests, summary, predictions and SEs, terms, partial effects, numerical k-check |
| Distributional GLMMs | Poisson, NB1, NB2, binomial, beta, Gaussian; fixed zero-inflation and dispersion formulas |
| Mixed effects and likelihood | Random intercepts and correlated slopes; normalized observed-Hessian Laplace likelihood |
| GLMM results and downstream analysis | Fixed components, covariance, random modes and conditional covariance, VarCorr, predictions, AIC/BIC, summary, component marginal means and Type II/III Wald Anova |

## Acceptance and local verification

The fixed corpus contains 1,200 GAM and 1,200 GLMM designs. The complete raw
comparison passes 2,356 of 2,400 cases (98.17%). The aggregate Stage 2 gate is
98%; neither module's failures are omitted from that denominator.

| Module | Cases | Passed | Failed | Pass rate |
| --- | ---: | ---: | ---: | ---: |
| GAM | 1,200 | 1,184 | 16 | 98.67% |
| GLMM | 1,200 | 1,172 | 28 | 97.67% |
| Stage 2 | 2,400 | 2,356 | 44 | 98.17% |

Final integrated local verification passed: 6,837 passed, 101 expected failures
and one actual pass under a previously reviewed platform-dependent marker.
There were no unexpected failures or skipped tests. The 101 expected failures
are the 57 Stage 1 and 44 Stage 2 numerical discrepancies, all counted as
failures in corpus rates. [The final outcome record](validation.json) identifies
every corpus case and all sixteen passing representative R examples.

Ruff passed, mypy passed for all 29 source files, and the strict MkDocs build
passed after coverage regeneration. README examples and two clean Python 3.11
installed environments passed with external process launch blocked. The final
wheel was rebuilt from the sdist in isolation with byte-identical output.

The Stage 1 regression baseline is 3,799 passes and 57 retained discrepancies
among 3,856 designs. The immutable v0.1.0 tag and assets remain unchanged.
Representative R examples include the previous fourteen and two added mcycle
and Salamanders checks. Documentation examples also execute with process
launch disabled.

Validation uses local pytest, Ruff, mypy, a strict MkDocs build and clean
wheel installation. Stage 1 retains exactly 3,799 passing cases and the same
57 discrepancies; the final regression gate reports no new failures. No GitHub Actions test job is started. Documentation
publication is limited to content and navigation changes.

## Independent reference precision and boundary accounting

Public optimizer starts refine stationary R references without changing the
input statistical problem. Each fixture preserves all earlier specifications,
outputs and warnings. Independently measured scores, conditional-mode
residuals, augmented-QR consistency or higher-order information stability
justify the corrections. Failed experiments remain recorded.

GAM comparisons map independently constructed bases into the same function
space. Exact design/penalty gauges use their identified coefficient quotient
and preserve the REML integration measure. Six Gaussian variance-boundary
certificates prove a global zero-variance minimum. The 145 ordinary and seven
joint-gauge positive-penalty limits recompute convex-score and Schur covariance
bounds. Identified estimates, covariance and predictions retain their original
tolerances. Metadata-only optimizer-event exceptions also require actual
Python convergence, stationary analytic score and nonnegative curvature.

Mixed-model boundary checks distinguish undefined singular R covariance from
finite identified inference. Stable independent public Hessians or finite
rank-one loading fits supply identified tangent covariance. Seven dispersion
or information references compare retained dispersion directions and verify
normalized NB-to-Poisson density and curvature bounds. A better limiting
optimum uses the original model's nominal parameter count for AIC/BIC.
Neither reference covariance nor coefficients enter the runtime fitter.

The adverse-input tests reject changed data, identified estimate errors,
nonstationary points, unresolved weak directions, indefinite information,
missing genuine warnings and unstable Richardson extrapolation. Independent
80-digit recurrence calculations verify the NB density bounds.

## Remaining discrepancies

Every remaining case is listed in [STAGE_2_FAILURES.json](STAGE_2_FAILURES.json),
with its failed comparison and cause. The report and coverage count expected
numerical failures as failed corpus cases; no failure is deleted or counted
as a pass.

| Module | First failed comparison | Cases |
| --- | --- | ---: |
| gam | mapped identified coefficients | 7 |
| gam | runtime warning correspondence | 3 |
| gam | newdata link prediction | 1 |
| gam | smoothing selection criterion | 2 |
| gam | mapped identified coefficient covariance | 1 |
| gam | newdata identified term | 1 |
| gam | fitted values | 1 |
| tmb | cond fixed effect covariance | 1 |
| tmb | normalized laplace log likelihood | 13 |
| tmb | cond fixed effect coefficients | 8 |
| tmb | runtime warning correspondence | 4 |
| tmb | disp fixed effect coefficients | 1 |
| tmb | disp prediction | 1 |

The first failing comparison does not certify later fields. The per-case
report includes additional identified discrepancies when independent stored
investigations measured them. Observed failure modes include finite stopping
precision, differing smoothing or dispersion minima, weak partial Poisson
limits and separation with highly correlated random slopes. Extreme R scores
or dispersion values alone do not establish exclusive responsibility for a
likelihood mismatch.

The final integrated comparisons record 125 better optima: 115 passing cases and
10 cases that still fail another required comparison. Of these, four are GAMs
and 121 are GLMMs. The final integrated run records all 2,400 objective properties before
asserting the objective so failed objective comparisons remain auditable too.
Stronger independently verified R references and all original attempts remain
separately auditable.

## Performance

Five representative warm public fits are timed on the same machine. Each
value is a median of three repetitions; R package/process startup is excluded.

| Design | Observations | Python seconds | R seconds | Python / R |
| --- | ---: | ---: | ---: | ---: |
| gam_0000: gam / tp / gaussian / REML | 137 | 0.006624 | 0.009000 | 0.74 |
| gam_0015: gam / te / binomial / REML | 145 | 0.101601 | 0.014000 | 7.26 |
| gam_0033: gam / ps / Gamma / REML | 98 | 0.013544 | 0.006000 | 2.26 |
| tmb_0001: glmmTMB / nbinom1 / zi ~0 / disp ~1 / (1\|g) | 89 | 0.053085 | 0.032000 | 1.66 |
| tmb_0068: glmmTMB / nbinom2 / zi ~z + x / disp ~1 / (x\|g) | 108 | 0.349171 | 0.068000 | 5.13 |

The machine architecture is ARM64 and Python is 3.12.13.
[Raw timing results](benchmark_stage2.json) record the selected specifications.


## Packaging and limitations

[The packaging audit](STAGE_2_PACKAGING.md) records clean installations,
platform and license metadata, artifact hashes and an isolated sdist rebuild.

Runtime dependencies remain NumPy, SciPy, pandas, formulaic and statsmodels.
Polars input is supported when installed. The build bundles the fixed
OpenBLAS numerical backend and its native dependency notices, using platform
wheel tags. Clean execution is verified on macOS ARM64; Linux and Windows
native packaging layouts are checked but execution is not claimed. The MIT
license covers rparity's own implementation; bundled third-party licenses
remain applicable.

GAM AIC omits mgcv's smoothing-uncertainty correction for RE/ML. Approximate
tests with multiple random-effect smooths have an observed statistic
discrepancy outside this corpus. Multidimensional TP native coefficient
coordinates can differ despite equivalent function spaces. Above 2,000 unique
TP locations, deterministic subsampling differs from the oracle convention.
Dense fitting targets moderate datasets. Custom knots, shared smoothing
parameters and smoothing-selection uncertainty are deferred.

No optional Stage 2 family or large-data extension is claimed: bam, nb/tw/betar
GAMs and factor-smooth bases remain deferred, as do tweedie, truncated and
hurdle GLMM families. No Stage 3 implementation or PyPI upload is performed.

## Audit artifacts

`validation.json` records sanitized final outcomes and the source hash.
`STAGE_1_REGRESSION_BASELINE.json` fixes Stage 1 acceptance.
The evidence files cover [GAM variance boundaries](STAGE_2_GAM_BOUNDARY_EVIDENCE.json),
[ordinary smooth limits](STAGE_2_GAM_VANISHING_EVIDENCE.json),
[joint gauge limits](STAGE_2_GAM_GAUGE_LIMIT_EVIDENCE.json),
[GAM reference precision](STAGE_2_GAM_REFERENCE_EVIDENCE.json),
[TMB boundaries and correlation information](STAGE_2_TMB_POLICY_EVIDENCE.json),
[TMB dispersion references](STAGE_2_TMB_DISPERSION_REFERENCE_EVIDENCE.json),
and [TMB stationary references](STAGE_2_TMB_REFERENCE_EVIDENCE.json). `CLEANROOM.md` records public numeric oracle use and
incidental unrequested search snippets, which were not implementation sources.
`REFERENCES.md` identifies the mathematical and public behavioral specifications.

Validated runtime source SHA-256 (including version 0.2.0):
`e6cef324cb542c557bd531b48cf5e9f73a95abbc6c59daedbf66ca66c23882f4`.
