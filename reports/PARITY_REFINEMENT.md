# Parity refinement after v0.1.0

The checkout resolves three of the release's 60 numerical failures without changing
any synthetic input, recorded R result, field tolerance or comparison rule.
The corpus now passes 3,799 of 3,856 cases (98.52%); 57 remain actual failures.
The published v0.1.0 tag, wheel and release report retain their original baseline.
These changes are unreleased and available by building the current checkout.

## Resolved cases

| Case | Failure | Implementation correction |
| --- | --- | --- |
| lmer_0585 | Relative fitted-value precision near zero | Safeguarded interior stationary-score refinement after objective stopping. |
| glmer_0361 | A small probit fitted probability | Analytic Laplace outer score, including conditional-mode movement and determinant derivatives. |
| glmer_0107 | Fixed-effect covariance precision | Richardson differentiation of the analytic score, avoiding subtraction of nearly equal likelihood values. |

No replacement R outputs were selected to obtain these passes. The corresponding
expected-failure markers were removed. All other reviewed failure markers remain.
There are 56 strict markers and two environment-dependent markers, lmer_0059 and
lmer_0364. The latter passes locally but remains tracked because of its prior Linux
observation. The reviewed union of remaining IDs has 58 entries; the local actual
failure count is 57. No claim of complete current Linux validation is made.

## Genuine GLS boundary correction

The corSymm factor used a floor on sqrt(1-cos(angle)^2), which both lost precision
and changed the covariance at nearly singular boundaries. The factor now uses
sin(angle) directly. Residual covariance factors are retained through whitening,
and the joint response/design is evaluated by augmented QR instead of normal equations.
Normalized residuals and ANOVA use the same factors.

Closed-form ML and REML tests at the last representable correlation below one verify
likelihood, scale, normalized residuals and extraction methods without regularization.
All three reviewed corSymm cases now attain likelihoods at least as high as R,
including gls_0173's former objective deficit. Their coefficients still fail the
strict comparison, so none is counted as resolved. Nearly singular endpoints and
applicable Python warnings remain visible.

## Why other differences remain

- Thirteen LMER prediction cases differ from R by only about 1e-9 to 2.2e-8.
  Their likelihood criteria agree to about 1.4e-13. Python covariance scores are
  near stationarity, while the recorded R endpoints retain scores around 1e-7 to
  1.2e-6. Moving Python away from its optimum to reproduce this stopping error is
  not an implementation correction.
- The 18 remaining GLMER cases include near-zero coefficient differences and
  unstable R finite-difference Hessians. Public R optimizer gradients reach
  roughly 4e-4 to 3e-3 in several difficult cases. Uniform tighter inner tolerances
  were probed without replacing fixtures: 1e-15 caused 2 of 20 fit errors, while
  1e-16 caused 20 of 20. Other covariance estimates worsened. This is not a reliable
  global reference improvement.
- Thirteen GLS cases and one GLS EMM contrast retain relative precision or nearly
  singular endpoint differences. Tiny variance/prediction errors occur despite
  stationary Python scores and likelihood agreement near floating-point precision.
  The three corSymm boundary coefficient solutions remain unstable. The near-unit
  MA coefficient also remains different.
- Ten LMER/inference/ANOVA cases require an R-specific optimizer convergence warning
  even though Python converges. glmer_0312 requires R's singular warning although
  Python finds a better interior fit. Fabricated warnings were not added.
- anova_0337 is quasi-separated: R and Python likelihoods agree to about 2e-11,
  while R's coefficient covariance condition number is about 6.8e12. Diverging
  coefficients and numerical hypothesis rank make its strict Wald table unstable.
  anova_0652 retains a small intercept/Wald difference caused by its R endpoint.

Detailed numerical observations, including score comparisons and R public optimizer
fields, are in [precision diagnostics](precision_diagnostics.json). All remaining IDs
and original classifications are in [known failures](../tests/golden/known_failures.json).

## Current coverage

| Module | Cases | Passed | Failed |
| --- | ---: | ---: | ---: |
| anova | 700 | 697 | 3 |
| emm | 753 | 752 | 1 |
| glmer | 603 | 585 | 18 |
| gls | 600 | 587 | 13 |
| inference | 600 | 594 | 6 |
| lmer | 600 | 584 | 16 |

Six cases have a Python objective improvement above 1e-6; this does not override
coefficient or required-warning failures. The original release had five such cases.

## Local validation

- Full pytest: 3,931 passed, 57 expected failures and one genuine environment-dependent
  XPASS, with no unexpected failure or skipped test. The JSON records that XPASS as
  an actual pass, giving 3,932 successful test outcomes in total.
- All 14 representative R examples pass.
- Seven independent GLMER derivative regressions and two closed-form GLS boundary
  regressions pass. All 3,856 golden cases execute with the original assertions.
- Ruff and mypy pass, including isolated local Python 3.11 typing.
- Local Python 3.11 checks of the resolved and environment-dependent IDs pass.
- An optional Python 3.13 targeted check was not executed because dependency
  downloads stalled. The complete local suite uses Python 3.12; no current
  six-environment matrix claim is made.
- MkDocs strict build, wheel build and fresh-environment README execution pass.
- Tests run locally. No GitHub Actions test was started for this refinement.

The validated runtime fingerprint is `e037083415d69ff8ecb5e31a5c83a3dfa58702c89a1126f3127a169a3889f4de`.
Current case outcomes are in [validation.json](validation.json), and option-level
coverage is regenerated in [coverage](../docs/coverage.md).

No target R package source or function body was inspected. Additional R numerical
observations use only oracle/run_case.R and public optimizer outputs. Stage 2 and
optional extensions are outside this refinement.
