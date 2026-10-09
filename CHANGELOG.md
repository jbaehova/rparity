# Changelog

## 0.2.0 (2026-10-09)

- Add GAMs with thin plate, cubic, shrinkage, P-spline and random-effect smooths, tensor products and `by` terms.
- Add REML, ML and GCV/UBRE smoothing selection for Gaussian, binomial, Poisson and Gamma models.
- Add GAM EDF and smooth tests, prediction standard errors, term effects, partial effects and numerical basis checks.
- Add distributional mixed models for Poisson, NB1, NB2, binomial, beta and Gaussian responses, including zero inflation and dispersion formulas.
- Add normalized observed-Hessian Laplace fitting, correlated random slopes, component results and predictions, marginal means and Type II/III Wald tests.
- Validate all 2,400 fixed Stage 2 designs: 2,356 pass (98.17%); 16 GAM and 28 mixed-model differences remain counted as failures.
- Preserve original R attempts and independently verify stationary references, identified boundary limits and covariance refinements without loosening tolerances.
- Bundle a fixed OpenBLAS backend and its third-party license notices in platform wheels. Retain five Python runtime dependencies and no runtime R dependency; verify clean wheel execution on macOS ARM64.
- Add mgcv and glmmTMB migration guides, representative R examples and Stage 2 benchmark and failure reports.
- Run tests locally. Require explicit manual dispatch for GitHub Actions tests, and restrict Pages deployment to documentation changes.

- Refine LMER interior covariance estimates with safeguarded stationary-score roots.
- Use implicit conditional-mode derivatives for the GLMER Laplace score and Richardson score-Hessian covariance, avoiding likelihood subtraction noise.
- Preserve GLS covariance factors at corSymm boundaries and evaluate whitened likelihoods by augmented QR.
- Resolve lmer_0585, glmer_0107 and glmer_0361 without modifying golden outputs or numerical tolerances.
- Add independent derivative and closed-form boundary regression checks. Keep all remaining discrepancies in failure accounting.
- Retain Stage 1 validation at 3,799/3,856 (98.52%), with 57 differences and no new regression. Preserve the v0.1.0 release baseline.

## 0.1.0 (2026-10-09)

- Add clean-room Gaussian and Laplace generalized mixed models.
- Add Satterthwaite and Kenward-Roger inference, ANOVA and ML model comparisons.
- Add marginal means, required contrasts and multiple comparisons.
- Add GLS correlation/variance structures, inference and predictions.
- Validate 3,856 synthetic R cases with 98.44% passing and 14 representative R examples.
- Document 60 strict numerical/warning differences and deferred optional extensions.
- Publish migration tables, coverage, MkDocs documentation and the release wheel.
- Add Python 3.11+ Linux/macOS CI; R remains development-only.

## 0.1.0.dev0

- Begin clean-room Stage 1 implementation and validation.
