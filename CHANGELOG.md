# Changelog

## Unreleased

- Refine LMER interior covariance estimates with safeguarded stationary-score roots.
- Use implicit conditional-mode derivatives for the GLMER Laplace score and Richardson score-Hessian covariance, avoiding likelihood subtraction noise.
- Preserve GLS covariance factors at corSymm boundaries and evaluate whitened likelihoods by augmented QR.
- Resolve lmer_0585, glmer_0107 and glmer_0361 without modifying golden outputs or numerical tolerances.
- Add independent derivative and closed-form boundary regression checks. Keep all remaining discrepancies in failure accounting.

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
