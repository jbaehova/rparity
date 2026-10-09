# rparity v0.2.0

rparity provides R-style mixed models, GAMs and marginal means in Python.
Stage 2 adds smooth models and distributional mixed models with zero inflation
and dispersion formulas. Fitting, inference and prediction require no R or
`rpy2` process.

The fixed Stage 2 corpus passes 2,356 of 2,400 cases (98.17%). Sixteen GAM and
28 extended-model differences remain counted as failures. Stage 1 retains
3,799 of 3,856 passes (98.52%) and 57 differences, with no new regression.
See the [validation coverage](coverage.md),
[Stage 2 report](https://github.com/jbaehova/rparity/blob/main/reports/STAGE_2_REPORT.md),
and [case-level failure evidence](https://github.com/jbaehova/rparity/blob/main/reports/STAGE_2_FAILURES.json).
Original R observations and unsuccessful reference attempts remain recorded.

The immutable v0.1.0 release retains its original 3,796 passes and 60
differences. Its [Stage 1 report](https://github.com/jbaehova/rparity/blob/main/reports/STAGE_1_REPORT.md)
and the [precision refinement report](https://github.com/jbaehova/rparity/blob/main/reports/PARITY_REFINEMENT.md)
describe that baseline and the subsequent fixes.

Install a compatible platform wheel from the
[GitHub release](https://github.com/jbaehova/rparity/releases/tag/v0.2.0).
The [README](https://github.com/jbaehova/rparity#quick-start) contains executable
synthetic examples. Start with the [mgcv migration guide](migration/mgcv.md)
for GAMs or the [glmmTMB guide](migration/glmmTMB.md) for distributional mixed
models. Other migration guides cover Stage 1 models and inference.

NumPy and SciPy handle numerical work. pandas, formulaic and statsmodels
provide data and model support. Platform wheels bundle a fixed OpenBLAS
backend for cubic shrinkage penalties, with its native dependency notices.
rparity's own implementation is MIT licensed. Clean wheel execution is
verified on macOS ARM64; execution on other platforms still needs validation.
R is needed only for development oracle regeneration.

Type III tests require sum contrasts for the documented main-effect
interpretation. Numerical boundary inference has explicit case-level limits
and evidence. The migration guides document unsupported extensions and
remaining limitations. Tests run locally; GitHub Actions tests require an
explicit manual request. Pages deployment is limited to documentation changes.
