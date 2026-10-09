# rparity v0.1.0

Stage 1 provides R-style mixed models, small-sample inference, marginal means
and generalized least squares in Python. The committed synthetic corpus passes
3,796 of 3,856 cases (98.44%). Sixty documented differences remain
counted as failures in the [validation coverage](coverage.md).

Install the wheel from the [GitHub release](https://github.com/jbaehova/rparity/releases/tag/v0.1.0).
The [README](https://github.com/jbaehova/rparity#quick-start) contains a complete
synthetic quick start. See the [Stage 1 report](https://github.com/jbaehova/rparity/blob/main/reports/STAGE_1_REPORT.md)
for exact checks, limitations and benchmarks.

NumPy and SciPy handle numerical work. pandas, formulaic and statsmodels
provide data and model support. R is needed only to regenerate the development
oracle. No target R package source was used in this MIT-licensed implementation.

Type III tests require sum contrasts for the documented main-effect interpretation.
Control contrasts need an explicit supported adjustment because `dunnettx`
is unavailable. Optional Stage 1 extensions are deferred. Stage 2 has not started.
