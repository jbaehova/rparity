# rparity

Stage 1 is in progress. This Python package targets R-style mixed models,
inference, marginal means, and generalized least squares. Its numerical
coverage is documented in the [validation table](coverage.md).

The runtime uses NumPy, SciPy, pandas, formulaic, and statsmodels. R is needed
only to regenerate the development oracle. No R package source is used in
this MIT-licensed implementation.

Public APIs remain provisional until the Stage 1 acceptance criteria pass.
