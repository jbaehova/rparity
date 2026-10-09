---
title: "rparity numerical accuracy, supported scope and known limits"
description: Review R comparison evidence and known numerical differences for rparity mixed models, GAMs and distributional models. Understand uncertainty and platform validation.
---

# Numerical accuracy and limitations

rparity implements statistical methods independently in Python and compares
their outputs against recorded observations from R packages. Comparisons
cover estimates, covariance, predictions and inference, with declared
tolerances for each quantity. A passing case establishes agreement for that
case and configuration.

## Recorded evidence

The synthetic corpus records **6,155 passes out of 6,256 cases**. Its
**101 unresolved differences remain recorded as failed comparisons**.
Sixteen examples from public R-package documentation provide additional
checks. Use the [option-level coverage](coverage.md) to see which model
configurations are covered.

| Model group | Passing comparisons | Recorded differences |
| --- | ---: | ---: |
| Mixed models, GLS and inference | 3,799 / 3,856 | 57 |
| GAMs | 1,184 / 1,200 | 16 |
| Distributional models | 1,172 / 1,200 | 28 |

Detailed evidence is kept in the repository's
[development directory](https://github.com/jbaehova/rparity/tree/main/development):
[mixed-model report](https://github.com/jbaehova/rparity/blob/main/development/reports/STAGE_1_REPORT.md),
[GAM and distributional-model report](https://github.com/jbaehova/rparity/blob/main/development/reports/STAGE_2_REPORT.md),
and [case-level discrepancies](https://github.com/jbaehova/rparity/blob/main/development/reports/STAGE_2_FAILURES.json).
This material documents reproducibility and numerical exceptions. It is not
required to fit a model or use the package.

## Before interpreting your fit

- Review convergence warnings and the model summary. An optimizer success
  flag or finite likelihood alone does not establish reliable uncertainty.
- Inspect singular random-effect covariance and weak zero-inflation or
  dispersion components. Their individual coefficients and Wald tests can
  become poorly identified.
- For GAMs, check basis size and residual patterns. Coefficients depend on
  the spline coordinates; fitted functions can agree while native coefficient
  coordinates differ.
- Choose the prediction and inference scale explicitly. Setting random
  effects to zero differs from integrating over their distribution.
- Use sum contrasts for Type III main-effect tests in interaction models.

The [migration guides](guides/r-packages-in-python.md) state the relevant
options and limits for each model family. GAM standard errors condition on
the fitted smoothing parameters. Distributional-model prediction standard
errors are unavailable. Dense numerical calculations target models with
modest basis dimensions and small or medium grouping structures.

## Runtime and platform scope

The fitting and inference runtime uses no R installation or `rpy2`.
The published wheel has been executed and checked on macOS ARM64. Other
platforms still need execution validation. Cubic shrinkage smooths use a
bundled native OpenBLAS backend, with dependency license notices included in
the wheel. The Python implementation is MIT licensed.

R is used only to regenerate development references. See the
[clean-room record](https://github.com/jbaehova/rparity/blob/main/development/CLEANROOM.md)
for the implementation and reference-generation boundaries.
