"""Generalized least squares with nlme-compatible covariance specifications."""

from ._fit import GLSResult, gls
from ._structures import (
    Correlation,
    Variance,
    corAR1,
    corARMA,
    corCompSymm,
    corSymm,
    varExp,
    varIdent,
    varPower,
)

__all__ = [
    "Correlation",
    "GLSResult",
    "Variance",
    "corAR1",
    "corARMA",
    "corCompSymm",
    "corSymm",
    "gls",
    "varExp",
    "varIdent",
    "varPower",
]
