"""Covariance specifications for generalized least squares.

These are independent implementations of the covariance models described in
Pinheiro and Bates (2000), chapter 5. Specifications are immutable and may be
reused across fits.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Correlation:
    """A within-group correlation specification."""

    kind: str
    value: float | Sequence[float] | None = None
    form: str = "~ 1"
    fixed: bool = False
    p: int = 0
    q: int = 0


@dataclass(frozen=True)
class Variance:
    """A residual standard deviation multiplier specification."""

    kind: str
    value: float | Sequence[float] | Mapping[str, float] | None = None
    form: str = "~ 1"
    fixed: float | Mapping[str, float] | None = None


def corAR1(value: float = 0.0, form: str = "~ 1", fixed: bool = False) -> Correlation:
    """Specify AR(1) correlation rho**abs(time difference), for integer times."""
    if not -1 < value < 1:
        raise ValueError("AR(1) correlation must be strictly between -1 and 1")
    return Correlation("AR1", value, form, fixed)


def corCompSymm(value: float = 0.0, form: str = "~ 1", fixed: bool = False) -> Correlation:
    """Specify a common correlation for every distinct pair within a group."""
    if not -1 < value < 1:
        raise ValueError("Compound symmetry correlation must be between -1 and 1")
    return Correlation("CompSymm", value, form, fixed)


def corSymm(
    value: Sequence[float] | None = None, form: str = "~ 1", fixed: bool = False
) -> Correlation:
    """Specify an unrestricted positive-definite within-group correlation matrix.

    Initial correlations follow the lower triangle in column order. Optimization
    uses a Cholesky partial-correlation parameterization (Pinheiro and Bates, 1996).
    """
    return Correlation("Symm", value, form, fixed)


def corARMA(
    value: Sequence[float] | float | None = None,
    form: str = "~ 1",
    p: int = 0,
    q: int = 0,
    fixed: bool = False,
) -> Correlation:
    """Specify stationary ARMA(p, q) correlation on distinct integer times.

    Values contain AR coefficients followed by MA coefficients, using the
    convention y[t] = sum(phi*y[t-k]) + e[t] + sum(theta*e[t-k]).
    """
    if not isinstance(p, int) or not isinstance(q, int) or p < 0 or q < 0 or p + q == 0:
        raise ValueError("p and q must be nonnegative integers with p + q > 0")
    return Correlation("ARMA", value, form, fixed, p, q)


def varIdent(
    value: float | Sequence[float] | Mapping[str, float] | None = None,
    form: str = "~ 1",
    fixed: Mapping[str, float] | None = None,
) -> Variance:
    """Specify group-specific residual SD ratios, relative to a reference level."""
    return Variance("Ident", value, form, fixed)


def varPower(
    value: float | Mapping[str, float] = 0.0,
    form: str = "~ fitted(.)",
    fixed: float | Mapping[str, float] | None = None,
) -> Variance:
    """Specify SD multipliers abs(covariate)**power, optionally by group."""
    return Variance("Power", value, form, fixed)


def varExp(
    value: float | Mapping[str, float] = 0.0,
    form: str = "~ fitted(.)",
    fixed: float | Mapping[str, float] | None = None,
) -> Variance:
    """Specify SD multipliers exp(power*covariate), optionally by group."""
    return Variance("Exp", value, form, fixed)
