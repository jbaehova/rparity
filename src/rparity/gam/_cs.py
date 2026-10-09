"""Natural cubic shrinkage penalties with reproducible null eigenvectors.

The roughness matrix is D.T B^{-1} D, where D takes differences of adjacent
secant slopes and B integrates piecewise linear second derivatives. A cubic
shrinkage penalty assigns distinct positive eigenvalues to its two-dimensional
null space. Consequently, changing the floating-point eigensolver can change
the model, even when the original roughness matrices agree to roundoff.

Only this degenerate eigensystem uses the portable OpenBLAS numerical backend.
The construction uses mathematical spline identities and generic LAPACK APIs;
it never imports R or uses reference observations at runtime.
"""
from __future__ import annotations

import ctypes
import math
from fractions import Fraction
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.linalg.lapack import dptsv

from rparity._openblas import load_library


def _exact_fma(a: float, b: float, c: float) -> float:
    """Return a*b+c with one rounding on Python versions before math.fma."""
    return float(Fraction(float(a)) * Fraction(float(b)) + Fraction(float(c)))


_fma = getattr(math, "fma", _exact_fma)


@lru_cache(maxsize=1)
def _eigen_routine() -> Any:
    """Load the packaged LP64 LAPACKE interface without host library searches."""
    dll = load_library()
    try:
        function = dll.scipy_LAPACKE_dsyevr
    except AttributeError as exc:
        raise RuntimeError(
            "The bundled numerical backend lacks the required LP64 LAPACKE "
            "dsyevr interface. Reinstall rparity."
        ) from exc
    function.argtypes = [
        ctypes.c_int, ctypes.c_char, ctypes.c_char, ctypes.c_char, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
    ]
    function.restype = ctypes.c_int
    return function


def _eigh_openblas(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """An ascending symmetric eigensystem from the same portable backend."""
    array = np.array(matrix, dtype=np.float64, order="F", copy=True)
    n = len(array)
    values = np.empty(n, dtype=np.float64)
    vectors = np.empty((n, n), dtype=np.float64, order="F")
    support = np.empty(2 * n, dtype=np.int32)
    count = ctypes.c_int()
    info = _eigen_routine()(
        102, b"V", b"A", b"L", n, array.ctypes.data, n,
        0.0, 0.0, 0, 0, 0.0, ctypes.byref(count), values.ctypes.data,
        vectors.ctypes.data, n, support.ctypes.data,
    )
    if info != 0 or count.value != n:
        raise np.linalg.LinAlgError(f"Cubic shrinkage eigensystem failed: LAPACK info={info}.")
    return values, vectors


def cubic_roughness_penalty(knots: np.ndarray) -> np.ndarray:
    """Integrate squared second derivatives in natural cubic cardinal coordinates.

    Ordered fused contractions preserve the boundary identities with a single
    rounding per multiply-add. They also keep the numerical ordering of the
    two clustered null eigenvalues reproducible before shrinkage is applied.
    """
    x = np.asarray(knots, dtype=float)
    h = np.diff(x)
    k = len(x)
    if x.ndim != 1 or k < 3 or np.any(h <= 0) or not np.all(np.isfinite(x)):
        raise ValueError("Cubic knots must be a finite increasing vector of length at least three.")
    difference = np.zeros((k - 2, k))
    for j in range(k - 2):
        difference[j, j:j + 3] = [1 / h[j], -1 / h[j] - 1 / h[j + 1], 1 / h[j + 1]]
    diagonal = (h[:-1] + h[1:]) / 3
    if k == 3:
        interior = difference / diagonal[:, None]
    else:
        _, _, interior, info = dptsv(diagonal, h[1:-1] / 6, difference)
        if info != 0:
            raise np.linalg.LinAlgError(f"Cubic derivative system failed: LAPACK info={info}.")
    roughness = np.empty((k, k))
    for i in range(k):
        # The first interior boundary row has two nonzero slope coefficients.
        # Its right contribution precedes the negative central contribution.
        ordering = (i, i - 1) if i == 1 else (i - 1, i - 2, i)
        rows = [j for j in ordering if 0 <= j < k - 2]
        for column in range(k):
            value = 0.0
            for j in rows:
                value = _fma(difference[j, i], interior[j, column], value)
            roughness[i, column] = value
    return (roughness + roughness.T) / 2


def cubic_shrinkage_penalty(knots: np.ndarray) -> np.ndarray:
    """Give the two cubic null directions successively smaller positive weights."""
    values, vectors = _eigh_openblas(cubic_roughness_penalty(knots))
    values[0], values[1] = values[2] * 0.01, values[2] * 0.1
    penalty = (vectors * values) @ vectors.T
    return (penalty + penalty.T) / 2
