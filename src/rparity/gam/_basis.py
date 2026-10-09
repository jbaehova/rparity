"""Spline spaces constructed from mathematical definitions, without R at runtime.

Natural cubic cardinal splines integrate squared second derivatives. P-splines
use B-splines and finite differences (Eilers and Marx, 1996). Thin plate bases
use a truncated eigensystem of the radial kernel (Wood, 2003). Coordinates are
internal: a nonsingular change of basis leaves fitted functions unchanged.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from math import factorial, pi

import numpy as np
from scipy.interpolate import BSpline, CubicSpline
from scipy.linalg import eigh_tridiagonal, null_space, qr
from scipy.spatial.distance import cdist
from scipy.special import gamma


@dataclass
class SplineBasis:
    """A reusable basis evaluator with its quadratic roughness penalties."""

    evaluate: Callable[[np.ndarray], np.ndarray]
    S: list[np.ndarray]
    kind: str
    k: int
    null_space_dim: int
    dim: int = 1

    def matrix(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.evaluate(np.asarray(x)), dtype=float)


def _symmetric(a: np.ndarray) -> np.ndarray:
    return (a + a.T) / 2


def _positive(a: np.ndarray) -> np.ndarray:
    """Remove negative eigenvalues attributable to floating point roundoff."""
    w, v = np.linalg.eigh(_symmetric(a))
    scale = max(float(np.max(np.abs(w))), 1e-300)
    w[np.abs(w) < scale * 1e-12] = 0
    if np.min(w) < -scale * 1e-8:
        raise ValueError("A spline roughness penalty is not positive semidefinite.")
    return _symmetric((v * np.maximum(w, 0)) @ v.T)


def scale_basis(basis: SplineBasis, x: np.ndarray) -> SplineBasis:
    """Normalize roughness relative to the training model-matrix magnitude.

    This keeps automatically optimized log smoothing parameters on useful
    numerical scales. The full quadratic form still defines the same model.
    """
    magnitude = float(np.linalg.norm(basis.matrix(x), np.inf)) ** 2
    penalties = [s / (float(np.linalg.norm(s, np.inf)) / magnitude) for s in basis.S]
    return SplineBasis(basis.evaluate, penalties, basis.kind, basis.k,
                       basis.null_space_dim, basis.dim)


def center_basis(basis: SplineBasis, x: np.ndarray) -> SplineBasis:
    """Absorb a zero empirical mean constraint by an orthonormal null space."""
    constraint = basis.matrix(x).mean(axis=0, keepdims=True)
    q = null_space(constraint)
    return SplineBasis(lambda z: basis.matrix(z) @ q,
                       [_positive(q.T @ s @ q) for s in basis.S], basis.kind,
                       q.shape[1], max(basis.null_space_dim - 1, 0), basis.dim)


def _cubic(x: np.ndarray, k: int, shrinkage: bool) -> SplineBasis:
    xx = np.asarray(x, dtype=float).reshape(-1)
    if len(np.unique(xx)) < k:
        raise ValueError("A cubic spline needs at least k distinct covariate values.")
    ordered = np.unique(xx)
    positions = 1 + (len(ordered) - 1) * np.linspace(0, 1, k)
    floor = np.floor(positions)
    left = floor.astype(int) - 1
    fraction = positions - floor
    right = np.minimum(left + 1, len(ordered) - 1)
    knots = (1 - fraction) * ordered[left] + fraction * ordered[right]
    spline = CubicSpline(knots, np.eye(k), axis=0, bc_type="natural")
    # On each interval the second derivative is linear. Two point Gaussian
    # quadrature integrates its pairwise products exactly.
    lengths = np.diff(knots)
    mids = (knots[1:] + knots[:-1]) / 2
    qpoints = np.concatenate([mids - lengths / np.sqrt(12),
                              mids + lengths / np.sqrt(12)])
    qweights = np.tile(lengths / 2, 2)
    second = spline(qpoints, 2)
    penalty = _positive(second.T @ (qweights[:, None] * second))
    if shrinkage:
        from ._cs import cubic_shrinkage_penalty

        penalty = cubic_shrinkage_penalty(knots)

    def evaluate(z: np.ndarray) -> np.ndarray:
        zz = np.asarray(z, dtype=float).reshape(-1)
        result = np.asarray(spline(np.clip(zz, knots[0], knots[-1])), dtype=float)
        for side, bound in ((zz < knots[0], knots[0]), (zz > knots[-1], knots[-1])):
            result[side] = spline(bound) + (zz[side, None] - bound) * spline(bound, 1)
        return result

    return SplineBasis(evaluate, [penalty], "cs" if shrinkage else "cr", k,
                       0 if shrinkage else 2)


def _pspline(x: np.ndarray, k: int, m: tuple[int, ...]) -> SplineBasis:
    xx = np.asarray(x, dtype=float).reshape(-1)
    order, difference = (m[0], m[1] if len(m) > 1 else m[0]) if m else (2, 2)
    degree = order + 1
    if k < degree + 1 or difference < 0 or difference >= k:
        raise ValueError("P-spline k must exceed spline degree and difference order.")
    lo, hi = float(np.min(xx)), float(np.max(xx))
    if hi <= lo:
        raise ValueError("A spline covariate must vary.")
    # The nominal data interval is slightly expanded; the remaining exterior
    # knots support a non-clamped uniform B-spline basis.
    span = hi - lo
    lo, hi = lo - span * 0.001, hi + span * 0.001
    inner_count = k - degree + 1
    inner = np.linspace(lo, hi, inner_count)
    step = (hi - lo) / (inner_count - 1)
    knots = np.concatenate([lo - np.arange(degree, 0, -1) * step, inner,
                            hi + np.arange(1, degree + 1) * step])
    spline = BSpline(knots, np.eye(k), degree, extrapolate=True)
    diff = np.diff(np.eye(k), n=difference, axis=0)
    penalty = diff.T @ diff

    def evaluate(z: np.ndarray) -> np.ndarray:
        zz = np.asarray(z, dtype=float).reshape(-1)
        result = np.asarray(spline(np.clip(zz, lo, hi)), dtype=float)
        for side, bound in ((zz < lo, lo), (zz > hi, hi)):
            result[side] = spline(bound) + (zz[side, None] - bound) * spline(bound, 1)
        return result

    return SplineBasis(evaluate, [penalty], "ps", k, difference)


def _polynomial_exponents(dim: int, m: int) -> list[tuple[int, ...]]:
    exponents = [e for e in product(range(m), repeat=dim) if sum(e) < m]
    return sorted(exponents, key=lambda e: (sum(e), tuple(reversed(e))))


def _polynomial(x: np.ndarray, exponents: list[tuple[int, ...]]) -> np.ndarray:
    return np.column_stack([np.prod(x ** np.asarray(e), axis=1) for e in exponents])


def _lanczos_vectors(a: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Orient an extreme eigensystem with fully reorthogonalized Lanczos.

    The deterministic starting vector uses the elementary congruential
    recurrence from Press et al. (1992), with its values mapped to (-1, 1).
    Public numerical observations identify the seed and the Ritz check
    interval. LAPACK's symmetric tridiagonal QR solver fixes eigenvector signs.
    """
    n = len(a)
    start = np.empty(n)
    state = 1
    for j in range(n):
        state = (106 * state + 1283) % 6075
        start[j] = 2 * state / 6075 - 1
    vectors = np.zeros((n, n))
    vectors[:, 0] = start / np.linalg.norm(start)
    diagonal: list[float] = []
    off_diagonal: list[float] = []
    interval = max(1, min(n // 10, 10))
    tolerance = np.sqrt(np.finfo(float).eps)
    for j in range(n):
        residual = a @ vectors[:, j]
        alpha = float(residual @ vectors[:, j])
        residual -= alpha * vectors[:, j]
        if j:
            residual -= off_diagonal[j - 1] * vectors[:, j - 1]
        residual -= vectors[:, :j + 1] @ (vectors[:, :j + 1].T @ residual)
        beta = float(np.linalg.norm(residual))
        diagonal.append(alpha)
        complete = j + 1 == n
        breakdown = beta < np.finfo(float).eps * max(np.linalg.norm(a, np.inf), 1)
        if complete or breakdown or (j >= k - 1 and j > 0 and j % interval == 0):
            values, tri_vectors = eigh_tridiagonal(
                diagonal, off_diagonal, lapack_driver="stev")
            selected = np.argsort(np.abs(values))[-k:]
            error = np.max(np.abs(beta * tri_vectors[-1, selected]))
            if complete or breakdown or error <= tolerance * np.max(np.abs(values)):
                selected = np.r_[np.sort(selected[values[selected] < 0]),
                                 np.sort(selected[values[selected] >= 0])[::-1]]
                return values[selected], vectors[:, :j + 1] @ tri_vectors[:, selected]
        off_diagonal.append(beta)
        vectors[:, j + 1] = residual / beta
    raise RuntimeError("Lanczos iteration did not produce an eigensystem.")


def _thin_plate(x: np.ndarray, k: int, m: tuple[int, ...]) -> SplineBasis:
    xx = np.asarray(x, dtype=float)
    if xx.ndim == 1:
        xx = xx[:, None]
    dim = xx.shape[1]
    order = m[0] if m else (dim + 1) // 2 + 1
    if 2 * order <= dim:
        raise ValueError("Thin plate derivative order must satisfy 2*m > dimension.")
    exponents = _polynomial_exponents(dim, order)
    null_dim = len(exponents)
    omit_null = len(m) > 1 and m[1] == 0
    if k <= null_dim:
        raise ValueError("Thin plate k must exceed its polynomial null space dimension.")
    shift = xx.mean(axis=0)
    knots = np.unique(xx - shift, axis=0)
    if len(knots) < k:
        raise ValueError("A thin plate spline needs at least k distinct locations.")
    if len(knots) > 2000:
        # Dense thin plate eigensystems have cubic setup cost. This deterministic
        # evenly indexed subset is documented rather than claiming R RNG parity.
        knots = knots[np.linspace(0, len(knots) - 1, 2000).astype(int)]

    def radial(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        r = cdist(a, b)
        power = 2 * order - dim
        if dim % 2:
            constant = float(gamma(dim / 2 - order)) / (
                2 ** (2 * order) * pi ** (dim / 2) * factorial(order - 1))
            return constant * r ** power
        constant = (-1) ** (order + 1 + dim // 2) / (
            2 ** (2 * order - 1) * pi ** (dim / 2)
            * factorial(order - 1) * factorial(order - dim // 2))
        return constant * r ** power * np.log(np.maximum(r, 1e-300))

    e = radial(knots, knots)
    lanczos_values, lanczos_vectors = _lanczos_vectors(e, k)
    # A dense refinement retains machine-precision function-space invariance.
    # Lanczos supplies only deterministic signs, rather than letting unrelated
    # dense eigensolver sign conventions rotate fractional-rank smooth tests.
    w, u = np.linalg.eigh(e)
    selected = np.argsort(np.abs(w))[-k:]
    selected = np.r_[np.sort(selected[w[selected] < 0]),
                     np.sort(selected[w[selected] >= 0])[::-1]]
    u = u[:, selected]
    matched = [int(np.argmin(np.abs(lanczos_values - value))) for value in w[selected]]
    signs = np.sign(np.sum(u * lanczos_vectors[:, matched], axis=0))
    u *= np.where(signs == 0, 1, signs)
    t = _polynomial(knots, exponents)
    q, _ = qr(u.T @ t, mode="full")
    radial_coef = u @ q[:, null_dim:]
    raw = np.column_stack([radial(xx - shift, knots) @ radial_coef,
                           _polynomial(xx - shift, exponents)])
    normalization = np.sqrt(np.mean(raw ** 2, axis=0))
    radial_coef /= normalization[:k - null_dim]
    penalty = _positive(radial_coef.T @ e @ radial_coef)
    width = radial_coef.shape[1] + (0 if omit_null else null_dim)
    full_penalty = np.zeros((width, width))
    full_penalty[:penalty.shape[0], :penalty.shape[1]] = penalty

    def evaluate(z: np.ndarray) -> np.ndarray:
        zz = np.asarray(z, dtype=float).reshape(-1, dim) - shift
        penalized = radial(zz, knots) @ radial_coef
        return penalized if omit_null else np.column_stack(
            [penalized, _polynomial(zz, exponents) / normalization[k - null_dim:]])

    return SplineBasis(evaluate, [full_penalty], "tp", width,
                       0 if omit_null else null_dim, dim)


def spline_basis(x: np.ndarray, kind: str = "tp", k: int = 10,
                 m: tuple[int, ...] = ()) -> SplineBasis:
    """Construct a supported univariate or thin plate regression spline."""
    if k < 3:
        raise ValueError("Spline basis dimension k must be at least three.")
    if kind in {"cr", "cs"}:
        return _cubic(x, k, kind == "cs")
    if kind == "ps":
        return _pspline(x, k, m)
    if kind == "tp":
        return _thin_plate(x, k, m)
    raise ValueError(f"Unsupported spline basis {kind!r}.")


def normal_parameterization(basis: SplineBasis, x: np.ndarray) -> SplineBasis:
    """Represent a univariate marginal by values at evenly spaced locations."""
    xx = np.asarray(x, dtype=float).reshape(-1)
    locations = np.linspace(np.min(xx), np.max(xx), basis.k)
    evaluation = basis.matrix(locations)
    change = np.linalg.inv(evaluation)
    return SplineBasis(lambda z: basis.matrix(z) @ change,
                       [_positive(change.T @ s @ change) for s in basis.S],
                       basis.kind, basis.k, basis.null_space_dim, basis.dim)


def tensor_basis(margins: list[SplineBasis], kind: str) -> SplineBasis:
    """Row-wise Kronecker basis and one Kronecker roughness penalty per margin."""
    sizes = [b.k for b in margins]
    penalties = []
    for j, margin in enumerate(margins):
        penalty = np.ones((1, 1))
        for i, size in enumerate(sizes):
            penalty = np.kron(penalty, margin.S[0] if i == j else np.eye(size))
        penalties.append(penalty)

    def evaluate(z: np.ndarray) -> np.ndarray:
        zz = np.asarray(z, dtype=float).reshape(-1, len(margins))
        result = np.ones((len(zz), 1))
        for j, margin in enumerate(margins):
            result = (result[:, :, None] * margin.matrix(zz[:, j])[:, None, :]).reshape(
                len(zz), -1)
        return result

    return SplineBasis(evaluate, penalties, kind, int(np.prod(sizes)),
                       int(np.prod([b.null_space_dim for b in margins])), len(margins))
