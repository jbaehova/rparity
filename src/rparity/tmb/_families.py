"""Normalized observation likelihoods and analytic predictor derivatives.

Negative-binomial parameterizations follow Brooks et al. (2017), table 1.
Beta regression follows Ferrari and Cribari-Neto (2004). Derivatives are
derived from those densities rather than from any target implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import special

Array = NDArray[np.float64]


@dataclass
class Terms:
    """Observation values and derivatives needed by implicit Laplace gradients."""

    value: Array
    score: Array
    curvature: Array
    third: Array
    disp_score: Array
    score_disp: Array
    curvature_disp: Array
    zi_score: Array
    score_zi: Array
    curvature_zi: Array


def inverse_link(eta: Array, link: str) -> Array:
    """Transform the conditional predictor to the response mean."""
    if link == "identity":
        return eta.copy()
    if link == "log":
        return np.exp(np.clip(eta, -700, 700))
    if link == "logit":
        return np.asarray(special.expit(eta), dtype=float)
    raise ValueError(f"Unsupported link: {link}")


def _gamma_differences(size: Array, y: Array) -> tuple[Array, Array, Array, Array]:
    """Stable gamma and polygamma differences at the Poisson limit.

    Integer-count recurrence avoids subtracting two nearly equal special
    functions when NB sizes become large. This is the exact gamma recurrence,
    not a changed likelihood or a tolerance relaxation.
    """
    gamma = np.asarray(special.gammaln(size + y) - special.gammaln(size), dtype=float)
    digamma = np.asarray(special.digamma(size + y) - special.digamma(size), dtype=float)
    trigamma = special.polygamma(1, size + y) - special.polygamma(1, size)
    tetragamma = special.polygamma(2, size + y) - special.polygamma(2, size)
    selected = (size > 1e4) & (y <= 10000) & (y == np.floor(y))
    if np.any(selected):
        r, counts = size[selected], y[selected]
        g, d, t, p = (np.zeros_like(r) for _ in range(4))
        for j in range(int(np.max(counts, initial=0))):
            active = counts > j
            denominator = r[active] + j
            g[active] += np.log(denominator)
            d[active] += 1 / denominator
            t[active] -= 1 / denominator**2
            p[active] += 2 * (1 / denominator) ** 3
        gamma[selected], digamma[selected], trigamma[selected], tetragamma[selected] = g, d, t, p
    return gamma, digamma, trigamma, tetragamma


def _count_recurrence(inverse_size: Array, y: Array) -> tuple[Array, Array, Array, Array]:
    """Integer gamma recurrence expressed in dimensionless size ratios.

    Returning log corrections and scaled derivative sums avoids both powers
    of a huge NB size and cancellation of the matching log-size terms.
    """
    correction, first, second, third = (np.zeros_like(inverse_size) for _ in range(4))
    for j in range(int(np.max(y, initial=0))):
        active = y > j
        ratio = 1 / (1 + j * inverse_size[active])
        correction[active] += np.log1p(j * inverse_size[active])
        first[active] += ratio
        second[active] += ratio**2
        third[active] += ratio**3
    return correction, first, second, third


def observation_terms(
    eta: Array,
    delta: Array,
    zeta: Array | None,
    y: Array,
    trials: Array,
    weights: Array,
    family: str,
) -> Terms:
    """Return normalized negative log densities and exact eta derivatives.

    ``delta`` is log variance, NB dispersion, or beta precision. The fitter
    doubles the public Gaussian log-standard-deviation predictor before this
    private likelihood call. For zero
    mixtures the observed curvature can be negative away from the mode;
    replacing it by Fisher information would change the Laplace likelihood.
    """
    zeros = np.zeros_like(eta)
    disp_score = zeros.copy()
    score_disp = zeros.copy()
    curvature_disp = zeros.copy()
    phi = np.exp(np.clip(delta, -700, 700))
    if family == "gaussian":
        residual = eta - y
        value = (np.log(2 * np.pi) + delta + residual**2 / phi) / 2
        score = residual / phi
        curvature = np.ones_like(eta) / phi
        third = zeros.copy()
        disp_score = (1 - residual**2 / phi) / 2
        score_disp = -score
        curvature_disp = -curvature
    elif family == "binomial":
        mu = np.asarray(special.expit(eta), dtype=float)
        k = y * trials
        value = trials * np.logaddexp(0, eta) - k * eta
        value -= special.gammaln(trials + 1) - special.gammaln(k + 1)
        value += special.gammaln(trials - k + 1)
        score = trials * (mu - y)
        curvature = trials * mu * (1 - mu)
        third = curvature * (1 - 2 * mu)
    elif family == "poisson":
        mu = np.exp(np.clip(eta, -700, 700))
        value = mu - y * eta + special.gammaln(y + 1)
        score = mu - y
        curvature = mu.copy()
        third = mu.copy()
    elif family == "nbinom2":
        mu = np.exp(np.clip(eta, -700, 700))
        denominator = phi + mu
        gamma, digamma, _, _ = _gamma_differences(phi, y)
        log_ratio = np.log1p(mu / phi)
        value = -gamma + special.gammaln(y + 1) - y * (eta - delta)
        value += (phi + y) * log_ratio
        score = (mu - y) / (1 + mu / phi)
        curvature = mu / (1 + mu / phi) * ((phi + y) / denominator)
        third = curvature * (phi - mu) / denominator
        disp_score = -phi * (digamma - log_ratio + (mu - y) / denominator)
        score_disp = score * (mu / denominator)
        curvature_disp = curvature * (1 + phi / (phi + y) - 2 * phi / denominator)
        selected = (phi > 1e4) & (y <= 10000) & (y == np.floor(y))
        if np.any(selected):
            correction, _, _, _ = _count_recurrence(1 / phi[selected], y[selected])
            relative_mean = mu[selected] / phi[selected]
            relative_log = np.divide(
                np.log1p(relative_mean),
                relative_mean,
                out=np.ones_like(relative_mean),
                where=relative_mean != 0,
            )
            value[selected] = (
                special.gammaln(y[selected] + 1)
                - y[selected] * eta[selected]
                + mu[selected] * relative_log
                + y[selected] * log_ratio[selected]
                - correction
            )
    elif family == "nbinom1":
        mu = np.exp(np.clip(eta, -700, 700))
        size = mu / phi
        logdenom = np.log1p(phi)
        gamma, digamma, trigamma, tetragamma = _gamma_differences(size, y)
        digamma -= logdenom
        value = -gamma
        value += special.gammaln(y + 1) + size * logdenom - y * (delta - logdenom)
        score = -size * digamma
        curvature = score - size**2 * trigamma
        third = score - 3 * size**2 * trigamma - size**3 * tetragamma
        selected = (size > 1e4) & (y <= 10000) & (y == np.floor(y))
        if np.any(selected):
            correction, first, second, third_sum = _count_recurrence(
                phi[selected] / mu[selected], y[selected]
            )
            relative_log = np.divide(
                logdenom[selected],
                phi[selected],
                out=np.ones_like(phi[selected]),
                where=phi[selected] != 0,
            )
            scaled_log = mu[selected] * relative_log
            value[selected] = (
                special.gammaln(y[selected] + 1)
                - y[selected] * eta[selected]
                + scaled_log
                + y[selected] * logdenom[selected]
                - correction
            )
            score[selected] = scaled_log - first
            curvature[selected] = score[selected] + second
            third[selected] = score[selected] + 3 * second - 2 * third_sum
        disp_score = -score + (mu - y) / (1 + phi)
        score_disp = -curvature + mu / (1 + phi)
        curvature_disp = -third + mu / (1 + phi)
    elif family == "beta":
        mu = np.asarray(special.expit(eta), dtype=float)
        # Exact zero observations belong to the structural mixture component.
        safe_y = np.where(y == 0, 0.5, y)
        logy, log1y = np.log(safe_y), np.log1p(-safe_y)
        a, b = mu * phi, (1 - mu) * phi
        first = mu * (1 - mu)
        second = first * (1 - 2 * mu)
        third_mu = first * (1 - 6 * first)
        dg = special.digamma(a) - special.digamma(b) - logy + log1y
        tg = special.polygamma(1, a) + special.polygamma(1, b)
        pg = special.polygamma(2, a) - special.polygamma(2, b)
        value = special.gammaln(a) + special.gammaln(b) - special.gammaln(phi)
        value -= (a - 1) * logy + (b - 1) * log1y
        score = phi * first * dg
        curvature = phi * second * dg + phi**2 * first**2 * tg
        third = phi * third_mu * dg + 3 * phi**2 * first * second * tg
        third += phi**3 * first**3 * pg
        disp_score = phi * (
            mu * special.digamma(a)
            + (1 - mu) * special.digamma(b)
            - special.digamma(phi)
            - mu * logy
            - (1 - mu) * log1y
        )
        cross = mu * special.polygamma(1, a) - (1 - mu) * special.polygamma(1, b)
        score_disp = score + phi**2 * first * cross
        curvature_disp = curvature + phi**2 * second * cross + phi**2 * first**2 * tg
        curvature_disp += (
            phi**3 * first**2 * (mu * special.polygamma(2, a) + (1 - mu) * special.polygamma(2, b))
        )
    else:
        raise ValueError(f"Unsupported family: {family}")

    zi_score, score_zi, curvature_zi = zeros.copy(), zeros.copy(), zeros.copy()
    if zeta is not None:
        probability = np.asarray(special.expit(zeta), dtype=float)
        is_zero = y == 0
        zi_score = probability.copy()
        value += np.logaddexp(0, zeta)
        if family == "beta":
            value[is_zero] = np.logaddexp(0, -zeta[is_zero])
            zi_score[is_zero] -= 1
            for derivative in (score, curvature, third, disp_score, score_disp, curvature_disp):
                derivative[is_zero] = 0
        else:
            # Base zero density is recovered before the mixture contribution.
            base_value = value[is_zero] - np.logaddexp(0, zeta[is_zero])
            q = np.asarray(special.expit(-zeta[is_zero] - base_value), dtype=float)
            value[is_zero] = np.logaddexp(0, zeta[is_zero]) - np.logaddexp(
                zeta[is_zero], -base_value
            )
            s, h, t = score[is_zero].copy(), curvature[is_zero].copy(), third[is_zero].copy()
            d = disp_score[is_zero].copy()
            sd, hd = score_disp[is_zero].copy(), curvature_disp[is_zero].copy()
            mixture = q * (1 - q)
            score[is_zero] = q * s
            curvature[is_zero] = q * h - mixture * s**2
            third[is_zero] = q * t - 3 * mixture * s * h + mixture * (1 - 2 * q) * s**3
            disp_score[is_zero] = q * d
            score_disp[is_zero] = q * sd - mixture * s * d
            curvature_disp[is_zero] = q * hd - mixture * h * d - 2 * mixture * s * sd
            curvature_disp[is_zero] += mixture * (1 - 2 * q) * s**2 * d
            zi_score[is_zero] += q - 1
            score_zi[is_zero] = -mixture * s
            curvature_zi[is_zero] = -mixture * h + mixture * (1 - 2 * q) * s**2
    return Terms(
        *[
            np.asarray(term * weights, dtype=float)
            for term in (
                value,
                score,
                curvature,
                third,
                disp_score,
                score_disp,
                curvature_disp,
                zi_score,
                score_zi,
                curvature_zi,
            )
        ]
    )
