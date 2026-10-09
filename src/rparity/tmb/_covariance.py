"""Native covariance coordinates and centered-score covariance convention.

Unstructured correlation matrices use normalized unit-diagonal Cholesky rows.
The unscaled 1e-3 score increment follows the public R optimHess convention.
Its match to the oracle covariance is established through numerical probes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def native_coordinates(
    parameters: Array,
    blocks: list[Any],
    coefficients: int,
) -> tuple[Array, Callable[[Array], tuple[Array, Array]]] | None:
    """Map lower factors to log SDs and scaled-Cholesky correlation coordinates."""
    natural = parameters.copy()
    boundary_rows: dict[tuple[int, int], Array] = {}
    cursor = coefficients
    for block in blocks:
        width = len(block.names)
        count = width * (width + 1) // 2
        lower = np.zeros((width, width))
        lower[np.tril_indices(width)] = parameters[cursor : cursor + count]
        diagonal = np.diag(lower)
        lower *= np.where(diagonal < 0, -1.0, 1.0)[None, :]
        standard_deviations = np.sqrt(np.sum(lower**2, axis=1))
        if np.any(standard_deviations < 1e-8):
            # A vanishing entire row has no finite log-SD coordinate.
            return None
        natural[cursor : cursor + width] = np.log(standard_deviations)
        correlation_cursor = cursor + width
        for i in range(1, width):
            if abs(lower[i, i]) < 1e-8:
                # At a correlation boundary the scaled Cholesky coordinate is
                # infinite. Retain its fitted limiting row direction while
                # differentiating all identifiable log-SD/fixed coordinates.
                boundary_rows[cursor, i] = lower[i, : i + 1] / standard_deviations[i]
                natural[correlation_cursor : correlation_cursor + i] = 0
            else:
                natural[correlation_cursor : correlation_cursor + i] = lower[i, :i] / lower[i, i]
            correlation_cursor += i
        cursor += count

    def transform(values: Array) -> tuple[Array, Array]:
        raw = values.copy()
        jacobian = np.eye(len(values))
        cursor = coefficients
        for block in blocks:
            width = len(block.names)
            count = width * (width + 1) // 2
            jacobian[cursor : cursor + count, cursor : cursor + count] = 0
            standard_deviations = np.exp(values[cursor : cursor + width])
            raw_cursor = cursor
            correlation_cursor = cursor + width
            for i in range(width):
                if (cursor, i) in boundary_rows:
                    factor_row = standard_deviations[i] * boundary_rows[cursor, i]
                    raw[raw_cursor : raw_cursor + i + 1] = factor_row
                    jacobian[raw_cursor : raw_cursor + i + 1, cursor + i] = factor_row
                    correlation_cursor += i
                    raw_cursor += i + 1
                    continue
                correlations = values[correlation_cursor : correlation_cursor + i]
                row = np.r_[correlations, 1.0]
                denominator = np.sqrt(1 + float(correlations @ correlations))
                factor_row = standard_deviations[i] * row / denominator
                raw[raw_cursor : raw_cursor + i + 1] = factor_row
                jacobian[raw_cursor : raw_cursor + i + 1, cursor + i] = factor_row
                for j, correlation in enumerate(correlations):
                    derivative = -standard_deviations[i] * row * correlation / denominator**3
                    derivative[j] += standard_deviations[i] / denominator
                    jacobian[raw_cursor : raw_cursor + i + 1, correlation_cursor + j] = derivative
                correlation_cursor += i
                raw_cursor += i + 1
            cursor += count
        return raw, jacobian

    return natural, transform


def centered_score_hessian(gradient: Callable[[Array], Array], parameters: Array) -> Array:
    """Symmetric central score differences with unscaled 1e-3 increments."""
    hessian = np.empty((len(parameters), len(parameters)))
    step = 1e-3
    for i in range(len(parameters)):
        direction = np.zeros_like(parameters)
        direction[i] = step
        hessian[:, i] = (gradient(parameters + direction) - gradient(parameters - direction)) / (
            2 * step
        )
    return (hessian + hessian.T) / 2
