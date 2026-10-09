"""Covariance coordinate derivatives and finite-score information identities."""

from types import SimpleNamespace

import numpy as np
from scipy import linalg

from rparity.tmb._covariance import (
    centered_score_hessian,
    native_coordinates,
    richardson_score_hessian,
    weak_component_information,
)


def test_unstructured_coordinates_preserve_covariance_and_derivatives():
    parameters = np.array([0.3, -0.5, 0.7, 0.2, 0.6, -0.1, 0.3, 0.4])
    blocks = [SimpleNamespace(names=["intercept", "x", "z"])]
    native = native_coordinates(parameters, blocks, 2)
    assert native is not None
    values, transform = native
    lower = np.array([[0.7, 0, 0], [0.2, 0.6, 0], [-0.1, 0.3, 0.4]])
    np.testing.assert_allclose(np.exp(values[2:5]), np.sqrt(np.diag(lower @ lower.T)))
    reconstructed, jacobian = transform(values)
    np.testing.assert_allclose(reconstructed, parameters, atol=1e-15)
    numerical = np.column_stack([
        (transform(values + direction * 1e-6)[0] - transform(values - direction * 1e-6)[0])
        / 2e-6
        for direction in np.eye(len(values))
    ])
    np.testing.assert_allclose(jacobian, numerical, rtol=1e-8, atol=1e-10)


def test_rank_one_limit_has_no_finite_correlation_information():
    parameters = np.array([0.2, 0.5, -0.3, 0.0])
    native = native_coordinates(parameters, [SimpleNamespace(names=["intercept", "x"])], 1)
    assert native is not None
    values, transform = native
    reconstructed, jacobian = transform(values)
    np.testing.assert_allclose(reconstructed, parameters, atol=1e-15)
    # Changing an arbitrary finite placeholder cannot move an infinite
    # scaled-correlation coordinate away from its fitted limiting direction.
    direction = np.array([0.0, 0.0, 0.0, 100.0])
    np.testing.assert_array_equal(transform(values + direction)[0], reconstructed)
    np.testing.assert_array_equal(jacobian[:, -1], np.zeros(4))


def test_centered_score_information_is_exact_for_a_quadratic():
    information = np.array([[4.0, 0.7, -0.3], [0.7, 2.0, 0.2], [-0.3, 0.2, 1.0]])
    location = np.array([0.4, -0.8, 1.3])

    def gradient(parameters):
        return information @ (parameters - location)

    np.testing.assert_allclose(
        centered_score_hessian(gradient, np.array([0.3, 0.1, -0.6])),
        information, rtol=1e-12, atol=1e-12,
    )


def test_profiled_weak_component_contrast_survives_rotation():
    """A nearly flat joint contrast has positive individual diagonals."""
    rotation = np.array([[1., 1.], [-1., 1.]]) / np.sqrt(2)
    component_information = rotation @ np.diag([1e-10, 9.]) @ rotation.T
    cross = np.array([[.3, -.4]])
    information = np.block([
        [np.array([[2.]]), cross],
        [cross.T, component_information + cross.T @ cross / 2],
    ])
    parameters = np.r_[.2, rotation @ np.array([25., .5])]
    assert np.min(np.diag(information)) > 1
    assert weak_component_information(information, parameters, 1, 3)
    change = linalg.block_diag(np.eye(1), rotation)
    assert weak_component_information(change.T @ information @ change,
                                      change.T @ parameters, 1, 3)
    assert not weak_component_information(information, parameters / 100, 1, 3)
    resolved = information.copy()
    resolved[1:, 1:] += np.eye(2) * .01
    assert not weak_component_information(resolved, parameters, 1, 3)


def test_richardson_information_removes_cubic_score_step_error():
    information = np.array([[4., .3], [.3, 2.]])
    cubic = np.array([3., 7.])
    location = np.array([.6, -.2])

    def gradient(parameters):
        return information @ parameters + cubic * parameters**3

    exact = information + np.diag(3 * cubic * location**2)
    assert np.max(abs(centered_score_hessian(gradient, location) - exact)) > 1e-6
    np.testing.assert_allclose(richardson_score_hessian(gradient, location), exact,
                               rtol=1e-12, atol=1e-12)
