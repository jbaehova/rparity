"""Cubic shrinkage mathematical identities and public function-space parity."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import CubicSpline

from rparity.gam._cs import (
    _eigh_openblas,
    _exact_fma,
    cubic_roughness_penalty,
    cubic_shrinkage_penalty,
)
from rparity.gam._formula import build_design


def test_single_rounding_preserves_a_cancelled_product():
    assert _exact_fma(2**27 + 1, 2**27 - 1, -(2**54)) == -1


@pytest.mark.parametrize("k", [3, 5, 6, 7, 8, 12])
def test_cubic_roughness_integrates_squared_second_derivatives(k):
    knots = np.sort(np.random.default_rng(138 + k).uniform(-2, 3, k))
    penalty = cubic_roughness_penalty(knots)
    spline = CubicSpline(knots, np.eye(k), bc_type="natural")
    h = np.diff(knots)
    mid = (knots[1:] + knots[:-1]) / 2
    locations = np.r_[mid - h / np.sqrt(12), mid + h / np.sqrt(12)]
    second = spline(locations, 2)
    integrated = second.T @ (np.tile(h / 2, 2)[:, None] * second)
    np.testing.assert_allclose(penalty, integrated, rtol=1e-12, atol=1e-10)
    for linear_function in [np.ones(k), knots]:
        np.testing.assert_allclose(penalty @ linear_function, 0,
                                   atol=1e-12 * np.linalg.norm(penalty))


def test_shrinkage_retains_curved_eigenvalues_and_penalizes_both_null_directions():
    knots = np.array([-2, -1.2, -0.3, 0.4, 1.1, 2.4])
    ordinary = cubic_roughness_penalty(knots)
    rough_values = np.linalg.eigvalsh(ordinary)
    shrinkage = cubic_shrinkage_penalty(knots)
    values = np.linalg.eigvalsh(shrinkage)
    np.testing.assert_allclose(values[2:], rough_values[2:], rtol=1e-12)
    np.testing.assert_allclose(values[:2], rough_values[2] * np.array([0.01, 0.1]),
                               rtol=1e-11)
    assert np.ones(len(knots)) @ shrinkage @ np.ones(len(knots)) > 0
    assert knots @ shrinkage @ knots > 0


def test_portable_eigensystem_reconstructs_a_symmetric_matrix():
    rng = np.random.default_rng(283)
    factor = rng.normal(size=(8, 8))
    matrix = factor.T @ factor
    values, vectors = _eigh_openblas(matrix)
    np.testing.assert_allclose(vectors @ np.diag(values) @ vectors.T, matrix,
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(8), atol=1e-12)


@pytest.mark.parametrize("knots", [[1, 2], [1, 1, 2], [2, 1, 3], [0, 1, np.inf]])
def test_invalid_cubic_knots_raise(knots):
    with pytest.raises(ValueError, match="Cubic knots"):
        cubic_roughness_penalty(np.array(knots))


def _cs_fixtures():
    directory = Path(__file__).parent / "golden/gam"
    return [path for path in sorted(directory.glob("*.json"))
            if '"cs"' in json.loads(path.read_text())["spec"]["formula"]]


@pytest.mark.parametrize("path", _cs_fixtures(), ids=lambda path: f"cs-penalty-{path.stem}")
def test_shrinkage_penalties_match_public_smooth_function_spaces(path):
    fixture = json.loads(path.read_text())
    spec, reference = fixture["spec"], fixture["oracle"]["result"]
    design = build_design(spec["formula"], pd.DataFrame(spec["data"]))
    own_smooths = {smooth.label: smooth for smooth in design.smooths}
    for smooth in reference["smooths"]:
        own = own_smooths[smooth["label"]]
        if own.kind != "cs":
            continue
        indices = np.arange(smooth["first_para"] - 1, smooth["last_para"])
        reference_matrix = np.asarray(reference["X"])[:, indices]
        own_matrix = design.X[:, own.indices]
        change = np.linalg.lstsq(reference_matrix, own_matrix, rcond=1e-12)[0]
        np.testing.assert_allclose(reference_matrix @ change, own_matrix,
                                   rtol=1e-9, atol=1e-10)
        target = change.T @ np.asarray(smooth["S"][0]) @ change
        actual = design.S[own.penalty_indices[0]][np.ix_(own.indices, own.indices)]
        # Smoothing optimization absorbs the arbitrary positive normalization.
        normalization = np.sum(target * actual) / np.sum(actual**2)
        assert normalization > 0
        relative_error = np.linalg.norm(target - normalization * actual) / np.linalg.norm(target)
        assert relative_error < 1e-10
