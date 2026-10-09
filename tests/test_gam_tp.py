"""Thin plate spline spaces, roughness, and public reference coordinates."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rparity.gam._basis import spline_basis
from rparity.gam._formula import build_design


@pytest.mark.parametrize("number", [0, 10, 20, 30, 80, 120, 200])
def test_thin_plate_space_and_penalty_match_public_r_observations(number):
    path = Path(__file__).parent / "golden/gam" / f"gam_{number:04d}.json"
    case = json.loads(path.read_text())
    design = build_design(case["spec"]["formula"], pd.DataFrame(case["spec"]["data"]))
    actual = design.smooths[0]
    reference = case["oracle"]["result"]["smooths"][0]
    indices = np.arange(reference["first_para"] - 1, reference["last_para"])
    reference_x = np.asarray(case["oracle"]["result"]["X"])[:, indices]
    change = np.linalg.lstsq(reference_x, design.X[:, actual.indices], rcond=None)[0]
    np.testing.assert_allclose(reference_x @ change, design.X[:, actual.indices],
                               rtol=1e-9, atol=1e-10)
    transformed = change.T @ np.asarray(reference["S"][0]) @ change
    own = design.S[actual.penalty_indices[0]][np.ix_(actual.indices, actual.indices)]
    # Smoothing parameters absorb each basis's documented penalty normalization.
    normalization = np.sum(transformed * own) / np.sum(own * own)
    np.testing.assert_allclose(transformed, normalization * own, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("number", [0, 10, 20, 28, 30, 58, 60, 120, 200, 489, 1160])
def test_thin_plate_native_coordinates_match_public_r_observations(number):
    # Fractional-rank smooth statistics depend on the representative selected
    # by the eigenvector and constraint QR conventions, even for the same fit.
    path = Path(__file__).parent / "golden/gam" / f"gam_{number:04d}.json"
    case = json.loads(path.read_text())
    design = build_design(case["spec"]["formula"], pd.DataFrame(case["spec"]["data"]))
    np.testing.assert_allclose(design.X, case["oracle"]["result"]["X"],
                               rtol=1e-9, atol=2e-10)


def test_multivariate_thin_plate_function_space_is_rotation_invariant():
    random = np.random.default_rng(514)
    training = random.uniform(-1, 1, size=(45, 2))
    prediction = random.uniform(-1.5, 1.5, size=(17, 2))
    angle = 0.53
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    original = spline_basis(training, "tp", 10)
    rotated = spline_basis(training @ rotation, "tp", 10)
    change = np.linalg.lstsq(original.matrix(training),
                             rotated.matrix(training @ rotation), rcond=None)[0]
    np.testing.assert_allclose(original.matrix(prediction) @ change,
                               rotated.matrix(prediction @ rotation), atol=2e-12)
    np.testing.assert_allclose(change.T @ original.S[0] @ change, rotated.S[0],
                               atol=2e-12)
