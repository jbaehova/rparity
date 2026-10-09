"""Independent numerical diagnostics, including the exact seeded R null."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rparity.gam._check import _RRandom, k_check
from rparity.gam._formula import build_design


def _case(number: int) -> dict:
    return json.loads((Path(__file__).parent / "golden" / "gam" /
                       f"gam_{number:04}.json").read_text())


def test_seed_stream_matches_public_r_rng() -> None:
    # Numeric observations from the public RNG oracle, independent of mgcv.
    observed = [0.1550508332438767, 0.9683788088150322, 0.46826308593153954,
                0.7768196517135948, 0.4078857412096113, 0.5387971485033631,
                0.2068876635748893, 0.18710355530492961, 0.779969688039273,
                0.193943927064538]
    np.testing.assert_array_equal(_RRandom(17).uniform(10), observed)
    np.testing.assert_array_equal(_RRandom(17).sample(10) + 1,
                                  [2, 8, 1, 6, 4, 5, 3, 10, 7, 9])


@pytest.mark.parametrize(("seed", "population", "expected"), [
    (0, 65536, [17401, 37543, 13218, 61911, 41230, 13499, 45025, 50453, 47030, 24906]),
    (123456, 65537, [6326, 10982, 38915, 57728, 58718, 12951, 50935, 5286, 11266, 45909]),
    (-1, 16, [7, 14, 8, 9, 4, 16, 1, 10, 13, 5]),
])
def test_integer_sampling_across_sixteen_bit_boundary(
    seed: int, population: int, expected: list[int],
) -> None:
    np.testing.assert_array_equal(_RRandom(seed).sample(population, 10) + 1, expected)


@pytest.mark.parametrize("number", [0, 1, 2, 3, 7, 8, 9, 10, 18])
def test_univariate_index_and_seeded_pvalue_from_r_residuals(number: int) -> None:
    # Separating checks from fitting makes a residual/optimizer discrepancy
    # distinguishable from an error in the diagnostic algorithm itself.
    case = _case(number)
    reference = case["oracle"]["result"]
    frame = pd.DataFrame(case["spec"]["data"])
    smooths = [SimpleNamespace(label=smooth["label"],
                              variables=np.atleast_1d(smooth["term"]).tolist(),
                              indices=np.arange(smooth["first_para"] - 1, smooth["last_para"]))
               for smooth in reference["smooths"]]
    table = k_check(SimpleNamespace(smooths=smooths), frame,
                    np.asarray(reference["residuals"]), np.asarray(reference["edf"]),
                    seed=1, n_rep=400)
    expected = reference["gam_check"]["k_check"]
    assert table.index.tolist() == np.atleast_1d(expected["rows"]).tolist()
    for column in table:
        np.testing.assert_allclose(table[column], np.atleast_1d(expected["values"][column]),
                                   rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("number", [5, 6, 15, 16, 25, 26, 35, 36])
def test_multivariate_geometry_from_r_fitted_functions(number: int) -> None:
    case = _case(number)
    reference = case["oracle"]["result"]
    frame = pd.DataFrame(case["spec"]["data"])
    for variable in case["spec"]["factors"]:
        frame[variable] = pd.Categorical(frame[variable])
    design = build_design(case["spec"]["formula"], frame)
    # Internal spline coordinates may differ, but the fitted smooth function
    # in the shared spline space uniquely determines the neighbor geometry.
    beta = np.linalg.lstsq(design.X, reference["linear_predictor"], rcond=1e-12)[0]
    np.testing.assert_allclose(design.X @ beta, reference["linear_predictor"],
                               rtol=1e-8, atol=1e-8)
    table = k_check(design, frame, np.asarray(reference["residuals"]),
                    np.zeros(len(beta)), coefficients=beta, seed=1, n_rep=400)
    expected = reference["gam_check"]["k_check"]["values"]
    for column in ["k'", "k-index", "p-value"]:
        np.testing.assert_allclose(table[column], np.atleast_1d(expected[column]),
                                   rtol=1e-13, atol=1e-13)


def test_factor_smooth_has_no_numeric_neighbor_test() -> None:
    smooth = SimpleNamespace(label="s(g)", variables=["g"], indices=np.array([0, 1]))
    frame = pd.DataFrame({"g": pd.Categorical(["a", "a", "b", "b"])})
    result = k_check(SimpleNamespace(smooths=[smooth]), frame,
                     np.array([1, -1, 2, -2]), np.array([0.7, 0.8]))
    assert result.loc["s(g)", "k'"] == 2
    assert result.loc["s(g)", "edf"] == 1.5
    assert result.loc["s(g)", ["k-index", "p-value"]].isna().all()


def test_checks_preserve_numpy_global_random_state() -> None:
    before = np.random.get_state()
    _RRandom(1).sample(100)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize(("number", "indices", "pvalues"), [
    (0, [0.9056273470943449], [0.14]),
    (5, [1.1158950297840629], [0.8]),
    (6, [1.1023686784632298, 0.8712267138536792, 1.0163199395423552], [0.68, 0.14, 0.48]),
    (8, [1.1279901526065483] * 3, [0.82, 0.92, 0.88]),
])
def test_subsample_and_multiple_smooth_rng_consumption(
    number: int, indices: list[float], pvalues: list[float],
) -> None:
    # Public R oracle, k.check(subsample=50, n.rep=50), set.seed(17).
    # These fixed diagnostic observations use the retained original fit.
    # Later higher-precision fitting references are tested by the golden suite.
    case = _case(number)
    original = case.get("oracle_attempts", [{"oracle": case["oracle"]}])[0]
    reference = original["oracle"]["result"]
    frame = pd.DataFrame(case["spec"]["data"])
    for variable in case["spec"]["factors"]:
        frame[variable] = pd.Categorical(frame[variable])
    design = build_design(case["spec"]["formula"], frame)
    beta = np.linalg.lstsq(design.X, reference["linear_predictor"], rcond=1e-12)[0]
    table = k_check(design, frame, np.asarray(reference["residuals"]),
                    np.zeros(len(beta)), coefficients=beta, seed=17, n_rep=50, subsample=50)
    np.testing.assert_allclose(table["k-index"], indices, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(table["p-value"], pvalues)


@pytest.mark.parametrize("options", [{"n_rep": 0}, {"n_rep": 1.5}, {"subsample": 1}])
def test_invalid_check_controls_are_rejected(options: dict) -> None:
    with pytest.raises(ValueError):
        k_check(SimpleNamespace(smooths=[]), pd.DataFrame({"x": [0, 1]}),
                np.array([1, -1]), np.array([]), **options)
