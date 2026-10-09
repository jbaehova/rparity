"""Published GAM inference laws and independent public R observations."""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from rparity.gam import gam
from rparity.gam._summary import (
    _fractional_sf,
    _weighted_chisq_sf,
    smooth_table,
    summary_statistics,
)

GOLDEN = Path(__file__).parent / "golden/gam"


def _observation(case):
    fixture = json.loads((GOLDEN / f"gam_{case:04d}.json").read_text())
    spec, reference = fixture["spec"], fixture["oracle"]["result"]
    data = pd.DataFrame(spec["data"])
    weights = spec.get("weights", np.ones(len(data)))
    weights = np.asarray(data[weights] if isinstance(weights, str) else weights, dtype=float)
    paired = re.match(r"\s*cbind\((\w+),\s*(\w+)\)", spec["formula"])
    if paired:
        weights *= np.asarray(data[paired.group(1)] + data[paired.group(2)])
        response = np.asarray(data[paired.group(1)]) / np.asarray(data[paired.group(1)] + data[paired.group(2)])
    else:
        response = np.asarray(data[spec["formula"].split("~")[0].strip()])
    offset = spec.get("offset", np.zeros(len(data)))
    offset = np.asarray(data[offset] if isinstance(offset, str) else offset, dtype=float)
    for name in re.findall(r"offset\((\w+)\)", spec["formula"]):
        offset = offset + np.asarray(data[name])
    X = np.asarray(reference["X"])
    penalty = np.zeros((X.shape[1], X.shape[1]))
    smoothing = np.atleast_1d(reference["sp"])
    smooths = []
    for smooth in reference["smooths"]:
        indices = np.arange(smooth["first_para"] - 1, smooth["last_para"])
        for position, component in enumerate(smooth["S"]):
            penalty[np.ix_(indices, indices)] += (
                smoothing[smooth["first_sp"] - 1 + position] * np.asarray(component)
            )
        smooths.append(SimpleNamespace(label=smooth["label"], indices=indices,
                                       null_space_dim=smooth["null_space_dim"]))
    model = SimpleNamespace(
        X=X, beta=np.asarray(reference["beta"]), cov_beta=np.asarray(reference["vcov"]),
        scale=reference["scale"], family=spec["family"], link=spec["link"], weights=weights,
        edf=np.asarray(reference["edf"]), edf1=np.asarray(reference["edf1"]),
        linear_predictor=np.asarray(reference["linear_predictor"]),
        fitted_values=np.asarray(reference["fitted"]), _penalty=penalty,
        _scale_known=spec["family"] in {"poisson", "binomial"},
        df_resid=len(data) - np.sum(reference["edf"]),
        design=SimpleNamespace(smooths=smooths),
        y=response, offset=offset, nobs=len(data), method=spec["args"]["method"],
        objective=reference["criterion"], deviance=lambda: reference["deviance"],
    )
    table = reference["smooth_table"]
    expected = pd.DataFrame({name: np.atleast_1d(value) for name, value in table["values"].items()},
                            index=np.atleast_1d(table["rows"]))
    return model, expected


@pytest.mark.parametrize("case", [0, 1, 2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 16, 18,
                                    20, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 35,
                                    36, 37, 38, 39, 379, 1099, 1109, 1119, 1129,
                                    1149, 1159, 1169, 1179])
def test_smooth_inference_from_public_observed_fit(case):
    model, expected = _observation(case)
    actual = smooth_table(model)
    np.testing.assert_allclose(actual[["edf", "Ref.df"]], expected[["edf", "Ref.df"]],
                               rtol=1e-10, atol=1e-10)
    statistic = "Chi.sq" if model._scale_known else "F"
    np.testing.assert_allclose(actual[statistic], expected[statistic], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual["p-value"], expected["p-value"], rtol=0, atol=1e-4)


@pytest.mark.parametrize("residual_df", [None, 47.5])
@pytest.mark.parametrize("rank", [1, 2, 5, 10])
def test_integer_rank_is_exact_chisquare_or_f(rank, residual_df):
    for statistic in [0.2, 3.0, 11.0]:
        expected = (stats.chi2.sf(statistic, rank) if residual_df is None
                    else stats.f.sf(statistic / rank, rank, residual_df))
        assert _fractional_sf(statistic, rank, residual_df) == pytest.approx(expected, abs=1e-13)


def test_fractional_rank_tail_has_correct_null_calibration():
    # Independent simulation checks the null law, including its non-unit
    # directions, rather than comparing two copies of the quadrature formula.
    rng = np.random.default_rng(3709)
    rank = 3.35
    fraction = rank - int(rank)
    B = np.array([[1, np.sqrt(fraction * (1 - fraction) / 2)],
                  [np.sqrt(fraction * (1 - fraction) / 2), fraction]])
    draws = rng.normal(size=(250000, 4))
    statistic = np.sum(draws[:, :2]**2, axis=1) + np.einsum("ij,jk,ik->i", draws[:, 2:], B, draws[:, 2:])
    assert np.mean(statistic) == pytest.approx(rank, abs=0.018)
    assert np.var(statistic) == pytest.approx(2 * rank, abs=0.06)
    for threshold in [1.5, 4.0, 8.0]:
        assert _fractional_sf(threshold, rank, None) == pytest.approx(
            np.mean(statistic > threshold), abs=0.004,
        )


@pytest.mark.parametrize("residual_df", [None, 63.5])
def test_general_quadratic_form_matches_equal_weight_law(residual_df):
    weights = np.full(7, 0.37)
    threshold = 4.5
    expected = (stats.chi2.sf(threshold / 0.37, 7) if residual_df is None
                else stats.f.sf(threshold / (0.37 * 7), 7, residual_df))
    assert _weighted_chisq_sf(threshold, weights, residual_df) == pytest.approx(expected, abs=1e-13)


def test_general_quadratic_form_against_independent_simulation():
    rng = np.random.default_rng(7801)
    weights = np.array([0.015, 0.1, 0.32, 0.7, 0.91])
    draws = np.sum(rng.normal(size=(250000, len(weights)))**2 * weights, axis=1)
    scale = rng.chisquare(80, size=len(draws)) / 80
    for threshold in [0.5, 2.0, 4.5]:
        assert _weighted_chisq_sf(threshold, weights) == pytest.approx(np.mean(draws > threshold), abs=0.004)
        assert _weighted_chisq_sf(threshold, weights, 80) == pytest.approx(
            np.mean(draws / scale > threshold), abs=0.004,
        )


@pytest.mark.parametrize("case", [0, 8, 14, 20, 33])
def test_native_fit_smooth_test_parity(case):
    fixture = json.loads((GOLDEN / f"gam_{case:04d}.json").read_text())
    spec = fixture["spec"]
    model = gam(spec["formula"], spec["data"], family=spec["family"], link=spec["link"],
                method=spec["args"]["method"], weights=spec.get("weights"), offset=spec.get("offset"))
    _, expected = _observation(case)
    actual = smooth_table(model)
    statistic = "Chi.sq" if model._scale_known else "F"
    np.testing.assert_allclose(actual[statistic], expected[statistic], rtol=1e-4)
    np.testing.assert_allclose(actual["p-value"], expected["p-value"], atol=1e-4, rtol=0)


# Additional public numeric observations obtained by development/oracle/run_case.R. The
# fitted corpus inputs remain the independent source for response and basis.
@pytest.mark.parametrize("case,r_squared,deviance_explained,rank", [
    (0, 0.4331682974167592, 0.4592805690792711, 10),
    (3, 0.12119843756038384, 0.14132366431819818, 7),
    (4, 0.4941545163827732, 0.5343754934250075, 14),
    (10, 0.06358093149446697, 0.08112367190483172, 10),
    (37, 0.14267500632455388, 0.19273271722185767, 10),
    (49, 0.33263704126529636, 0.3671877824939762, 11),
    (194, 0.4158869921881262, 0.49135438767582124, 13),
    (379, 0.017343232634171035, 0.03575518848992768, 17),
    (716, 0.605263517256275, 0.680697867813502, 26),
    (718, 0.4918077592907739, 0.6542788513998988, 21),
    (1109, 0.034579402509006485, 0.047023908736504146, 15),
])
def test_summary_statistics_include_weights_offsets_and_penalized_rank(
    case, r_squared, deviance_explained, rank,
):
    model, _ = _observation(case)
    actual = summary_statistics(model)
    assert actual["r.sq"] == pytest.approx(r_squared, rel=1e-4)
    assert actual["dev.expl"] == pytest.approx(deviance_explained, rel=1e-4)
    assert actual["rank"] == rank
    assert actual["n"] == model.nobs
