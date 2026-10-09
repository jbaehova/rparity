"""Public Stage 2 examples with data fetched through the development oracle.

The mcycle example adapts the mean smooth in mgcv's public gaulss example to
the supported Gaussian family with a cubic regression basis of dimension 10:
https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/gaulss.html

The Salamanders formula is the zero-inflated Poisson example in:
https://glmmtmb.github.io/glmmTMB/reference/glmmTMB.html

Package data and numerical observations remain in ignored development cache
files. Runtime fitting never invokes R.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from rparity import gam, glmmTMB

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "development" / "oracle" / "cache"
pytestmark = [
    pytest.mark.needs_r,
    pytest.mark.skipif(shutil.which("Rscript") is None,
                       reason="R is a development-only oracle"),
]


def _oracle(label: str, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    CACHE.mkdir(parents=True, exist_ok=True)
    source = CACHE / f"s2-public-r-examples-{label}-input.json"
    destination = CACHE / f"s2-public-r-examples-{label}-output.json"
    source.write_text(json.dumps({"cases": cases}))
    completed = subprocess.run(
        ["Rscript", "development/oracle/run_case.R", str(source), str(destination)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if completed.returncode and "there is no package called" in completed.stderr:
        pytest.skip(completed.stderr.strip())
    assert completed.returncode == 0, completed.stderr
    outputs = json.loads(destination.read_text())
    for output in outputs:
        error = output["result"].get("error", "")
        if "there is no package called" in error or "package or namespace load failed" in error:
            pytest.skip(error)
        assert not error, error
        assert not output["warnings"], output["warnings"]
    return [output["result"] for output in outputs]


def _table(result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        {name: np.atleast_1d(values) for name, values in result["values"].items()},
        index=np.atleast_1d(result["rows"]), dtype=float,
    )


def _coefficient_table(actual: pd.DataFrame, reference: dict[str, Any]) -> None:
    expected = _table(reference)
    assert list(actual.index) == list(expected.index)
    assert list(actual.columns) == list(expected.columns)
    for column in actual:
        is_probability = column.startswith("Pr") or column == "p-value"
        tolerance = 1e-5 if column == "Estimate" else 1e-4
        np.testing.assert_allclose(
            actual[column], expected[column], rtol=0 if is_probability else tolerance,
            atol=1e-4 if is_probability else 1e-12,
        )


@pytest.fixture(scope="module")
def public_examples() -> dict[str, Any]:
    observations = _oracle("datasets", [
        {"id": "s2-public-mcycle-data", "operation": "data",
         "dataset": "mcycle", "package": "MASS"},
        {"id": "s2-public-Salamanders-data", "operation": "data",
         "dataset": "Salamanders", "package": "glmmTMB"},
    ])
    motorcycle = pd.DataFrame(observations[0]["data"])
    salamanders = pd.DataFrame(observations[1]["data"])
    # The public example uses yes as the treatment reference for mined.
    salamanders["mined"] = pd.Categorical(salamanders["mined"], categories=["yes", "no"])
    new_motorcycle = pd.DataFrame({"times": np.linspace(
        motorcycle["times"].min(), motorcycle["times"].max(), 15,
    )})
    gam_formula = "accel ~ s(times,bs='cr',k=10)"
    tmb_formula = "count ~ mined + (1|site)"
    specifications = [
        {"id": "s2-public-mcycle-gam", "call": "gam", "formula": gam_formula,
         "dataset": "mcycle", "package": "MASS", "args": {"method": "REML"},
         "gam_prediction": True, "newdata": new_motorcycle.to_dict(orient="list")},
        {"id": "s2-public-Salamanders-glmmTMB", "call": "glmmTMB",
         "formula": tmb_formula, "dataset": "Salamanders", "package": "glmmTMB",
         "family": "poisson", "ziformula": "~ mined", "score_polish": True},
    ]
    references = _oracle("fits", specifications)
    return {
        "motorcycle": motorcycle, "new_motorcycle": new_motorcycle,
        "salamanders": salamanders, "gam_formula": gam_formula,
        "tmb_formula": tmb_formula, "gam_reference": references[0],
        "tmb_reference": references[1],
    }


def test_public_mcycle_gaussian_gam(public_examples: dict[str, Any]) -> None:
    example = public_examples
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = gam(example["gam_formula"], example["motorcycle"], method="REML")
    assert not [warning for warning in caught if issubclass(warning.category, RuntimeWarning)]
    reference = example["gam_reference"]
    assert model.converged and reference["converged"]
    # Coefficients depend on the spline coordinates. Compare in the public
    # R design basis after confirming both complete function spaces agree.
    reference_design = np.asarray(reference["X"], dtype=float)
    transform = np.zeros((len(model.beta), len(model.beta)))
    transform[0, 0] = 1
    transform[1:, 1:] = np.linalg.lstsq(reference_design[:, 1:], model.X[:, 1:],
                                      rcond=1e-12)[0]
    assert np.linalg.matrix_rank(transform) == len(model.beta)
    np.testing.assert_allclose(reference_design @ transform, model.X,
                               rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(transform @ model.beta, reference["beta"],
                               rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(transform @ model.cov_beta @ transform.T, reference["vcov"],
                               rtol=1e-4, atol=1e-12)
    assert abs(model.objective - reference["criterion"]) <= 1e-6
    np.testing.assert_allclose(model.fitted(), reference["fitted"], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(model.smooth_edf, _table(reference["smooth_table"])["edf"],
                               rtol=1e-3, atol=1e-12)
    summary = model.summary_tables()
    _coefficient_table(summary["parametric"], reference["coefficients"])
    _coefficient_table(summary["smooth"], reference["smooth_table"])
    prediction = model.predict(example["new_motorcycle"], type="link", se_fit=True)
    np.testing.assert_allclose(prediction["fit"], reference["prediction"]["prediction"],
                               rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(prediction["se.fit"], reference["prediction"]["se_fit"],
                               rtol=1e-4, atol=1e-12)
    effect = model.partial_effects("s(times)", newdata=example["new_motorcycle"])
    np.testing.assert_allclose(effect["fit"], np.asarray(reference["terms"]["prediction"]).ravel(),
                               rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(effect["se"], np.asarray(reference["terms"]["se_fit"]).ravel(),
                               rtol=1e-4, atol=1e-12)
    sections = ["Family:", "Parametric coefficients:",
                "Approximate significance of smooth terms:"]
    for text in [model.summary(), reference["summary"]]:
        positions = [text.index(section) for section in sections]
        assert positions == sorted(positions)


def test_public_salamanders_zero_inflated_poisson(public_examples: dict[str, Any]) -> None:
    example = public_examples
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = glmmTMB(example["tmb_formula"], example["salamanders"],
                        family="poisson", ziformula="~ mined")
    assert not [warning for warning in caught if issubclass(warning.category, RuntimeWarning)]
    reference = example["tmb_reference"]
    assert reference["optimizer_diagnostics"]["convergence"] == 0
    assert reference["optimizer_diagnostics"]["pdHess"]
    assert abs(model.logLik() - reference["logLik"]) <= 1e-6
    np.testing.assert_allclose([model.AIC(), model.BIC()], [reference["AIC"], reference["BIC"]],
                               rtol=1e-6, atol=1e-12)
    for component in ["cond", "zi"]:
        coefficients = model.fixef()[component]
        assert list(coefficients.index) == reference["fixef_names"][component]
        np.testing.assert_allclose(coefficients, reference["fixef"][component],
                                   rtol=1e-5, atol=1e-12)
        np.testing.assert_allclose(model.vcov(component), reference["vcov_components"][component],
                                   rtol=1e-4, atol=1e-12)
        standard_error = np.sqrt(np.diag(model.vcov(component)))
        statistic = coefficients.to_numpy() / standard_error
        table = pd.DataFrame({"Estimate": coefficients, "Std. Error": standard_error,
                              "z value": statistic,
                              "Pr(>|z|)": 2 * stats.norm.sf(np.abs(statistic))})
        _coefficient_table(table, reference["coefficients"][component])
    assert model.fixef()["disp"].empty
    np.testing.assert_allclose(model.VarCorr()["cond"]["site"],
                               reference["VarCorr"]["cond"]["site"]["covariance"],
                               rtol=1e-4, atol=1e-12)
    modes = model.ranef()["cond"]["site"]
    expected_modes = reference["ranef"]["cond"]["site"]
    np.testing.assert_allclose(modes.loc[expected_modes["levels"]], expected_modes["values"],
                               rtol=1e-5, atol=1e-12)
    for prediction_type in ["response", "cond", "zprob", "disp"]:
        np.testing.assert_allclose(model.predict(type=prediction_type),
                                   reference["predictions"][prediction_type],
                                   rtol=1e-6, atol=1e-12)
    sections = ["Random effects:", "Conditional model:", "Zero-inflation model:"]
    for text in [model.summary(), reference["summary"]]:
        positions = [text.index(section) for section in sections]
        assert positions == sorted(positions)
