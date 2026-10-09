"""Committed R black-box parity checks. R is not required to run these tests."""

from __future__ import annotations

import importlib
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rparity.gls import gls

gls_module = importlib.import_module("rparity.gls")

FILES = sorted((Path(__file__).parent / "golden/gls").glob("*.json"))


def fit_spec(spec: dict):
    """Fit the exact formula and covariance specifications sent to R."""
    arguments = dict(spec["args"])
    for key, dest in [("correlation", "correlation"), ("variance", "weights")]:
        if key in spec:
            structure = spec[key]
            arguments[dest] = getattr(gls_module, structure["name"])(**structure["args"])
    data = pd.DataFrame(spec["data"])
    for column in spec.get("factors", []):
        data[column] = pd.Categorical(data[column], categories=sorted(data[column].unique()))
    return gls(spec["formula"], data, **arguments)


def summary_structure(text: str) -> list[str]:
    """Mask numbers, whitespace and call provenance, retaining every displayed label."""
    signature = []
    number = re.compile(r"(?<![\w])[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Data:"):
            stripped = "Data: data"
        stripped = stripped.replace("~ ", "~")
        signature.append(" ".join(number.sub("<number>", stripped).split()))
    return signature


@pytest.mark.golden
@pytest.mark.parametrize("index", range(20))
def test_gls_summary_structure_matches_r(index):
    # Both methods and every covariance family are represented by actual R summaries.
    fixture = json.loads((Path(__file__).parent / f"golden/gls/gls_{index:04d}.json").read_text())
    model = fit_spec(fixture["spec"])
    assert summary_structure(model.summary()) == summary_structure(
        fixture["oracle"]["result"]["summary"]
    )


@pytest.mark.golden
@pytest.mark.parametrize("path", FILES, ids=lambda path: path.stem)
def test_gls_golden(path, request):
    fixture = json.loads(path.read_text())
    oracle = fixture["oracle"]
    result = oracle["result"]
    assert "error" not in result, result.get("error")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = fit_spec(fixture["spec"])
    # A superior objective is allowed by TASK.md; no tolerance is loosened.
    assert model.logLik() >= result["logLik"] - 1e-6
    request.node.user_properties.append(("better_optimum", model.logLik() > result["logLik"] + 1e-6))
    if oracle["warnings"]:
        assert caught, "R warned, but Python did not emit an optimization warning"
        return
    np.testing.assert_allclose(model.beta, np.atleast_1d(result["beta"]), rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(model.cov_beta, result["vcov"], rtol=1e-4, atol=1e-12)
    np.testing.assert_allclose(model.sigma, result["sigma"], rtol=1e-4, atol=1e-12)
    np.testing.assert_allclose(model.fitted(), result["fitted"], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(
        model.coefficient_table()["p-value"],
        result["coefficients"]["values"]["p-value"],
        atol=1e-4,
        rtol=0,
    )
    if "correlation" in result:
        np.testing.assert_allclose(
            model.correlation_parameters,
            np.atleast_1d(result["correlation"]),
            rtol=1e-4,
            atol=1e-12,
        )
    if "variance" in result:
        values = result["variance"]
        if isinstance(values, dict):
            values = [values[name] for name in model.variance_parameters.index]
        np.testing.assert_allclose(
            model.variance_parameters, np.atleast_1d(values), rtol=1e-4, atol=1e-12
        )
