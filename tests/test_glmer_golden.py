"""Strict synthetic R parity fixtures for GLMM Laplace models."""

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rparity.lmm.glmer import glmer

FIXTURES = sorted((Path(__file__).parent / "golden/glmer").glob("*.json"))


@pytest.mark.golden
@pytest.mark.parametrize("path", FIXTURES, ids=lambda path: path.stem)
def test_glmer_golden(path, record_property):
    fixture = json.loads(path.read_text())
    spec = fixture["spec"]
    oracle = fixture["oracle"]
    expected = oracle["result"]
    assert "error" not in expected, expected.get("error")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = glmer(
            spec["formula"],
            pd.DataFrame(spec["data"]),
            family=spec["family"],
            link=spec["link"],
            weights=spec.get("weights"),
            offset=spec.get("offset"),
        )
    _assert_summary_structure(result.summary(), expected["summary"])
    assert result.logLik() >= expected["logLik"] - 1e-6
    record_property('better_optimum', result.logLik() > expected['logLik'] + 1e-6)
    oracle_warnings = oracle["warnings"]
    messages = (
        oracle_warnings if isinstance(oracle_warnings, str) else " ".join(oracle_warnings)
    ).lower()
    if "non-integer" in messages:
        assert any("non-integer" in str(w.message).lower() for w in caught)
    if expected.get("singular") or "singular" in messages:
        assert any("singular" in str(w.message).lower() for w in caught)
        return
    if "converg" in messages or "hessian" in messages or "gradient" in messages:
        assert any(issubclass(w.category, RuntimeWarning) for w in caught)
        return
    np.testing.assert_allclose(result.beta, expected["beta"], rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(result.cov_beta, expected["vcov"], rtol=1e-4, atol=1e-12)
    np.testing.assert_allclose(result.fitted(), expected["fitted"], rtol=1e-6, atol=1e-12)
    # R may reorder blocks, so compare all covariance entries by group and terms.
    table = expected["VarCorr"]["values"]
    expected_variance = {
        (re.sub(r"\.\d+$", "", str(group)), tuple(sorted((str(first), str(first if second is None else second))))): value
        for group, first, second, value in zip(
            np.atleast_1d(table["grp"]),
            np.atleast_1d(table["var1"]),
            np.atleast_1d(table["var2"]),
            np.atleast_1d(table["vcov"]),
        )
    }
    for block, covariance in zip(result.random_blocks, result.random_covariances):
        for i, first in enumerate(block.names):
            for j, second in enumerate(block.names[i:], start=i):
                key = (block.term.group, tuple(sorted((first, second))))
                np.testing.assert_allclose(covariance[i, j], expected_variance[key],
                                           rtol=1e-4, atol=1e-12)


@pytest.mark.parametrize("case_id", ["glmer_0003", "glmer_0004", "glmer_0013"])
def test_glmer_summary_structure(case_id):
    """Compare the stable R section labels, excluding call provenance and numbers."""
    path = Path(__file__).parent / "golden/glmer" / f"{case_id}.json"
    if not path.exists():
        pytest.skip("R fixture has not been generated")
    fixture = json.loads(path.read_text())
    spec = fixture["spec"]
    model = glmer(
        spec["formula"],
        pd.DataFrame(spec["data"]),
        family=spec["family"],
        link=spec["link"],
        weights=spec.get("weights"),
    )
    actual = model.summary()
    expected = fixture["oracle"]["result"]["summary"]
    _assert_summary_structure(actual, expected)


def _assert_summary_structure(actual: str, expected: str) -> None:
    labels = [
        "Scaled residuals:",
        "Random effects:",
        "Fixed effects:",
        "Signif. codes:",
        "Correlation of Fixed Effects:",
    ]
    for text in [actual, expected]:
        positions = [text.index(label) for label in labels]
        assert positions == sorted(positions)
    for marker in ["AIC", "Groups Name", "Estimate Std.", "Min"]:

        def header(text, marker=marker):
            return next(
                line.split() for line in text.splitlines() if marker in " ".join(line.split())
            )

        assert header(actual) == header(expected)
    assert "[glmerMod]" in actual and "[glmerMod]" in expected
    actual_family = next(line for line in actual.splitlines() if "Family:" in line)
    expected_family = next(line for line in expected.splitlines() if "Family:" in line)
    assert actual_family.split() == expected_family.split()
