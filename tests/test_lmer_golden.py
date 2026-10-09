"""Seeded public-R lmer comparisons at the TASK.md numerical thresholds."""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rparity.lmm.lmer import ConvergenceWarning, SingularFitWarning, lmer

GOLDENS = sorted((Path(__file__).parent / "golden/lmer").glob("*.json"))


def fit_spec(spec):
    data = pd.DataFrame(spec["data"])
    for name in spec.get("factors", []):
        data[name] = pd.Categorical(data[name])
    return lmer(spec["formula"], data, reml=spec.get("args", {}).get("REML", True),
                weights=spec.get("weights"), offset=spec.get("offset"),
                contrasts=spec.get("contrasts", "treatment"))


def close(actual, expected, rtol):
    # Only a floating-point roundoff floor accompanies the prescribed relative tolerance.
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=rtol, atol=1e-12)


def summary_structure(text):
    """Keep section order, labels, row names and headings, excluding provenance.

    Data/control expressions identify the caller's object and optimizer; their
    labels are structural, while those arbitrary expressions are not.
    """
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("Formula:"))
    header = " ".join(" ".join(lines[:start]).split()).replace("likelihood .", "likelihood.").replace("[ ", "[")
    signature = [header]
    for label in ("Formula:", "Data:", "Weights:", "Control:"):
        if any(line.strip().startswith(label) for line in lines):
            signature.append(label)
    known = {"Scaled residuals:", "Random effects:", "Fixed effects:",
             "Correlation of Fixed Effects:"}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("REML criterion at convergence:"):
            signature.append("REML criterion at convergence:")
        elif stripped.startswith("AIC") and "BIC" in stripped:
            signature.append(" ".join(stripped.split()))
        elif stripped in known:
            signature.append(stripped)
            signature.append(" ".join(lines[i+1].split()))
        elif stripped.startswith("Number of obs:"):
            signature.append(re.sub(r"\d+", "#", " ".join(stripped.split())))
        elif stripped.startswith("Signif. codes:"):
            signature.append(re.sub(r"[0-9]+(?:\.[0-9]+)?", "#", stripped))
    return signature


@pytest.mark.golden
@pytest.mark.parametrize("path", GOLDENS, ids=lambda p: p.stem)
def test_lmer_golden(path, record_property):
    case = json.loads(path.read_text())
    spec, oracle = case["spec"], case["oracle"]
    expected = oracle["result"]
    assert "error" not in expected
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        model = fit_spec(spec)
    criterion = -2*float(expected["logLik"])
    assert model.deviance() <= criterion + 1e-6
    record_property("better_optimum", model.deviance() < criterion - 1e-6)
    assert summary_structure(model.summary()) == summary_structure(expected["summary"])
    oracle_warnings = oracle["warnings"]
    if isinstance(oracle_warnings, str):
        oracle_warnings = [oracle_warnings]
    singular = any("singular" in warning.lower() for warning in oracle_warnings)
    convergence = any("converg" in warning.lower() or "Hessian" in warning
                      for warning in oracle_warnings)
    if singular:
        assert any(issubclass(w.category, SingularFitWarning) for w in recorded)
    if convergence:
        assert any(issubclass(w.category, ConvergenceWarning) for w in recorded)
    if singular or convergence:
        return
    close(model.fixef(), np.atleast_1d(expected["beta"]), 1e-5)
    assert model.coef_names == expected["coef_names"]
    close(model.vcov(), np.atleast_2d(expected["vcov"]), 1e-4)
    close(model.sigma, expected["sigma"], 1e-4)
    close(model.fitted(), expected["fitted"], 1e-6)
    # Residuals must recover the same observed responses and predictions.
    # TASK.md specifies a prediction threshold, rather than a residual threshold.
    close(model.y-model.residuals(), model.y-np.asarray(expected["residuals"]), 1e-6)
    close(model.AIC(), expected["AIC"], 1e-6)
    close(model.BIC(), expected["BIC"], 1e-6)
    assert model.is_singular() == expected["singular"]
    covariance = model.VarCorr()
    values = expected["VarCorr"]["values"]
    for group, name, other, estimate in zip(values["grp"], values["var1"],
                                           values["var2"], values["vcov"], strict=True):
        if group == "Residual":
            actual = model.sigma**2
        else:
            frame = covariance[group]
            actual = frame.loc[name, name if other is None else other]
        close(actual, estimate, 1e-4)
    actual_random = model.ranef()
    for group, expected_random in expected["ranef"].items():
        frame = actual_random[group]
        levels = np.atleast_1d(expected_random["levels"])
        names = np.atleast_1d(expected_random["names"])
        assert list(frame.index.astype(str)) == list(levels)
        assert list(frame.columns) == list(names)
        close(frame, np.asarray(expected_random["values"]).reshape(frame.shape), 1e-4)
        conditional = expected_random["conditional_variance"]
        if isinstance(conditional, dict):
            # Public R returns separate arrays for independent double-bar blocks.
            for name, block in conditional.items():
                i = list(frame.columns).index(name)
                close(frame.attrs["postVar"][i:i+1, i:i+1], block, 1e-4)
        else:
            close(frame.attrs["postVar"], np.asarray(conditional).reshape(frame.attrs["postVar"].shape), 1e-4)
    for prediction_spec, prediction_oracle in zip(case.get("prediction_specs", []),
                                                 case.get("prediction_oracles", []), strict=True):
        prediction = model.predict(pd.DataFrame(prediction_spec["newdata"]),
                                   re_form="NA" if prediction_spec.get("population") else None,
                                   allow_new_levels=prediction_spec.get("allow_new_levels", False))
        close(prediction, prediction_oracle["result"]["prediction"], 1e-6)
