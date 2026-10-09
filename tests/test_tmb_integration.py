"""Public downstream inference and new-row predictions for distributional GLMMs."""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rparity import Anova, emmeans, glmmTMB

CASES = json.loads((Path(__file__).parent / "data/tmb_integration.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["case"])
def test_tmb_downstream_and_newdata(case):
    spec = case["spec"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = glmmTMB(
            spec["formula"], pd.DataFrame(spec["data"]), family=spec["family"],
            link=spec["link"], ziformula=spec["ziformula"], dispformula=spec["dispformula"],
            weights=spec.get("weights"), offset=spec.get("offset"),
        )
    fit_reference = case["fit_observation"]["result"]
    diagnostics = fit_reference["optimizer_diagnostics"]
    if diagnostics["convergence"] != 0 or not diagnostics["pdHess"]:
        assert model.logLik() >= -diagnostics["objective"] - 1e-6
        assert any(issubclass(warning.category, RuntimeWarning) for warning in caught)
        return
    for operation in case["operations"]:
        request, reference = operation["request"], operation["oracle"]["result"]
        assert "error" not in reference
        kind = request["operation"]
        if kind == "predict":
            actual = model.predict(pd.DataFrame(spec["newdata"]), type=request["prediction_type"])
            np.testing.assert_allclose(actual, reference["prediction"], rtol=1e-6, atol=1e-12)
        elif kind == "Anova":
            actual = Anova(model, type=request["type"], component=request["component"])
            reference_frame = pd.DataFrame(
                {name: np.atleast_1d(value) for name, value in reference["values"].items()},
                index=np.atleast_1d(reference["rows"]),
            )
            assert list(actual.index) == list(reference_frame.index)
            assert list(actual.columns) == list(reference_frame.columns)
            for name in actual.columns:
                np.testing.assert_allclose(actual[name], reference_frame[name],
                                           rtol=0 if name.startswith("Pr") else 1e-4,
                                           atol=1e-4 if name.startswith("Pr") else 1e-12)
        else:
            grid = emmeans(model, **request["emm_args"])
            actual = grid.summary(type=request["response_type"], infer=(True, True))
            reference_frame = pd.DataFrame(
                {name: np.atleast_1d(values) for name, values in reference["values"].items()}
            )
            estimate = next(name for name in reference_frame if name in {
                "emmean", "response", "rate", "prob"
            })
            actual_estimate = next(name for name in actual if name in {
                "emmean", "response", "rate", "prob"
            })
            np.testing.assert_allclose(actual[actual_estimate], reference_frame[estimate],
                                       rtol=1e-6, atol=1e-12)
            for name in ["SE", "asymp.LCL", "asymp.UCL", "lower.CL", "upper.CL"]:
                if name in reference_frame:
                    np.testing.assert_allclose(actual[name], reference_frame[name],
                                               rtol=1e-4, atol=1e-12)
            expected_df = np.asarray([np.inf if value is None else value
                                      for value in reference_frame["df"]])
            np.testing.assert_allclose(actual["df"], expected_df, rtol=1e-3, atol=1e-12)
            np.testing.assert_allclose(actual["p.value"], reference_frame["p.value"],
                                       rtol=0, atol=1e-4)
