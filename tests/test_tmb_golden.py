"""Strict normalized-Laplace and component-specific Stage 2 R observations."""

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from scipy.stats import norm

FIXTURES = sorted((Path(__file__).parent / "golden/tmb").glob("*.json"))
POLICIES = json.loads((Path(__file__).parent / "golden/tmb_case_policy.json").read_text())


def _reference_covariance(expected, component, names):
    covariance = expected["vcov_components"].get(component)
    if covariance is None:
        full_names = np.atleast_1d(expected["vcov_full_names"]).tolist()
        indices = [full_names.index(f"{component}~{name}") for name in names]
        covariance = np.asarray(expected["vcov_full"], dtype=float)[np.ix_(indices, indices)]
    return np.atleast_2d(np.asarray(covariance, dtype=float))


def _boundary_limits(model, expected, policy):
    """Reprove an indexed zero limit rather than relax an entire model's tests.

    The mixing probability, density perturbation and observed logit curvature
    must all vanish. A zero random-effect policy requires both the covariance
    and every conditional mode to be within its explicit absolute zero bound.
    Other components and all identifiable predictions retain the TASK limits.
    """
    from rparity.tmb._families import observation_terms

    assert policy["kind"] == "tmb_boundary_identification"
    fields = set(policy["exception_fields"])
    permitted = {"zi_beta", "zi_vcov", "zi_wald_p", "zprob_prediction"}
    for group in policy.get("zero_random_effect", {}):
        permitted.update({f"{group}_random_covariance", f"{group}_random_modes"})
    assert fields <= permitted
    diagnostics = {}
    if policy.get("absent_zi"):
        names = np.atleast_1d(expected["fixef_names"]["zi"]).tolist()
        assert list(model.fixef()["zi"].index) == names
        reference_beta = np.atleast_1d(expected["fixef"]["zi"]).astype(float)
        assert len(reference_beta) == len(model.zi_beta) > 0
        assert {"zi_beta", "zi_vcov", "zi_wald_p"} <= fields
        total_weight = float(np.sum(model.weights))
        assert total_weight > 0
        probability_bound = policy["mixture_likelihood_bound"] / total_weight
        for label, mean, dispersion, zeta, probability in [
            (
                "R", np.asarray(expected["predictions"]["cond"], dtype=float),
                np.asarray(expected["predictions"]["disp"], dtype=float),
                model.zi_X @ reference_beta,
                np.asarray(expected["predictions"]["zprob"], dtype=float),
            ),
            (
                "Python", model.predict(type="cond"), model.predict(type="disp"),
                model.zi_X @ model.zi_beta, model.predict(type="zprob"),
            ),
        ]:
            assert np.all(np.isfinite(probability)) and np.all(probability >= 0)
            assert float(np.max(probability)) <= probability_bound, f"{label} absent ZI"
            np.testing.assert_allclose(expit(zeta), probability, rtol=1e-6, atol=1e-12)
            if model.link == "logit":
                eta = logit(mean)
            elif model.link == "log":
                eta = np.log(mean)
            else:
                assert model.link == "identity"
                eta = mean
            delta = np.log(dispersion) * (2 if model.family == "gaussian" else 1)
            base = observation_terms(
                eta, delta, None, model.y, model.trials,
                np.ones_like(model.weights), model.family,
            ).value
            zero = model.y == 0
            # A beta zero has no density under the uninflated continuous model.
            assert model.family != "beta" or not np.any(zero)
            perturbation = np.logaddexp(0, zeta)
            perturbation[zero] -= np.logaddexp(0, zeta[zero] + base[zero])
            density_difference = float(np.sum(model.weights * np.abs(perturbation)))
            assert np.isfinite(density_difference)
            assert density_difference <= policy["mixture_likelihood_bound"], (
                f"{label} zero-mixture density limit"
            )
            curvature = probability * (1 - probability)
            posterior = expit(-zeta[zero] - base[zero])
            curvature[zero] -= posterior * (1 - posterior)
            information = (model.zi_X.T * (model.weights * curvature)) @ model.zi_X
            assert np.all(np.isfinite(information))
            curvature_norm = float(np.max(np.abs(np.linalg.eigvalsh(information))))
            assert curvature_norm <= policy["zi_curvature_bound"], (
                f"{label} negligible observed ZI curvature"
            )
            diagnostics[label] = {
                "probability_max": float(np.max(probability)),
                "probability_bound": probability_bound,
                "conditional_density_difference": density_difference,
                "observed_zi_curvature_norm": curvature_norm,
            }
    else:
        assert not any(field.startswith("zi_") for field in fields)
    for group, limits in policy.get("zero_random_effect", {}).items():
        reference = expected["VarCorr"]["cond"][group]
        names = np.atleast_1d(reference["names"]).tolist()
        levels = np.atleast_1d(expected["ranef"]["cond"][group]["levels"]).tolist()
        reference_covariance = np.asarray(reference["covariance"], dtype=float)
        covariance = model.VarCorr()["cond"][group].to_numpy()
        reference_modes = np.asarray(expected["ranef"]["cond"][group]["values"], dtype=float)
        modes = model.ranef()["cond"][group].loc[levels, names].to_numpy()
        assert covariance.shape == reference_covariance.shape
        assert modes.shape == reference_modes.shape
        for label, matrix, values in [
            ("R", reference_covariance, reference_modes), ("Python", covariance, modes),
        ]:
            assert np.all(np.isfinite(matrix)) and np.all(np.isfinite(values))
            assert np.max(np.abs(np.linalg.eigvalsh(matrix))) <= limits["covariance_bound"]
            assert np.max(np.abs(values)) <= limits["mode_bound"]
            diagnostics.setdefault(label, {})[group] = {
                "covariance_absolute_max": float(np.max(np.abs(matrix))),
                "mode_absolute_max": float(np.max(np.abs(values))),
            }
    return fields, diagnostics


def _summary_structure(actual, reference, expected):
    """Compare section order, labels and headers without formatted numbers."""
    prefixes = (
        "Family:", "Formula:", "Zero inflation:", "Dispersion:", "Data:", "Weights:",
        "Random effects:", "Conditional model:", "Number of obs:",
        "Dispersion parameter for", "Dispersion estimate for", "Zero-inflation model:",
        "Dispersion model:", "Signif. codes:",
    )

    def structure(text):
        result = []
        for raw in text.splitlines():
            line = " ".join(raw.split())
            if not line:
                continue
            if line.startswith(prefixes):
                result.append(line.split(":", 1)[0])
            elif line.startswith(("AIC ", "Groups ", "Estimate ")):
                result.append(line)
        return result

    assert structure(actual) == structure(reference), "summary sections, labels and headers"
    # Each fixed-effect row label belongs to its own model section, including
    # separate intercepts that appear in the conditional, zero and dispersion tables.
    for component, title in [
        ("cond", "Conditional model:"),
        ("zi", "Zero-inflation model:"),
        ("disp", "Dispersion model:"),
    ]:
        table = expected["coefficients"][component]
        names = np.atleast_1d(table["rows"]).tolist()
        if not names:
            continue
        for text in [actual, reference]:
            tail = text.rsplit(title, 1)[1]
            labels = []
            in_table = False
            for line in tail.splitlines():
                tokens = line.split()
                if not tokens:
                    continue
                if line.strip().endswith("model:") or tokens[0] in {"---", "Signif."}:
                    break
                if tokens[0] == "Estimate":
                    in_table = True
                elif in_table and tokens[0] in names:
                    labels.append(tokens[0])
            assert labels == names, f"summary {component} coefficient rows and order"
    for groups in expected["VarCorr"].values():
        if not groups:
            continue
        for group, covariance in groups.items():
            assert re.search(rf"^\s*{re.escape(group)}\s", actual, re.MULTILINE), (
                f"summary random-effect group {group}"
            )
            for name in np.atleast_1d(covariance["names"]):
                assert name in actual, f"summary random-effect label {name}"


@pytest.mark.golden
@pytest.mark.parametrize("path", FIXTURES, ids=lambda path: path.stem)
def test_tmb_golden(path, record_property):
    from rparity.tmb import glmmTMB

    fixture = json.loads(path.read_text())
    spec = fixture["spec"]
    observation = fixture["oracle"]
    expected = observation["result"]
    assert "error" not in expected, expected.get("error")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = glmmTMB(spec["formula"], pd.DataFrame(spec["data"]), family=spec["family"], link=spec["link"], ziformula=spec["ziformula"], dispformula=spec["dispformula"], weights=spec.get("weights"), offset=spec.get("offset"))
    reference_loglik = expected["logLik"]
    if reference_loglik is None:
        reference_loglik = -expected["optimizer_diagnostics"]["objective"]
    assert np.isfinite(model.logLik()), "finite normalized Laplace log likelihood"
    assert model.logLik() >= reference_loglik - 1e-6, (
        f"normalized log likelihood: Python {model.logLik():.15g}, "
        f"R {reference_loglik:.15g}"
    )
    record_property("reference_log_likelihood", float(reference_loglik))
    record_property("python_log_likelihood", float(model.logLik()))
    record_property("better_optimum", bool(model.logLik() > reference_loglik + 1e-6))
    messages = " ".join(np.atleast_1d(observation["warnings"]).tolist()).lower()
    diagnostics = expected["optimizer_diagnostics"]
    record_property("reference_convergence", diagnostics["convergence"])
    record_property("reference_pd_hessian", diagnostics["pdHess"])
    if not diagnostics["pdHess"] or diagnostics["convergence"] != 0 or any(word in messages for word in ["converg", "hessian", "singular"]):
        assert any(issubclass(warning.category, RuntimeWarning) for warning in caught), (
            "R convergence or singular warning requires a genuine Python RuntimeWarning"
        )
        return
    effects = model.fixef()
    policy = POLICIES.get(path.stem)
    fields = set()
    if policy is not None:
        fields, boundary_diagnostics = _boundary_limits(model, expected, policy)
        record_property("boundary_policy_fields", json.dumps(sorted(fields)))
        record_property("boundary_policy_guards", json.dumps(boundary_diagnostics))
    for component in ["cond", "zi", "disp"]:
        reference = np.atleast_1d(expected["fixef"][component]).astype(float)
        assert len(effects[component]) == len(reference)
        if not len(reference):
            continue
        names = np.atleast_1d(expected["fixef_names"][component]).tolist()
        assert list(effects[component].index) == names
        if f"{component}_beta" not in fields:
            np.testing.assert_allclose(
                effects[component], reference, rtol=1e-5, atol=1e-12,
                err_msg=f"{component} fixed-effect coefficients",
            )
        covariance = model.vcov(component)
        assert covariance.shape == (len(names), len(names))
        assert list(covariance.index) == names
        assert list(covariance.columns) == names
        reference_covariance = _reference_covariance(expected, component, names)
        if f"{component}_vcov" not in fields:
            np.testing.assert_allclose(
                covariance, reference_covariance, rtol=1e-4, atol=1e-12,
                err_msg=f"{component} fixed-effect covariance",
            )
        # R reports intercept-only dispersion via sigma, without a Wald table.
        table = expected["coefficients"][component]
        if (
            table and table.get("values") and "Pr(>|z|)" in table["values"]
            and f"{component}_wald_p" not in fields
        ):
            statistic = effects[component].to_numpy() / np.sqrt(np.diag(covariance))
            reference_p = np.asarray(table["values"]["Pr(>|z|)"], dtype=float)
            np.testing.assert_allclose(
                2 * norm.sf(np.abs(statistic)), reference_p, rtol=0, atol=1e-4,
                err_msg=f"{component} Wald p values",
            )
    np.testing.assert_allclose(
        model.AIC(), expected["AIC"], rtol=0, atol=2e-6, err_msg="AIC",
    )
    np.testing.assert_allclose(
        model.BIC(), expected["BIC"], rtol=0, atol=2e-6, err_msg="BIC",
    )
    assert model.df_resid == expected["df_resid"], "residual degrees of freedom"
    if len(effects["disp"]) <= 1:
        np.testing.assert_allclose(
            model.sigma(), expected["sigma"], rtol=1e-4, atol=1e-12,
            err_msg="intercept-only dispersion",
        )
    for kind in ["response", "cond", "zprob", "disp"]:
        if f"{kind}_prediction" in fields:
            # The indexed absent-ZI proof already rechecked both probability
            # and likelihood bounds. Other predictions remain strict.
            assert kind == "zprob" and policy["absent_zi"]
            continue
        np.testing.assert_allclose(
            model.predict(type=kind), expected["predictions"][kind], rtol=1e-6,
            atol=1e-12, err_msg=f"{kind} predictions",
        )
    for component, groups in expected["VarCorr"].items():
        if not groups:
            continue
        covariance = model.VarCorr()[component]
        modes = model.ranef()[component]
        assert set(covariance) == set(groups)
        for group, reference in groups.items():
            names = np.atleast_1d(reference["names"]).tolist()
            assert list(covariance[group].index) == names
            covariance_bound = 1e-12
            covariance_rtol = 1e-4
            if f"{group}_random_covariance" in fields:
                assert component == "cond" and group in policy["zero_random_effect"]
                covariance_bound = policy["zero_random_effect"][group]["covariance_bound"]
                covariance_rtol = 0
            np.testing.assert_allclose(
                covariance[group], reference["covariance"],
                rtol=covariance_rtol, atol=covariance_bound,
                err_msg=f"{component} {group} random-effect covariance",
            )
            random = expected["ranef"][component][group]
            levels = np.atleast_1d(random["levels"]).tolist()
            mode_bound = 1e-12
            mode_rtol = 1e-4
            if f"{group}_random_modes" in fields:
                assert component == "cond" and group in policy["zero_random_effect"]
                mode_bound = policy["zero_random_effect"][group]["mode_bound"]
                mode_rtol = 0
            np.testing.assert_allclose(
                modes[group].loc[levels, names], random["values"], rtol=mode_rtol,
                atol=mode_bound, err_msg=f"{component} {group} random modes",
            )
    _summary_structure(model.summary(), expected["summary"], expected)
