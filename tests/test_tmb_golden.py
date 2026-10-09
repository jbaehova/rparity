"""Strict normalized-Laplace and component-specific Stage 2 R observations."""

import copy
import hashlib
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _tmb_dispersion_limits import dispersion_reference
from _tmb_rank_one_limits import fingerprint, rank_one_reference
from scipy.special import expit, logit
from scipy.stats import norm

FIXTURES = sorted((Path(__file__).parent / "golden/tmb").glob("*.json"))
POLICIES = json.loads((Path(__file__).parent / "golden/tmb_case_policy.json").read_text())
RANK_ONE_REFERENCES = json.loads(
    (Path(__file__).parent / "golden/tmb_rank_one_reference.json").read_text(),
)
DISPERSION_REFERENCES = json.loads(
    (Path(__file__).parent / "golden/tmb_dispersion_reference.json").read_text(),
)


def _reference_covariance(expected, component, names):
    covariance = expected["vcov_components"].get(component)
    if covariance is None:
        full_names = np.atleast_1d(expected["vcov_full_names"]).tolist()
        indices = [full_names.index(f"{component}~{name}") for name in names]
        covariance = np.asarray(expected["vcov_full"], dtype=float)[np.ix_(indices, indices)]
    return np.atleast_2d(np.asarray(covariance, dtype=float))


def _correlation_tangent_reference(model, spec, observation, policy, caught):
    """Recheck a fingerprinted rank-one correlation boundary and its tangent.

    A scaled-correlation coordinate diverges at correlation +/-1. Inverting
    its numerically zero information direction can contaminate an otherwise
    identified covariance block. The public R score Hessian restricted to
    the remaining coordinates supplies an independent limiting covariance.
    This does not waive covariance or Wald comparisons: they retain the
    original tolerances against that explicitly derived R reference.
    """
    assert policy["kind"] == "tmb_correlation_tangent_reference"
    payload = json.dumps(
        {"spec": spec, "oracle": observation}, sort_keys=True, separators=(",", ":"),
    ).encode()
    assert hashlib.sha256(payload).hexdigest() == policy["reference_fingerprint"]
    expected = observation["result"]
    hessian = np.asarray(policy["public_R_hessian"], dtype=float)
    assert hessian.shape == model.outer_information.shape
    assert np.all(np.isfinite(hessian))
    np.testing.assert_allclose(hessian, hessian.T, rtol=0, atol=1e-12)
    coordinate = policy["correlation_direction_index"]
    native_names = policy["public_R_parameter_names"]
    assert native_names[coordinate] == "theta"
    assert len(native_names) == len(model.parameters)
    assert coordinate == len(model.parameters) - 1
    reference_parameters = np.asarray(policy["public_R_parameters"], dtype=float)
    reference_coefficients = np.concatenate([
        np.atleast_1d(expected["fixef"][component]).astype(float)
        for component in ["cond", "zi", "disp"]
    ])
    np.testing.assert_array_equal(
        reference_parameters[: len(reference_coefficients)], reference_coefficients,
    )
    # The indexed corpus contains one unstructured two-dimensional block.
    assert len(model.random_blocks) == 1 and len(model.random_blocks[0].names) == 2
    assert model.random_blocks[0].term.group == policy["group"]
    reference_random = expected["VarCorr"]["cond"][policy["group"]]
    scaled_correlation = reference_parameters[coordinate]
    correlation = scaled_correlation / np.sqrt(1 + scaled_correlation**2)
    standard_deviations = np.exp(reference_parameters[-3:-1])
    reconstructed_covariance = np.outer(standard_deviations, standard_deviations) * np.array([
        [1, correlation], [correlation, 1],
    ])
    np.testing.assert_allclose(
        reconstructed_covariance, reference_random["covariance"], rtol=1e-4, atol=1e-12,
    )
    np.testing.assert_allclose(
        correlation,
        np.asarray(reference_random["correlation"])[0, 1], rtol=0, atol=1e-12,
    )
    information_bound = np.sqrt(np.finfo(float).eps)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    weak = int(np.argmin(np.abs(eigenvalues)))
    information_norm = max(1.0, float(np.max(np.abs(eigenvalues))))
    assert abs(eigenvalues[weak]) <= information_bound * information_norm
    assert abs(eigenvectors[coordinate, weak]) >= 1 - information_bound, (
        "The weak R information direction must be the nonregular correlation"
    )
    reported_covariance = np.asarray(expected["vcov_full"], dtype=float)
    assert np.all(np.isfinite(reported_covariance))
    assert np.linalg.cond(reported_covariance) >= 1 / information_bound
    score = np.asarray(policy["public_R_score"], dtype=float)
    assert score.shape == model.parameters.shape and np.all(np.isfinite(score))
    assert np.max(np.abs(score)) <= policy["reference_score_bound"]
    assert np.max(np.abs(model.outer_gradient)) <= policy["python_score_bound"]
    own_information = np.asarray(model.outer_information, dtype=float)
    assert np.all(np.isfinite(own_information))
    own_norm = max(1.0, float(np.linalg.norm(own_information, 2)))
    assert np.linalg.norm(own_information[:, coordinate]) <= information_bound * own_norm
    assert np.linalg.norm(model.information_transform[:, coordinate]) <= information_bound
    assert any(issubclass(warning.category, RuntimeWarning) for warning in caught), (
        "The correlation boundary must produce a genuine Python curvature warning"
    )
    covariance_eigenvalues = {}
    for label, covariance in [
        ("R", expected["VarCorr"]["cond"][policy["group"]]["covariance"]),
        ("Python", model.VarCorr()["cond"][policy["group"]].to_numpy()),
    ]:
        matrix = np.asarray(covariance, dtype=float)
        assert matrix.shape == (2, 2) and np.all(np.isfinite(matrix))
        values = np.linalg.eigvalsh(matrix)
        assert abs(values[0]) < 1e-8 and values[1] > 1e-8
        covariance_eigenvalues[label] = values.tolist()
    keep = np.delete(np.arange(len(hessian)), coordinate)
    retained_information = hessian[np.ix_(keep, keep)]
    retained_eigenvalues = np.linalg.eigvalsh(retained_information)
    assert retained_eigenvalues[0] > information_bound * np.max(retained_eigenvalues), (
        "All retained R tangent directions must have positive, resolved information"
    )
    own_retained = own_information[np.ix_(keep, keep)]
    own_retained_eigenvalues = np.linalg.eigvalsh(own_retained)
    assert own_retained_eigenvalues[0] > information_bound * np.max(own_retained_eigenvalues)
    tangent_covariance = np.linalg.inv(retained_information)
    components = {}
    cursor = 0
    for component in ["cond", "zi", "disp"]:
        size = len(model.fixef()[component])
        components[component] = tangent_covariance[cursor : cursor + size, cursor : cursor + size]
        cursor += size
    assert cursor + 2 == len(keep)
    return components, {
        "covariance_reference": "inverse public R Hessian on identified correlation tangent",
        "correlation_direction_index": coordinate,
        "R_information_weak_eigenvalue": float(eigenvalues[weak]),
        "R_information_norm": information_norm,
        "R_reported_covariance_condition": float(np.linalg.cond(reported_covariance)),
        "R_retained_information_minimum_eigenvalue": float(retained_eigenvalues[0]),
        "Python_retained_information_minimum_eigenvalue": float(own_retained_eigenvalues[0]),
        "R_score_max": float(np.max(np.abs(score))),
        "Python_score_max": float(np.max(np.abs(model.outer_gradient))),
        "physical_covariance_eigenvalues": covariance_eigenvalues,
        "original_R_Wald_covariance_comparison_claimed": False,
    }


def _singular_reference_limit(model, spec, observation, policy, caught):
    """Verify an indexed, numerically singular public R information matrix.

    This is the TASK singular-fit likelihood/warning rule. It never declares
    an undefined R Wald covariance equal to a finite Python covariance. The
    reference fingerprint ties its independently observed Hessian to this
    exact input and output, including the original R convergence metadata.
    """
    assert policy["kind"] == "tmb_singular_reference"
    payload = json.dumps(
        {"spec": spec, "oracle": observation}, sort_keys=True, separators=(",", ":"),
    ).encode()
    assert hashlib.sha256(payload).hexdigest() == policy["reference_fingerprint"]
    expected = observation["result"]
    reference_covariance = np.asarray(expected["vcov_full"], dtype=float)
    assert not np.any(np.isfinite(reference_covariance)), "undefined R Wald covariance"
    hessian = np.asarray(policy["public_R_hessian"], dtype=float)
    assert hessian.shape == (len(model.parameters), len(model.parameters))
    assert np.all(np.isfinite(hessian))
    np.testing.assert_allclose(hessian, hessian.T, rtol=0, atol=1e-12)
    eigenvalues = np.linalg.eigvalsh(hessian)
    norm = max(1.0, float(np.max(np.abs(eigenvalues))))
    assert np.min(np.abs(eigenvalues)) <= policy["information_relative_bound"] * norm
    score = np.asarray(policy["public_R_score"], dtype=float)
    assert score.shape == model.parameters.shape and np.all(np.isfinite(score))
    assert np.max(np.abs(score)) <= policy["reference_score_bound"]
    assert any(issubclass(warning.category, RuntimeWarning) for warning in caught), (
        "A singular R reference still requires a genuine Python RuntimeWarning"
    )
    own_information = np.asarray(model.outer_information, dtype=float)
    assert own_information.shape == hessian.shape and np.all(np.isfinite(own_information))
    own_eigenvalues = np.linalg.eigvalsh(own_information)
    own_information_norm = max(1.0, float(np.max(np.abs(own_eigenvalues))))
    own_singular_information = bool(
        np.min(np.abs(own_eigenvalues))
        <= policy["information_relative_bound"] * own_information_norm
    )
    own_score = np.asarray(model.outer_gradient, dtype=float)
    assert own_score.shape == score.shape and np.all(np.isfinite(own_score))
    assert np.max(np.abs(own_score)) <= policy["reference_score_bound"]
    own_zero_variance = any(
        np.min(np.linalg.eigvalsh(matrix)) < 1e-8
        for matrix in model.random_covariances
    )
    assert own_singular_information or own_zero_variance, (
        "Python warning must be supported by its own measured information or variance"
    )
    return {
        "R_information_smallest_absolute_eigenvalue": float(np.min(np.abs(eigenvalues))),
        "R_information_norm": norm,
        "R_information_relative_bound": policy["information_relative_bound"],
        "R_score_max": float(np.max(np.abs(score))),
        "R_Wald_covariance_defined": False,
        "Python_singular_information": own_singular_information,
        "Python_information_smallest_absolute_eigenvalue": float(
            np.min(np.abs(own_eigenvalues))
        ),
        "Python_information_norm": own_information_norm,
        "Python_score_max": float(np.max(np.abs(own_score))),
        "Python_zero_variance": own_zero_variance,
    }


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
    dispersion_fields = set()
    dispersion_proof = None
    if path.stem in DISPERSION_REFERENCES:
        expected, dispersion_fields, dispersion_proof = dispersion_reference(
            model, spec, observation, DISPERSION_REFERENCES[path.stem], caught,
        )
        record_property("dispersion_reference", json.dumps(dispersion_proof, sort_keys=True))
    reference_loglik = observation["result"]["logLik"]
    if reference_loglik is None:
        reference_loglik = -observation["result"]["optimizer_diagnostics"]["objective"]
    record_property("reference_log_likelihood", float(reference_loglik))
    record_property("python_log_likelihood", float(model.logLik()))
    record_property("better_optimum", bool(model.logLik() > reference_loglik + 1e-6))
    assert np.isfinite(model.logLik()), "finite normalized Laplace log likelihood"
    assert model.logLik() >= reference_loglik - 1e-6, (
        f"normalized log likelihood: Python {model.logLik():.15g}, "
        f"R {reference_loglik:.15g}"
    )
    messages = " ".join(np.atleast_1d(observation["warnings"]).tolist()).lower()
    diagnostics = expected["optimizer_diagnostics"]
    record_property("reference_convergence", diagnostics["convergence"])
    record_property("reference_pd_hessian", diagnostics["pdHess"])
    policy = POLICIES.get(path.stem)
    tangent_covariances = None
    rank_one_p_values = {}
    rank_one_predictions = {}
    if policy is not None and policy["kind"] == "tmb_rank_one_auxiliary_reference":
        (
            tangent_covariances, rank_one_p_values, rank_one_predictions,
            rank_one_diagnostics,
        ) = rank_one_reference(
            model, spec, observation, policy, RANK_ONE_REFERENCES[path.stem], caught,
        )
        record_property("rank_one_auxiliary_reference", json.dumps(rank_one_diagnostics))
    if policy is not None and policy["kind"] == "tmb_correlation_tangent_reference":
        tangent_covariances, tangent_diagnostics = _correlation_tangent_reference(
            model, spec, observation, policy, caught,
        )
        record_property("correlation_tangent_reference", json.dumps(tangent_diagnostics))
    if policy is not None and policy["kind"] == "tmb_singular_reference":
        singular_diagnostics = _singular_reference_limit(
            model, spec, observation, policy, caught,
        )
        record_property("reference_singular_information", json.dumps(singular_diagnostics))
        return
    if dispersion_proof is None and tangent_covariances is None and (not diagnostics["pdHess"] or diagnostics["convergence"] != 0 or any(word in messages for word in ["converg", "hessian", "singular"])):
        assert any(issubclass(warning.category, RuntimeWarning) for warning in caught), (
            "R convergence or singular warning requires a genuine Python RuntimeWarning"
        )
        return
    effects = model.fixef()
    fields = set(dispersion_fields)
    if policy is not None and policy["kind"] == "tmb_boundary_identification":
        boundary_fields, boundary_diagnostics = _boundary_limits(model, expected, policy)
        fields.update(boundary_fields)
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
        if tangent_covariances is not None:
            reference_covariance = tangent_covariances[component]
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
            if component in rank_one_p_values:
                reference_p = rank_one_p_values[component]
            elif tangent_covariances is not None:
                reference_p = 2 * norm.sf(np.abs(reference / np.sqrt(np.diag(reference_covariance))))
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
    if len(effects["disp"]) <= 1 and "sigma" not in dispersion_fields:
        np.testing.assert_allclose(
            model.sigma(), expected["sigma"], rtol=1e-4, atol=1e-12,
            err_msg="intercept-only dispersion",
        )
    for kind in ["response", "cond", "zprob", "disp"]:
        if f"{kind}_prediction" in fields:
            if f"{kind}_prediction" in dispersion_fields:
                assert kind == "disp" and dispersion_proof
                continue
            # The indexed absent-ZI proof already rechecked both probability
            # and likelihood bounds. Other predictions remain strict.
            assert kind == "zprob" and policy["absent_zi"]
            continue
        np.testing.assert_allclose(
            model.predict(type=kind),
            rank_one_predictions.get(kind, expected["predictions"][kind]), rtol=1e-6,
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


@pytest.fixture(scope="module")
def rank_one_guard_example():
    from rparity.tmb import glmmTMB

    fixture = json.loads((Path(__file__).parent / "golden/tmb/tmb_0162.json").read_text())
    spec = fixture["spec"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = glmmTMB(
            spec["formula"], pd.DataFrame(spec["data"]), family=spec["family"],
            link=spec["link"], ziformula=spec["ziformula"],
            dispformula=spec["dispformula"], weights=spec.get("weights"),
            offset=spec.get("offset"),
        )
    return model, spec, fixture["oracle"], POLICIES["tmb_0162"], RANK_ONE_REFERENCES["tmb_0162"], caught


@pytest.mark.parametrize("mutation", [
    "original_observation", "auxiliary_observation", "changed_response",
    "full_rank_covariance", "nonstationary_auxiliary", "indefinite_auxiliary",
    "resolved_python_correlation", "missing_warning",
])
def test_rank_one_limit_rejects_hostile_mutation(rank_one_guard_example, mutation):
    model, spec, observation, policy, auxiliary, caught = copy.deepcopy(rank_one_guard_example)
    reseal = False
    if mutation == "original_observation":
        observation["result"]["fixef"]["cond"][0] += 1
    elif mutation == "auxiliary_observation":
        auxiliary["oracle"]["result"]["fixef"]["cond"][0] += 1
    elif mutation == "changed_response":
        auxiliary["spec"]["data"]["y"][0] += 1
        reseal = True
    elif mutation == "full_rank_covariance":
        auxiliary["oracle"]["result"]["VarCorr"]["cond"]["g"]["covariance"][1][1] += 1e-5
        reseal = True
    elif mutation == "nonstationary_auxiliary":
        auxiliary["oracle"]["result"]["covariance_diagnostics"]["score"][0] = 1e-3
        reseal = True
    elif mutation == "indefinite_auxiliary":
        auxiliary["oracle"]["result"]["covariance_diagnostics"]["hessian"][0][0] = -1
        reseal = True
    elif mutation == "resolved_python_correlation":
        model.outer_information[-1, -1] = 1
    elif mutation == "missing_warning":
        caught = []
    if reseal:
        policy["auxiliary_fingerprint"] = fingerprint(auxiliary)
    with pytest.raises(AssertionError):
        rank_one_reference(model, spec, observation, policy, auxiliary, caught)


def test_rank_one_auxiliary_scope_requires_indexed_proofs():
    assert len(RANK_ONE_REFERENCES) == 12
    for case in ["tmb_1047", "tmb_1180"]:
        assert RANK_ONE_REFERENCES[case]["regular_hessian_refinement"]
        assert POLICIES[case]["kind"] == "tmb_rank_one_auxiliary_reference"
    for case in ["tmb_0000", "tmb_0037"]:
        assert case not in RANK_ONE_REFERENCES
