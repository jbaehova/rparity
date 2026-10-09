"""Strict independent R references on an indexed rank-one covariance face."""

import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import norm


def fingerprint(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _covariance(result, component, names):
    matrix = result["vcov_components"].get(component)
    if matrix is None:
        full_names = np.atleast_1d(result["vcov_full_names"]).tolist()
        indices = [full_names.index(f"{component}~{name}") for name in names]
        matrix = np.asarray(result["vcov_full"])[np.ix_(indices, indices)]
    return np.atleast_2d(np.asarray(matrix, dtype=float))


def _comparison(actual, reference, rtol, atol):
    actual = np.atleast_1d(np.asarray(actual, dtype=float))
    reference = np.atleast_1d(np.asarray(reference, dtype=float))
    assert actual.shape == reference.shape
    ratio = np.abs(actual - reference) / (atol + rtol * np.abs(reference))
    return {
        "pass": bool(np.all(np.isfinite(ratio)) and np.all(ratio <= 1)),
        "maximum_tolerance_ratio": float(np.max(ratio, initial=0)),
        "maximum_absolute_difference": float(np.max(np.abs(actual - reference), initial=0)),
    }


def _same_fit_inputs(original, auxiliary):
    """A different covariance parameterization may only change the formula."""
    assert "".join(auxiliary["formula"].split()) == "".join(original["formula"].replace(
        "(x|g)", "rr(1+x|g,d=1)",
    ).split())
    for name in [
        "call", "family", "link", "ziformula", "dispformula", "factors",
        "data", "weights", "offset", "tmb_control", "score_polish",
    ]:
        assert auxiliary.get(name) == original.get(name), f"unchanged fit input {name}"
    # Starting coordinates may change between parameterizations. Other
    # do.call arguments have to stay exactly the same.
    own_args = {k: v for k, v in original.get("args", {}).items() if k != "start"}
    auxiliary_args = {k: v for k, v in auxiliary.get("args", {}).items() if k != "start"}
    assert own_args == auxiliary_args


def _refined_auxiliary_covariance(spec, auxiliary, native):
    """Resolve score-difference truncation in an independent loading chart.

    Small physical loadings make a fixed 1e-3 loading-coordinate increment
    relatively large. Three independently observed public R Hessians allow
    two Richardson extrapolations to remove their quadratic step error.
    Their full inverses must agree at the original covariance tolerance.
    The reported covariance remains preserved in the auxiliary observation.
    """
    refinements = auxiliary["regular_hessian_refinement"]
    assert set(refinements) == {"0.000125", "0.00025", "0.0005"}
    hessians = {}
    for step, probe in refinements.items():
        _same_fit_inputs(spec, probe["spec"])
        assert probe["spec"]["tmb_covariance_diagnostics"]
        assert probe["spec"]["tmb_hessian_step"] == float(step)
        observation = probe["oracle"]
        reference = observation["result"]
        assert "error" not in reference and not observation["warnings"]
        assert reference["optimizer_diagnostics"]["convergence"] == 0
        assert reference["optimizer_diagnostics"]["pdHess"]
        diagnostic = reference["covariance_diagnostics"]
        assert diagnostic["finite_difference_step"] == float(step)
        np.testing.assert_array_equal(diagnostic["native_parameters"], native)
        score = np.asarray(diagnostic["score"], dtype=float)
        assert score.shape == native.shape and np.all(np.isfinite(score))
        assert np.max(np.abs(score)) <= 1e-7
        hessian = np.asarray(diagnostic["hessian"], dtype=float)
        assert hessian.shape == (len(native), len(native))
        assert np.all(np.isfinite(hessian))
        np.testing.assert_allclose(hessian, hessian.T, rtol=0, atol=1e-12)
        hessians[step] = hessian
    fine = (4 * hessians["0.000125"] - hessians["0.00025"]) / 3
    coarse = (4 * hessians["0.00025"] - hessians["0.0005"]) / 3
    eigenvalues = {}
    covariances = {}
    for label, hessian in [("fine", fine), ("coarse", coarse)]:
        values = np.linalg.eigvalsh(hessian)
        assert values[0] > np.sqrt(np.finfo(float).eps) * values[-1]
        eigenvalues[label] = values.tolist()
        covariances[label] = np.linalg.inv(hessian)
    np.testing.assert_allclose(
        covariances["coarse"], covariances["fine"], rtol=1e-4, atol=1e-12,
    )
    return covariances["coarse"], {
        "reference": "inverse Richardson-extrapolated public R loading-score Hessian",
        "steps": sorted(float(step) for step in hessians),
        "information_eigenvalues": eigenvalues,
        "full_covariance_stability": _comparison(
            covariances["coarse"], covariances["fine"], 1e-4, 1e-12,
        ),
        "reported_auxiliary_covariance_preserved": True,
    }


def rank_one_reference(model, spec, observation, policy, auxiliary, caught):
    """Reprove the exact physical face and compare every identified field.

    A rank-one random vector is loading times one standard Gaussian variable.
    Its covariance is loading loading.T. Public R rr(d=1) therefore removes
    the nonregular correlation coordinate without changing that physical
    face. Its finite Wald covariance is an independently fitted reference;
    its lower parameter count never replaces the original model's AIC or df.
    """
    assert policy["kind"] == "tmb_rank_one_auxiliary_reference"
    assert fingerprint({"spec": spec, "oracle": observation}) == policy["reference_fingerprint"]
    assert fingerprint(auxiliary) == policy["auxiliary_fingerprint"]
    _same_fit_inputs(spec, auxiliary["spec"])
    expected = observation["result"]
    reference = auxiliary["oracle"]["result"]
    assert not auxiliary["oracle"]["warnings"] and "error" not in reference
    assert reference["optimizer_diagnostics"]["convergence"] == 0
    assert reference["optimizer_diagnostics"]["pdHess"]
    assert len(model.random_blocks) == 1
    block = model.random_blocks[0]
    group = policy["group"]
    assert block.term.group == group
    assert block.names == policy["random_design_columns"]
    assert block.names == ["(Intercept)", "x"]
    assert any(issubclass(warning.category, RuntimeWarning) for warning in caught), (
        "The original nonregular correlation model requires its genuine Python warning"
    )

    information_bound = np.sqrt(np.finfo(float).eps)
    diagnostic = reference["covariance_diagnostics"]
    native = np.asarray(diagnostic["native_parameters"], dtype=float)
    assert native.shape == (len(model.parameters) - 1,)
    assert diagnostic["native_parameter_names"][-2:] == ["theta", "theta"]
    auxiliary_coefficients = np.concatenate([
        np.atleast_1d(reference["fixef"][component]).astype(float)
        for component in ["cond", "zi", "disp"]
    ])
    np.testing.assert_array_equal(native[:-2], auxiliary_coefficients)
    loading = native[-2:]
    reference_random = reference["VarCorr"]["cond"][group]
    assert np.atleast_1d(reference_random["names"]).tolist() == block.names
    reference_covariance = np.asarray(reference_random["covariance"], dtype=float)
    np.testing.assert_allclose(
        np.outer(loading, loading), reference_covariance, rtol=1e-4, atol=1e-12,
    )
    physical_loading = np.array([
        np.sqrt(reference_covariance[0, 0]),
        reference_covariance[1, 0] / np.sqrt(reference_covariance[0, 0]),
    ])
    np.testing.assert_allclose(
        np.outer(physical_loading, physical_loading), reference_covariance, rtol=0,
        atol=64 * np.finfo(float).eps * max(1, np.linalg.norm(reference_covariance, 2)),
    )
    physical_eigenvalues = {}
    for label, matrix in [
        ("original_R", expected["VarCorr"]["cond"][group]["covariance"]),
        ("auxiliary_R", reference_covariance),
        ("Python", model.VarCorr()["cond"][group].to_numpy()),
    ]:
        matrix = np.asarray(matrix, dtype=float)
        assert matrix.shape == (2, 2) and np.all(np.isfinite(matrix))
        eigenvalues = np.linalg.eigvalsh(matrix)
        assert abs(eigenvalues[0]) < 1e-8 and eigenvalues[1] > 1e-8
        physical_eigenvalues[label] = eigenvalues.tolist()
        np.testing.assert_allclose(matrix, reference_covariance, rtol=1e-4, atol=1e-12)

    auxiliary_hessian = np.asarray(diagnostic["hessian"], dtype=float)
    assert auxiliary_hessian.shape == (len(native), len(native))
    assert np.all(np.isfinite(auxiliary_hessian))
    np.testing.assert_allclose(auxiliary_hessian, auxiliary_hessian.T, rtol=0, atol=1e-12)
    auxiliary_eigenvalues = np.linalg.eigvalsh(auxiliary_hessian)
    assert auxiliary_eigenvalues[0] > information_bound * auxiliary_eigenvalues[-1]
    auxiliary_score = np.asarray(diagnostic["score"], dtype=float)
    assert auxiliary_score.shape == native.shape and np.all(np.isfinite(auxiliary_score))
    assert np.max(np.abs(auxiliary_score)) <= 1e-7
    auxiliary_wald = np.asarray(reference["vcov_full"], dtype=float)
    assert auxiliary_wald.shape == auxiliary_hessian.shape
    assert np.all(np.isfinite(auxiliary_wald)) and np.linalg.eigvalsh(auxiliary_wald)[0] > 0
    refined_covariance = None
    refinement_diagnostics = None
    if "regular_hessian_refinement" in auxiliary:
        refined_covariance, refinement_diagnostics = _refined_auxiliary_covariance(
            spec, auxiliary, native,
        )

    original_probe = auxiliary["original_R_information_probe"]
    original_native = np.asarray(original_probe["R_native_parameters"], dtype=float)
    original_coefficients = np.concatenate([
        np.atleast_1d(expected["fixef"][component]).astype(float)
        for component in ["cond", "zi", "disp"]
    ])
    np.testing.assert_array_equal(original_native[: len(original_coefficients)], original_coefficients)
    rho = original_native[-1] / np.sqrt(1 + original_native[-1] ** 2)
    sd = np.exp(original_native[-3:-1])
    reconstructed = np.outer(sd, sd) * np.array([[1, rho], [rho, 1]])
    np.testing.assert_allclose(
        reconstructed, expected["VarCorr"]["cond"][group]["covariance"], rtol=1e-4,
        atol=1e-12,
    )
    # rho = theta/sqrt(1+theta**2). The physical covariance derivative
    # in the discarded scaled-correlation coordinate vanishes as theta
    # diverges. This proves the boundary independently of an unreliable
    # eigenvector from the original ill-conditioned numerical Hessian.
    covariance_derivative = np.array([
        [0, sd[0] * sd[1] / (1 + original_native[-1] ** 2) ** 1.5],
        [sd[0] * sd[1] / (1 + original_native[-1] ** 2) ** 1.5, 0],
    ])
    assert np.linalg.norm(covariance_derivative, 2) <= (
        information_bound * max(1, np.linalg.norm(reconstructed, 2))
    )
    original_hessian = np.asarray(original_probe["R_Hessian"], dtype=float)
    assert original_hessian.shape == model.outer_information.shape
    assert np.all(np.isfinite(original_hessian))
    np.testing.assert_allclose(original_hessian, original_hessian.T, rtol=0, atol=1e-12)
    eigenvalues, eigenvectors = np.linalg.eigh(original_hessian)
    weak = np.argmin(np.abs(eigenvalues))
    assert abs(eigenvalues[weak]) <= information_bound * max(1, np.max(np.abs(eigenvalues)))
    assert np.linalg.cond(np.asarray(expected["vcov_full"])) >= 1 / information_bound
    own_information = np.asarray(model.outer_information, dtype=float)
    assert np.all(np.isfinite(own_information))
    own_norm = max(1.0, np.linalg.norm(own_information, 2))
    assert np.linalg.norm(own_information[:, -1]) <= information_bound * own_norm
    assert np.linalg.norm(model.information_transform[:, -1]) <= information_bound
    retained_eigenvalues = np.linalg.eigvalsh(own_information[:-1, :-1])
    assert retained_eigenvalues[0] > information_bound * retained_eigenvalues[-1]
    assert np.max(np.abs(model.outer_gradient)) <= 1e-7

    original_loglik = expected["logLik"]
    if original_loglik is None:
        original_loglik = -expected["optimizer_diagnostics"]["objective"]
    reference_loglik = reference["logLik"]
    assert np.isfinite(reference_loglik)
    assert abs(reference_loglik - original_loglik) <= 1e-6
    assert abs(model.logLik() - reference_loglik) <= 1e-6
    covariances = {}
    p_values = {}
    original_comparisons = {}
    cursor = 0
    for component in ["cond", "zi", "disp"]:
        original = np.atleast_1d(expected["fixef"][component]).astype(float)
        beta = np.atleast_1d(reference["fixef"][component]).astype(float)
        assert len(model.fixef()[component]) == len(original) == len(beta)
        if not len(beta):
            continue
        names = np.atleast_1d(reference["fixef_names"][component]).tolist()
        assert names == np.atleast_1d(expected["fixef_names"][component]).tolist()
        assert list(model.fixef()[component].index) == names
        np.testing.assert_allclose(original, beta, rtol=1e-5, atol=1e-12)
        np.testing.assert_allclose(model.fixef()[component], beta, rtol=1e-5, atol=1e-12)
        if refined_covariance is None:
            covariance = _covariance(reference, component, names)
        else:
            covariance = refined_covariance[cursor : cursor + len(beta), cursor : cursor + len(beta)]
        cursor += len(beta)
        covariances[component] = covariance
        np.testing.assert_allclose(model.vcov(component), covariance, rtol=1e-4, atol=1e-12)
        original_comparisons[f"{component}_vcov"] = _comparison(
            model.vcov(component), _covariance(expected, component, names), 1e-4, 1e-12,
        )
        table = reference["coefficients"][component]
        if "Pr(>|z|)" in table.get("values", {}):
            if refined_covariance is None:
                p_values[component] = np.asarray(table["values"]["Pr(>|z|)"], dtype=float)
            else:
                p_values[component] = 2 * norm.sf(np.abs(beta / np.sqrt(np.diag(covariance))))
            statistic = model.fixef()[component].to_numpy() / np.sqrt(np.diag(model.vcov(component)))
            actual_p = 2 * norm.sf(np.abs(statistic))
            np.testing.assert_allclose(actual_p, p_values[component], rtol=0, atol=1e-4)
            original_comparisons[f"{component}_wald_p"] = _comparison(
                actual_p, expected["coefficients"][component]["values"]["Pr(>|z|)"], 0, 1e-4,
            )

    prediction_fields = policy["prediction_limit_fields"]
    assert set(prediction_fields) <= {"response", "cond", "disp"}
    predictions = {}
    for kind in ["response", "cond", "zprob", "disp"]:
        actual = model.predict(type=kind)
        np.testing.assert_allclose(actual, reference["predictions"][kind], rtol=1e-6, atol=1e-12)
        original_comparisons[f"{kind}_prediction"] = _comparison(
            actual, expected["predictions"][kind], 1e-6, 1e-12,
        )
        if kind in prediction_fields:
            predictions[kind] = reference["predictions"][kind]
    levels = np.atleast_1d(reference["ranef"]["cond"][group]["levels"]).tolist()
    assert levels == np.atleast_1d(expected["ranef"]["cond"][group]["levels"]).tolist()
    modes = reference["ranef"]["cond"][group]["values"]
    np.testing.assert_allclose(
        expected["ranef"]["cond"][group]["values"], modes, rtol=1e-4, atol=1e-12,
    )
    np.testing.assert_allclose(
        model.ranef()["cond"][group].loc[levels, block.names], modes, rtol=1e-4, atol=1e-12,
    )
    newdata_comparisons = {}
    for kind, new_reference in auxiliary["newdata"].items():
        _same_fit_inputs(spec, new_reference["spec"])
        assert new_reference["spec"]["operation"] == "predict"
        assert new_reference["spec"]["prediction_type"] == kind
        n_rows = len(next(iter(spec["newdata"].values())))
        for name, values in spec["newdata"].items():
            assert new_reference["spec"]["newdata"][name][:n_rows] == values
        new_result = new_reference["oracle"]["result"]
        assert "error" not in new_result and not new_reference["oracle"]["warnings"]
        assert abs(new_result["logLik"] - reference_loglik) <= 1e-6
        actual = model.predict(pd.DataFrame(spec["newdata"]), type=kind)
        new_prediction = np.asarray(new_result["prediction"])[:n_rows]
        np.testing.assert_allclose(actual, new_prediction, rtol=1e-6, atol=1e-12)
        newdata_comparisons[kind] = _comparison(actual, new_prediction, 1e-6, 1e-12)
    assert set(newdata_comparisons) == {"response", "cond"}
    return covariances, p_values, predictions, {
        "reference": "public R rr(d=1) exact physical rank-one auxiliary fit",
        "original_observation_preserved": True,
        "nominal_AIC_BIC_df_reference": "original unstructured model, unchanged",
        "auxiliary_parameter_count": len(native),
        "original_parameter_count": len(model.parameters),
        "auxiliary_minus_original_log_likelihood": float(reference_loglik - original_loglik),
        "python_minus_auxiliary_log_likelihood": float(model.logLik() - reference_loglik),
        "physical_covariance_eigenvalues": physical_eigenvalues,
        "auxiliary_score_max": float(np.max(np.abs(auxiliary_score))),
        "python_score_max": float(np.max(np.abs(model.outer_gradient))),
        "auxiliary_information_minimum_eigenvalue": float(auxiliary_eigenvalues[0]),
        "python_retained_information_minimum_eigenvalue": float(retained_eigenvalues[0]),
        "original_weak_information_eigenvalue": float(eigenvalues[weak]),
        "original_weak_direction_correlation_loading": float(abs(eigenvectors[-1, weak])),
        "original_physical_correlation_derivative_norm": float(
            np.linalg.norm(covariance_derivative, 2),
        ),
        "original_raw_comparisons": original_comparisons,
        "prediction_limit_fields": prediction_fields,
        "newdata_comparisons": newdata_comparisons,
        "regular_hessian_refinement": refinement_diagnostics,
    }
