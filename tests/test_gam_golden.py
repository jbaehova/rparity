"""Stage 2 GAM parity, including basis-invariant public R observations."""

import json
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _gam_gauge_limits import numerical_gauge_penalty_limit
from _gam_limits import numerical_positive_penalty_limit
from numpy.polynomial import polynomial
from scipy.linalg import eigh, null_space, pinvh

FIXTURES = sorted((Path(__file__).parent / "golden/gam").glob("*.json"))
POLICIES = json.loads((Path(__file__).parent / "golden/stage2_case_policy.json").read_text())
BOUNDARIES = json.loads((Path(__file__).parent / "golden/gam_boundary_policy.json").read_text())
VANISHING = json.loads((Path(__file__).parent / "golden/gam_vanishing_policy.json").read_text())
GAUGE_LIMITS = json.loads((Path(__file__).parent / "golden/gam_gauge_vanishing_policy.json").read_text())


def _table(observation):
    return pd.DataFrame(
        {name: np.atleast_1d(value) for name, value in observation["values"].items()},
        index=np.atleast_1d(observation["rows"]),
        dtype=float,
    )


def _coordinate_map(model, expected):
    """Map independently chosen bases using each public smooth's function space.

    Blockwise mapping preserves a random-effect smooth's identity when its
    dummy columns overlap the fixed intercept. A whole-design least-squares
    map would be ambiguous in that otherwise legitimate penalized design.
    """
    reference = np.asarray(expected["X"], dtype=float)
    actual = np.asarray(model.X, dtype=float)
    assert reference.shape == actual.shape
    transform = np.zeros((reference.shape[1], actual.shape[1]))
    smooths = expected["smooths"]
    first = min(smooth["first_para"] for smooth in smooths) - 1
    blocks = [(np.arange(first), np.asarray(model.design.parametric_indices, dtype=int))]
    own_smooths = {smooth.label: smooth for smooth in model.design.smooths}
    for smooth in smooths:
        assert smooth["label"] in own_smooths
        reference_indices = np.arange(smooth["first_para"] - 1, smooth["last_para"])
        blocks.append((reference_indices, own_smooths[smooth["label"]].indices))
    for reference_indices, actual_indices in blocks:
        assert len(reference_indices) == len(actual_indices)
        block = np.linalg.lstsq(reference[:, reference_indices], actual[:, actual_indices], rcond=1e-12)[0]
        np.testing.assert_allclose(reference[:, reference_indices] @ block, actual[:, actual_indices], rtol=1e-9, atol=1e-10)
        transform[np.ix_(reference_indices, actual_indices)] = block
    assert np.linalg.matrix_rank(transform) == len(transform)
    np.testing.assert_allclose(reference @ transform, actual, rtol=1e-9, atol=1e-10)
    return transform


def _summary_structure(actual, reference):
    sections = ["Family:", "Link function:", "Formula:", "Parametric coefficients:", "Approximate significance of smooth terms:", "R-sq.(adj)"]
    for text in [actual, reference]:
        positions = [text.index(section) for section in sections]
        assert positions == sorted(positions)
    for prefix in ["Family:", "Link function:"]:
        left = next(line.strip() for line in actual.splitlines() if line.startswith(prefix))
        right = next(line.strip() for line in reference.splitlines() if line.startswith(prefix))
        assert left == right
    for marker in ["Estimate", "Ref.df"]:
        left = next(line.split() for line in actual.splitlines() if marker in line)
        right = next(line.split() for line in reference.splitlines() if marker in line)
        assert left == right


def _identifiable_projection(model, expected, transform, policy):
    """Verify an indexed joint nullspace and compare its identifiable quotient.

    A duplicated numeric by-variable can move between its fixed coefficient
    and the smooth's unpenalized constant without changing any fitted function
    or roughness. R's numerical pivot and our constraint need not choose the
    same representative of that one equivalence class.
    """
    assert policy["kind"] == "gam_coefficient_gauge"
    reference = np.asarray(expected["X"], dtype=float)
    null = null_space(reference, rcond=1e-12)
    assert null.shape[1] == policy["joint_nullity"]
    np.testing.assert_allclose(reference @ null, 0, atol=1e-10)
    for smooth in expected["smooths"]:
        indices = np.arange(smooth["first_para"] - 1, smooth["last_para"])
        for penalty in smooth["S"]:
            matrix = np.asarray(penalty)
            np.testing.assert_allclose(matrix @ null[indices], 0,
                                       atol=1e-10 * max(1, np.linalg.norm(matrix)))
    own_null = np.linalg.solve(transform, null)
    np.testing.assert_allclose(model.X @ own_null, 0, atol=1e-10)
    for matrix in model.design.S:
        np.testing.assert_allclose(matrix @ own_null, 0,
                                   atol=1e-10 * max(1, np.linalg.norm(matrix)))
    return np.eye(reference.shape[1]) - null @ null.T


def _positive_boundary_certificate(method, values, scores, raw_values, deviance, degrees):
    """Certify the entire nonnegative variance axis, not a sampled profile.

    Gaussian one-penalty REML, ML and GCV derivatives are rational functions
    of variance tau. Their denominators are positive. Strictly positive
    numerator coefficients therefore prove the global minimum is tau=0.
    Long double arithmetic and a cancellation margin protect this sufficient
    certificate; a failed certificate never authorizes a boundary policy.
    """
    def product(excluded=(), spectrum=values):
        result = np.ones(1, dtype=np.longdouble)
        for index, value in enumerate(spectrum):
            if index not in excluded:
                result = polynomial.polymul(result, np.array([1, value], dtype=np.longdouble))
        return result

    denominator = product()
    residual = (deviance - float(np.sum(scores))) * denominator
    score_numerator = np.zeros(1, dtype=np.longdouble)
    trace_numerator = np.zeros(1, dtype=np.longdouble)
    for index, value in enumerate(values):
        remaining = product((index,))
        residual = polynomial.polyadd(residual, scores[index] * remaining)
        squared = polynomial.polymul(remaining, remaining)
        score_numerator = polynomial.polyadd(score_numerator, scores[index] * value * squared)
        trace_numerator = polynomial.polyadd(trace_numerator, value * squared)
    if method == "REML":
        determinant_score = np.zeros(1, dtype=np.longdouble)
        for index, value in enumerate(values):
            determinant_score = polynomial.polyadd(determinant_score, value * product((index,)))
        positive = polynomial.polymul(determinant_score, residual)
        negative = degrees * score_numerator
    elif method == "ML":
        raw_denominator = product(spectrum=raw_values)
        determinant_score = np.zeros(1, dtype=np.longdouble)
        for index, value in enumerate(raw_values):
            determinant_score = polynomial.polyadd(determinant_score, value * product((index,), raw_values))
        positive = polynomial.polymul(polynomial.polymul(determinant_score, residual), denominator)
        negative = degrees * polynomial.polymul(score_numerator, raw_denominator)
    else:
        assert method == "GCV.Cp"
        residual_degrees = (degrees - len(values)) * denominator
        data_residual = (deviance - float(np.sum(scores))) * polynomial.polymul(denominator, denominator)
        data_score = np.zeros(1, dtype=np.longdouble)
        for index, value in enumerate(values):
            remaining = product((index,))
            residual_degrees = polynomial.polyadd(residual_degrees, remaining)
            squared = polynomial.polymul(remaining, remaining)
            data_residual = polynomial.polyadd(data_residual, scores[index] * squared)
            data_score = polynomial.polyadd(data_score, scores[index] * value * polynomial.polymul(squared, remaining))
        positive = 2 * polynomial.polymul(data_residual, trace_numerator)
        negative = 2 * polynomial.polymul(data_score, residual_degrees)
    numerator = polynomial.polysub(positive, negative)
    magnitude = polynomial.polyadd(np.abs(positive), np.abs(negative))
    margin = numerator / np.maximum(magnitude, np.finfo(np.longdouble).tiny)
    assert np.all(numerator > 0)
    assert np.min(margin) > 1e-12
    return numerator, float(np.min(margin))


def _certified_boundary(model, expected, transform, projection, policy, criterion):
    """Recompute an indexed exact Gaussian zero-variance limit and bounds.

    These policies apply only when the global minimum is independently
    certified. Finite solver representations are checked against their exact
    variance-dependent perturbation bounds. Identified model predictions,
    their errors, total edf and the objective retain the ordinary tolerances.
    """
    assert model.family == "gaussian" and model.link == "identity"
    assert len(model.design.S) == 1 and len(model.sp) == 1
    penalty = model.design.S[0]
    eigenvalues, eigenvectors = eigh(penalty)
    rank = policy["penalty_rank"]
    assert rank == expected["smooths"][0]["rank"]
    null, penalized = eigenvectors[:, :-rank], eigenvectors[:, -rank:]
    np.testing.assert_allclose(penalty @ null, 0, atol=1e-10 * max(1, np.linalg.norm(penalty)))
    assert np.min(eigenvalues[-rank:]) > 0
    inverse_sqrt = np.diag(1 / np.sqrt(eigenvalues[-rank:]))
    information = model.X.T @ (model.weights[:, None] * model.X)
    score = model.X.T @ (model.weights * (model.y - model.offset))
    null_information = null.T @ information @ null
    inverse = pinvh(null_information, rtol=1e-12)
    beta0 = null @ inverse @ null.T @ score
    covariance_root0 = null @ inverse @ null.T
    eliminated = penalized - null @ inverse @ null.T @ information @ penalized
    perturbation = eliminated @ inverse_sqrt
    reduced_information = perturbation.T @ information @ perturbation
    values, rotation = eigh(reduced_information)
    assert np.min(values) > 0
    reduced_score = inverse_sqrt @ penalized.T @ (score - information @ beta0)
    squared_scores = (rotation.T @ reduced_score) ** 2 / values
    raw_values = np.linalg.eigvalsh(inverse_sqrt @ penalized.T @ information @ penalized @ inverse_sqrt)
    residual = model.y - model.offset - model.X @ beta0
    deviance = float(residual @ (model.weights * residual))
    edf0 = float(np.trace(covariance_root0 @ information))
    degrees = len(model.y) if model.method == "ML" else len(model.y) - edf0
    coefficients, margin = _positive_boundary_certificate(
        model.method, values, squared_scores, raw_values, deviance, degrees)
    scale0 = deviance / (len(model.y) - edf0)
    if model.method == "GCV.Cp":
        criterion0 = len(model.y) * deviance / (len(model.y) - edf0) ** 2
    else:
        criterion_scale = deviance / degrees
        criterion0 = .5 * (degrees * (1 + np.log(2 * np.pi * criterion_scale)) - np.sum(np.log(model.weights)))
        if model.method == "REML":
            positive = np.linalg.eigvalsh(null_information)
            positive = positive[positive > max(float(np.max(positive)), 1) * 1e-12]
            criterion0 += .5 * np.sum(np.log(positive))
    criterion0 += criterion - model.objective
    assert abs(criterion - criterion0) <= 1e-6
    assert abs(expected["criterion"] - criterion0) <= 1e-6

    reference_penalty = np.zeros_like(penalty)
    smooth = expected["smooths"][0]
    indices = np.arange(smooth["first_para"] - 1, smooth["last_para"])
    reference_penalty[np.ix_(indices, indices)] = np.asarray(smooth["S"][0])
    reference_penalty = transform.T @ reference_penalty @ transform
    ratio = np.sum(reference_penalty * penalty) / np.sum(penalty * penalty)
    assert ratio > 0
    np.testing.assert_allclose(reference_penalty, ratio * penalty, rtol=1e-8, atol=1e-10)
    reference_tau = 1 / (ratio * float(np.atleast_1d(expected["sp"])[0]))
    actual_tau = 1 / float(model.sp[0])
    mapped = projection @ transform
    normal_projection = mapped @ null @ null.T
    penalized_projection = mapped @ penalized @ penalized.T
    reference_inverse = np.linalg.inv(transform)
    reference_beta = reference_inverse @ np.asarray(expected["beta"])
    reference_covariance = reference_inverse @ np.asarray(expected["vcov"]) @ reference_inverse.T
    # Both finite representations preserve the same estimated nullspace.
    np.testing.assert_allclose(normal_projection @ model.beta, normal_projection @ reference_beta, rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(normal_projection @ model.cov_beta @ normal_projection.T,
                               normal_projection @ reference_covariance @ normal_projection.T, rtol=1e-4, atol=1e-12)
    reference_edf = np.sum(expected["edf"])
    actual_edf = np.trace(model._inner.inverse @ model._inner.information)
    np.testing.assert_allclose(actual_edf, reference_edf, rtol=1e-3, atol=1e-12)
    envelopes = []
    mapped_perturbation = mapped @ perturbation
    penalized_perturbation = penalized_projection @ perturbation
    for source, tau, beta, covariance, scale, edf in [
        ("Python", actual_tau, model.beta, model.cov_beta, model.scale, actual_edf),
        ("R", reference_tau, reference_beta, reference_covariance, expected["scale"], reference_edf),
    ]:
        beta_bound = tau * np.linalg.norm(penalized_perturbation, 2) * np.linalg.norm(reduced_score)
        covariance_bound = scale * tau * np.linalg.norm(penalized_perturbation, 2) * np.linalg.norm(mapped_perturbation, 2)
        beta_error = np.linalg.norm(penalized_projection @ beta)
        covariance_error = np.linalg.norm(penalized_projection @ covariance @ mapped.T, 2)
        assert beta_error <= beta_bound + 1e-12
        assert covariance_error <= covariance_bound + 1e-12
        np.testing.assert_allclose(edf, edf0, rtol=1e-3, atol=1e-12)
        envelopes.append({"source": source, "variance": tau,
                          "beta_error": float(beta_error), "beta_bound": float(beta_bound),
                          "covariance_error": float(covariance_error), "covariance_bound": float(covariance_bound)})
    return {"beta0": beta0, "covariance_root0": covariance_root0, "scale0": scale0,
            "perturbation": perturbation, "score": reduced_score,
            "information": reduced_information, "actual_tau": actual_tau,
            "reference_tau": reference_tau, "edf0": edf0,
            "reference_scale": expected["scale"],
            "limit_influence": transform @ covariance_root0 @ information @ reference_inverse,
            "mapped_perturbation": transform @ perturbation,
            "reference_information": reference_inverse.T @ information @ reference_inverse,
            "criterion_limit": float(criterion0), "positive_coefficients": coefficients.tolist(),
            "minimum_relative_margin": margin, "envelopes": envelopes}


def _boundary_edf(actual, expected, boundary):
    limit = np.diag(boundary["limit_influence"])
    identified = np.abs(limit) > 1e-12
    np.testing.assert_allclose(actual[identified], np.asarray(expected)[identified], rtol=1e-3, atol=1e-12)
    root = boundary["mapped_perturbation"]
    score_root = root.T @ boundary["reference_information"]
    norm_product = np.linalg.norm(root, axis=1) * np.linalg.norm(score_root, axis=0)
    for observed, tau in [(actual, boundary["actual_tau"]), (np.asarray(expected), boundary["reference_tau"])]:
        assert np.all(np.abs(observed[~identified] - limit[~identified]) <= tau * norm_product[~identified] + 1e-12)


def _boundary_term(model, newdata, label, observed, reference, field, boundary):
    rows = np.zeros_like(model.design.predict(newdata))
    indices = model.design.term_slices[label]
    rows[:, indices] = model.design.predict(newdata)[:, indices]
    perturbation = rows @ boundary["perturbation"]
    null_variance = np.einsum("ij,jk,ik->i", rows, boundary["covariance_root0"], rows)
    if field == "fit":
        limit = rows @ boundary["beta0"]
    else:
        limit = np.sqrt(np.maximum(0, boundary["scale0"] * null_variance))
    identified = np.abs(limit) > 1e-12
    np.testing.assert_allclose(np.asarray(observed)[identified], np.asarray(reference)[identified],
                               rtol=1e-6 if field == "fit" else 1e-4, atol=1e-12)
    for actual, tau, scale in [(observed, boundary["actual_tau"], model.scale),
                               (reference, boundary["reference_tau"], boundary["reference_scale"])]:
        if field == "fit":
            bound = tau * np.linalg.norm(perturbation, axis=1) * np.linalg.norm(boundary["score"])
        else:
            variance_bound = scale * tau * np.sum(perturbation ** 2, axis=1)
            variance_bound += abs(scale - boundary["scale0"]) * np.abs(null_variance)
            bound = np.sqrt(np.maximum(0, variance_bound))
        assert np.all(np.abs(np.asarray(actual)[~identified] - limit[~identified]) <= bound[~identified] + 1e-12)


@pytest.mark.golden
@pytest.mark.parametrize("path", FIXTURES, ids=lambda path: path.stem)
def test_gam_golden(path, record_property):
    from rparity.gam import gam

    fixture = json.loads(path.read_text())
    spec = fixture["spec"]
    observation = fixture["oracle"]
    expected = observation["result"]
    indexed_policy = POLICIES.get(path.stem, {})
    policy = (indexed_policy if indexed_policy.get("kind") == "gam_coefficient_gauge" else None)
    warning_policy = indexed_policy.get("optimizer_warning_difference")
    gauge_limit_policy = GAUGE_LIMITS.get(path.stem)
    boundary_policy = BOUNDARIES.get(path.stem)
    vanishing_policy = VANISHING.get(path.stem)
    assert "error" not in expected, expected.get("error")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = gam(spec["formula"], pd.DataFrame(spec["data"]), family=spec["family"], link=spec["link"], method=spec["args"]["method"], weights=spec.get("weights"), offset=spec.get("offset"))
    transform = _coordinate_map(model, expected)
    constraint = None
    if policy and spec["args"]["method"] == "REML":
        # REML integrates an improper coefficient prior on the reported slice.
        # Its measure changes under an oblique constraint, even though fitted
        # values and the identifiable coefficient quotient are unchanged.
        covariance = np.asarray(expected["vcov"], dtype=float)
        values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
        # A strongly penalized, identified coefficient can have a variance
        # much smaller than another coefficient. Its relative eigenvalue is
        # not evidence for an additional coefficient constraint. The joint
        # X/S nullspace independently fixes the exact constraint dimension.
        indices = np.argsort(np.abs(values))[:policy["joint_nullity"]]
        constraint = vectors[:, indices]
        np.testing.assert_allclose(covariance @ constraint, 0,
                                   atol=1e-10 * max(1, np.linalg.norm(covariance)))
        joint_null = null_space(np.asarray(expected["X"], dtype=float), rcond=1e-12)
        assert joint_null.shape[1] == policy["joint_nullity"]
        assert np.min(np.linalg.svd(constraint.T @ joint_null, compute_uv=False)) > 1e-6
    criterion = model.criterion_in_basis(transform, constraint=constraint)
    record_property("better_optimum", bool(criterion < expected["criterion"] - 1e-6))
    record_property("reference_smoothing_criterion", float(expected["criterion"]))
    record_property("python_smoothing_criterion", float(criterion))
    assert criterion <= expected["criterion"] + 1e-6
    gauge_limit = (numerical_gauge_penalty_limit(
        model, expected, transform, spec, gauge_limit_policy, policy, criterion, constraint)
        if gauge_limit_policy else None)
    if gauge_limit:
        record_property("verified_gauge_positive_penalty_limit", True)
        record_property("gauge_positive_penalty_limit_evidence",
                        json.dumps(gauge_limit, sort_keys=True))
    assert not (vanishing_policy and (policy or boundary_policy))
    vanishing = (numerical_positive_penalty_limit(
        model, expected, transform, spec, vanishing_policy, criterion)
        if vanishing_policy else None)
    vanishing_warning = ((vanishing_policy
                          and vanishing_policy["optimizer_warning_metadata_difference"])
                         or (gauge_limit_policy
                             and gauge_limit_policy["optimizer_warning_metadata_difference"]))
    if vanishing:
        record_property("verified_numerical_positive_penalty_limit", True)
        record_property("positive_penalty_limit_max_criterion_change",
                        max(abs(source["criterion_change"])
                            for source in vanishing["sources"].values()))
        measured = {
            name: {key: value for key, value in source.items()
                   if key not in {"diagonal_null", "edf_bounds", "term_limits"}}
            for name, source in vanishing["sources"].items()
        }
        record_property("positive_penalty_limit_evidence", json.dumps(measured, sort_keys=True))
    messages = " ".join(np.atleast_1d(observation["warnings"]).tolist()).lower()
    if not expected["converged"] or any(word in messages for word in ["converg", "iteration limit", "step failure"]):
        if not warning_policy and not vanishing_warning:
            assert any(issubclass(warning.category, RuntimeWarning) for warning in caught)
            return
        assert model.converged
        gradient, hessian = model.smoothing_derivatives()
        assert np.max(np.abs(gradient), initial=0) / max(1, abs(criterion)) <= 1e-7
        eigenvalues = np.linalg.eigvalsh((hessian + hessian.T) / 2)
        assert np.min(eigenvalues, initial=0) >= -1e-6 * max(1, np.max(np.abs(eigenvalues), initial=0))
        record_property("reviewed_optimizer_warning_difference", True)
    projection = (_identifiable_projection(model, expected, transform, policy)
                  if policy else np.eye(len(model.beta)))
    boundary = (_certified_boundary(model, expected, transform, projection, boundary_policy, criterion)
                if boundary_policy else None)
    if boundary:
        record_property("certified_gaussian_boundary", True)
    if not ((gauge_limit_policy and gauge_limit_policy["coefficients"])
            or (boundary_policy and boundary_policy.get("coefficients"))
            or (vanishing_policy and vanishing_policy["coefficients"])):
        np.testing.assert_allclose(projection @ transform @ model.beta,
                                   projection @ expected["beta"], rtol=1e-5, atol=1e-12)
    if not ((gauge_limit_policy and gauge_limit_policy["covariance"])
            or (boundary_policy and boundary_policy.get("covariance"))
            or (vanishing_policy and vanishing_policy["covariance"])):
        np.testing.assert_allclose(projection @ transform @ model.cov_beta @ transform.T @ projection,
                                   projection @ expected["vcov"] @ projection, rtol=1e-4, atol=1e-12)
    np.testing.assert_allclose(model.fitted(), expected["fitted"], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(model.scale, expected["scale"], rtol=1e-4, atol=1e-12)
    # Effective coefficient degrees of freedom change with the coordinate map.
    influence = model._inner.inverse @ model._inner.information
    reference_influence = transform @ influence @ np.linalg.inv(transform)
    if policy:
        np.testing.assert_allclose(np.trace(reference_influence), np.sum(expected["edf"]),
                                   rtol=1e-3, atol=1e-12)
    elif boundary_policy and boundary_policy.get("coefficient_edf"):
        _boundary_edf(np.diag(reference_influence), expected["edf"], boundary)
    elif vanishing_policy and vanishing_policy["coefficient_edf"]:
        # The helper checks each indexed vanishing EDF against its measured
        # Schur envelope and keeps identified EDF comparisons unchanged.
        identified = vanishing["shared_identified_edf"]
        np.testing.assert_allclose(np.diag(reference_influence)[identified],
                                   np.asarray(expected["edf"])[identified],
                                   rtol=1e-3, atol=1e-12)
    else:
        np.testing.assert_allclose(np.diag(reference_influence), expected["edf"], rtol=1e-3, atol=1e-12)
    predicted = model.predict(pd.DataFrame(spec["newdata"]), se_fit=True)
    np.testing.assert_allclose(predicted["fit"], expected["prediction"]["prediction"], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(predicted["se.fit"], expected["prediction"]["se_fit"], rtol=1e-4, atol=1e-12)
    term_prediction = model.predict(pd.DataFrame(spec["newdata"]), type="terms", se_fit=True)
    columns = np.atleast_1d(expected["terms"]["columns"]).tolist()
    assert set(term_prediction["fit"].columns) == set(columns)
    if policy:
        np.testing.assert_allclose(term_prediction["fit"][columns].sum(axis=1),
                                   np.sum(expected["terms"]["prediction"], axis=1), rtol=1e-6, atol=1e-12)
    else:
        for column, label in enumerate(columns):
            for field, expected_field, tolerance in [("fit", "prediction", 1e-6), ("se.fit", "se_fit", 1e-4)]:
                reference_values = np.asarray(expected["terms"][expected_field])[:, column]
                boundary_field = "term_fit" if field == "fit" else "term_se"
                if boundary_policy and label in boundary_policy.get(boundary_field, []):
                    _boundary_term(model, pd.DataFrame(spec["newdata"]), label,
                                   term_prediction[field][label], reference_values,
                                   "fit" if field == "fit" else "se", boundary)
                elif vanishing_policy and label in vanishing_policy[boundary_field]:
                    # Both finite term values and their constrained limits
                    # were independently checked by the executable helper.
                    assert label in vanishing["sources"]["Python"]["term_envelopes"]
                else:
                    np.testing.assert_allclose(term_prediction[field][label], reference_values, rtol=tolerance, atol=1e-12)
    effects = model.partial_effects(newdata=pd.DataFrame(spec["newdata"]))
    for label, effect in effects.items():
        column = columns.index(label)
        if not policy:
            for field, expected_field, tolerance in [("fit", "prediction", 1e-6), ("se", "se_fit", 1e-4)]:
                boundary_field = "term_fit" if field == "fit" else "term_se"
                reference_values = np.asarray(expected["terms"][expected_field])[:, column]
                if boundary_policy and label in boundary_policy.get(boundary_field, []):
                    _boundary_term(model, pd.DataFrame(spec["newdata"]), label,
                                   effect[field], reference_values, field, boundary)
                elif vanishing_policy and label in vanishing_policy[boundary_field]:
                    term_field = "fit" if field == "fit" else "se.fit"
                    np.testing.assert_array_equal(effect[field], term_prediction[term_field][label])
                else:
                    np.testing.assert_allclose(effect[field], reference_values, rtol=tolerance, atol=1e-12)
    smooth = model.summary_tables()["smooth"]
    reference_smooth = _table(expected["smooth_table"])
    assert set(smooth.index) == set(reference_smooth.index)
    if not policy:
        for name in ["edf", "Ref.df"]:
            for label in reference_smooth.index:
                if name == "edf" and boundary_policy and label in boundary_policy.get("smooth_edf", []):
                    for edf, tau in [(smooth.loc[label, name], boundary["actual_tau"]),
                                     (reference_smooth.loc[label, name], boundary["reference_tau"])]:
                        assert abs(edf) <= tau * np.trace(boundary["information"]) + 1e-12
                elif name == "edf" and vanishing_policy and label in vanishing_policy["smooth_edf"]:
                    for values, source in [(smooth, "Python"), (reference_smooth, "R")]:
                        bound = vanishing["sources"][source]["smooth_bounds"][label]["edf"]
                        assert abs(values.loc[label, name]) <= bound + 1e-12
                else:
                    np.testing.assert_allclose(smooth.loc[label, name], reference_smooth.loc[label, name], rtol=1e-3, atol=1e-12)
        statistic = "F" if "F" in reference_smooth else "Chi.sq"
        for label in reference_smooth.index:
            if boundary_policy and label in boundary_policy.get("smooth_statistic", []):
                for value, tau, scale in [(smooth.loc[label, statistic], boundary["actual_tau"], model.scale),
                                           (reference_smooth.loc[label, statistic], boundary["reference_tau"], expected["scale"])]:
                    assert abs(value) <= tau * np.sum(boundary["score"] ** 2) / scale + 1e-12
            elif vanishing_policy and label in vanishing_policy["smooth_statistic"]:
                for values, source in [(smooth, "Python"), (reference_smooth, "R")]:
                    bound = vanishing["sources"][source]["smooth_bounds"][label]["statistic"]
                    assert abs(values.loc[label, statistic]) <= bound + 1e-12
            else:
                np.testing.assert_allclose(smooth.loc[label, statistic], reference_smooth.loc[label, statistic], rtol=1e-4, atol=1e-12)
        np.testing.assert_allclose(smooth.loc[reference_smooth.index, "p-value"], reference_smooth["p-value"], rtol=0, atol=1e-4)
    diagnostics = model.check(k_rep=spec.get("k_rep", 400), k_sample=spec.get("k_subsample", 5000), seed=spec.get("diagnostic_seed", 1))
    reference_check = _table(expected["gam_check"]["k_check"])
    actual_check = diagnostics["k.check"].loc[reference_check.index]
    np.testing.assert_array_equal(actual_check["k'"], reference_check["k'"])
    if not policy:
        for label in reference_check.index:
            if boundary_policy and label in boundary_policy.get("smooth_edf", []):
                for edf, tau in [(actual_check.loc[label, "edf"], boundary["actual_tau"]),
                                 (reference_check.loc[label, "edf"], boundary["reference_tau"])]:
                    assert abs(edf) <= tau * np.trace(boundary["information"]) + 1e-12
            elif vanishing_policy and label in vanishing_policy["smooth_edf"]:
                for values, source in [(actual_check, "Python"), (reference_check, "R")]:
                    bound = vanishing["sources"][source]["smooth_bounds"][label]["edf"]
                    assert abs(values.loc[label, "edf"]) <= bound + 1e-12
            else:
                np.testing.assert_allclose(actual_check.loc[label, "edf"], reference_check.loc[label, "edf"], rtol=1e-3, atol=1e-12)
    np.testing.assert_allclose(actual_check["k-index"], reference_check["k-index"], rtol=1e-6, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(actual_check["p-value"], reference_check["p-value"], rtol=0, atol=1e-4, equal_nan=True)
    if warning_policy or vanishing_warning:
        assert diagnostics["converged"]
    else:
        assert diagnostics["converged"] == expected["gam_check"]["converged"]
    assert diagnostics["rank"] == expected["gam_check"]["rank"]
    assert diagnostics["model_rank"] == expected["gam_check"]["model_rank"]
    _summary_structure(model.summary(), expected["summary"])


@pytest.mark.parametrize("invalid_field", ["identified_beta", "identified_covariance", "factor_exception"])
def test_gam_limit_guard_rejects_invalid_observations(invalid_field):
    """An indexed limit cannot hide fixed-factor or identified-field errors."""
    from rparity.gam import gam

    fixture = json.loads((Path(__file__).parent / "golden/gam/gam_0006.json").read_text())
    spec = fixture["spec"]
    expected = deepcopy(fixture["oracle"]["result"])
    model = gam(spec["formula"], pd.DataFrame(spec["data"]), family=spec["family"],
                link=spec["link"], method=spec["args"]["method"],
                weights=spec.get("weights"), offset=spec.get("offset"))
    transform = _coordinate_map(model, expected)
    policy = deepcopy(VANISHING["gam_0006"])
    if invalid_field == "identified_beta":
        expected["beta"][0] += .01
    elif invalid_field == "identified_covariance":
        expected["vcov"][0][0] += .01
    else:
        policy["term_se"].append("f")
    with pytest.raises(AssertionError):
        numerical_positive_penalty_limit(model, expected, transform, spec, policy,
                                         model.criterion_in_basis(transform))


@pytest.mark.parametrize("invalid_field", [
    "identified_beta", "identified_covariance", "positive_beta", "positive_covariance",
    "fitted", "prediction", "prediction_se", "total_edf", "wrong_slice", "wrong_rank",
])
def test_gam_gauge_limit_rejects_invalid_observations(invalid_field):
    """A joint gauge cannot conceal an error in the identified quotient."""
    from rparity.gam import gam

    fixture = json.loads((Path(__file__).parent / "golden/gam/gam_0017.json").read_text())
    spec = fixture["spec"]
    expected = deepcopy(fixture["oracle"]["result"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = gam(spec["formula"], pd.DataFrame(spec["data"]), family=spec["family"],
                    link=spec["link"], method=spec["args"]["method"],
                    weights=spec.get("weights"), offset=spec.get("offset"))
    transform = _coordinate_map(model, expected)
    values, vectors = np.linalg.eigh(expected["vcov"])
    normals = vectors[:, np.argsort(abs(values))[:1]]
    criterion = model.criterion_in_basis(transform, constraint=normals)
    policy = deepcopy(GAUGE_LIMITS["gam_0017"])
    if invalid_field == "identified_beta":
        expected["beta"][0] += .01
    elif invalid_field == "identified_covariance":
        expected["vcov"][0][0] += .01
    elif invalid_field == "positive_beta":
        expected["beta"][4] += .01
    elif invalid_field == "positive_covariance":
        expected["vcov"][4][4] += .01
    elif invalid_field == "fitted":
        expected["fitted"][0] += .01
    elif invalid_field == "prediction":
        expected["prediction"]["prediction"][0] += .01
    elif invalid_field == "prediction_se":
        expected["prediction"]["se_fit"][0] += .01
    elif invalid_field == "total_edf":
        expected["edf"][0] += .01
    elif invalid_field == "wrong_slice":
        normals[:, 0] = 0
        normals[0, 0] = 1
    else:
        policy["positive_rank"] += 1
    with pytest.raises(AssertionError):
        numerical_gauge_penalty_limit(model, expected, transform, spec, policy,
                                      POLICIES["gam_0017"], criterion, normals)
