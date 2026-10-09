"""Case-indexed positive-penalty limits on an identifiable GAM quotient.

This numerical guard is confined to tests. It neither supplies reference
coordinates to the packaged fit nor relaxes identified model predictions.
"""

import numpy as np
import pandas as pd
from _gam_limits import _limit_fit, _spectrum
from scipy import linalg

from rparity.gam._fit import _Family


def numerical_gauge_penalty_limit(model, expected, transform, spec, policy,
                                  gauge_policy, criterion, constraint=None):
    """Bound vanishing penalized directions after removing an exact gauge.

    The joint X/S nullspace is removed before the boundary refit. REML keeps
    its nominal improper-prior dimension and the observed oblique-slice
    measure. For orthonormal constraint normals C and joint-null basis G,
    the slice determinant differs from the orthogonal quotient determinant
    by det(C.T @ G)**2. This factor is unchanged as the penalty tends to
    infinity, and must therefore appear in both finite and limiting criteria.
    """
    assert policy["kind"] == "gam_gauge_positive_penalty_limit"
    assert gauge_policy["kind"] == "gam_coefficient_gauge"
    assert model.converged
    # These indexed cases have one penalty and a canonical convex mean loss.
    assert len(model.design.S) == 1 and policy["collapsed_penalty_indices"] == [0]
    assert (model.family, model.link) in {
        ("binomial", "logit"), ("poisson", "log"), ("Gamma", "inverse"),
    }
    X = np.asarray(expected["X"], dtype=float)
    joint = linalg.null_space(X, rcond=1e-12)
    assert joint.shape[1] == gauge_policy["joint_nullity"]
    np.testing.assert_allclose(X @ joint, 0, atol=1e-10)
    inverse_transform = linalg.inv(transform)
    penalty = np.zeros((len(model.beta),)*2)
    smooth = expected["smooths"][0]
    assert len(expected["smooths"]) == 1 and len(smooth["S"]) == 1
    indices = np.arange(smooth["first_para"]-1, smooth["last_para"])
    penalty[np.ix_(indices, indices)] = smooth["S"][0]
    np.testing.assert_allclose(penalty @ joint, 0, atol=1e-10)
    own_joint = inverse_transform @ joint
    np.testing.assert_allclose(model.X @ own_joint, 0, atol=1e-10)
    np.testing.assert_allclose(model.design.S[0] @ own_joint, 0, atol=1e-10)
    mapped_penalty = inverse_transform.T @ model.design.S[0] @ inverse_transform
    ratio = float(np.sum(mapped_penalty*penalty)/np.sum(penalty**2))
    assert ratio > 0
    np.testing.assert_allclose(mapped_penalty, ratio*penalty, rtol=1e-8, atol=1e-10)
    _, positive, _ = _spectrum(penalty)
    assert positive.shape[1] == policy["positive_rank"] == smooth["rank"]
    null = linalg.null_space(np.vstack([positive.T, joint.T]), rcond=1e-12)
    quotient = np.column_stack([null, positive])
    np.testing.assert_allclose(quotient.T @ quotient, np.eye(quotient.shape[1]), atol=1e-10)
    np.testing.assert_allclose(penalty @ null, 0, atol=1e-10)
    assert np.linalg.matrix_rank(X @ null, tol=1e-10) == null.shape[1]
    assert null.shape[1] == policy["identified_null_dimension"]
    # A gauge is not an additional penalized direction or a deleted nominal
    # unpenalized coefficient in the public REML convention.
    nominal_null_dimension = len(model.beta)-positive.shape[1]
    measure_correction = 0.
    if model.method == "REML":
        assert constraint is not None
        normals = np.asarray(constraint, dtype=float)
        assert normals.shape == joint.shape
        np.testing.assert_allclose(normals.T @ normals, np.eye(joint.shape[1]), atol=1e-10)
        reference_covariance = np.asarray(expected["vcov"], dtype=float)
        np.testing.assert_allclose(reference_covariance @ normals, 0,
                                   atol=1e-10*max(1, np.linalg.norm(reference_covariance)))
        singular = linalg.svdvals(normals.T @ joint)
        assert np.min(singular) > 1e-6
        measure_correction = float(np.sum(np.log(singular)))
    family = _Family(model.family, model.link)
    limit = _limit_fit(X, null, model, family, np.zeros_like(penalty),
                      nominal_null_dimension)
    limit["criterion"] += measure_correction
    beta0 = limit["beta"]
    projection = null @ null.T
    quotient_projection = quotient @ quotient.T
    python_beta = transform @ model.beta
    python_covariance = transform @ model.cov_beta @ transform.T
    reference_beta = np.asarray(expected["beta"], dtype=float)
    reference_covariance = np.asarray(expected["vcov"], dtype=float)
    np.testing.assert_allclose(projection @ python_beta, projection @ reference_beta,
                               rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(projection @ python_covariance @ projection,
                               projection @ reference_covariance @ projection,
                               rtol=1e-4, atol=1e-12)
    derivative, _ = family.derivatives(model.linear_predictor, model.fitted_values)
    noise = np.sqrt(model.scale*family.variance(model.fitted_values)[0])/np.abs(derivative)
    noise_rms = float(np.sqrt(np.average(noise**2, weights=model.weights)))
    assert noise_rms > 0 and np.isfinite(noise_rms)
    newdata = pd.DataFrame(spec["newdata"])
    new_X = model.design.predict(newdata) @ inverse_transform
    np.testing.assert_allclose(new_X @ joint, 0, atol=1e-10)
    prediction = model.predict(newdata, se_fit=True)
    np.testing.assert_allclose(prediction["fit"], expected["prediction"]["prediction"],
                               rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(prediction["se.fit"], expected["prediction"]["se_fit"],
                               rtol=1e-4, atol=1e-12)
    np.testing.assert_allclose(model.fitted_values, expected["fitted"], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(np.sum(model.edf), np.sum(expected["edf"]), rtol=1e-3, atol=1e-12)
    np.testing.assert_allclose(model.scale, expected["scale"], rtol=1e-4, atol=1e-12)
    gradient, hessian = model.smoothing_derivatives()
    assert np.max(np.abs(gradient), initial=0)/max(1, abs(criterion)) <= 1e-7
    curvature = linalg.eigvalsh((hessian+hessian.T)/2)
    assert np.min(curvature, initial=0) >= -1e-6*max(1, np.max(np.abs(curvature), initial=0))
    evidence = {}
    for source, sp, beta, covariance, scale, finite_criterion in [
        ("Python", float(model.sp[0])*ratio, python_beta, python_covariance, model.scale, criterion),
        ("R", float(np.atleast_1d(expected["sp"])[0]), reference_beta,
         reference_covariance, float(expected["scale"]), float(expected["criterion"])),
    ]:
        eta = X @ beta+model.offset
        mu = family.inverse(eta)
        assert family.valid(mu)
        first, second = family.derivatives(eta, mu)
        variance, prime = family.variance(mu)
        working = model.weights*first**2/variance
        observed_weights = model.weights*(first**2/variance-(model.y-mu)*(
            second/variance-first**2*prime/variance**2))
        # Canonical loss has positive curvature throughout its convex domain,
        # and here its observed and expected information coincide exactly.
        assert np.min(observed_weights) > 0
        np.testing.assert_allclose(observed_weights, working, rtol=1e-12, atol=1e-12)
        information = X.T @ (working[:, None]*X)
        A, B = null.T @ information @ null, null.T @ information @ positive
        eliminated = positive-null @ linalg.solve(A, B, assume_a="pos")
        schur = positive.T @ information @ positive-B.T @ linalg.solve(A, B, assume_a="pos")
        assert linalg.eigvalsh((schur+schur.T)/2).min() >= -1e-10*max(1, np.linalg.norm(information, 2))
        minimum_penalty = float(linalg.eigvalsh(positive.T @ (sp*penalty) @ positive).min())
        assert minimum_penalty > 0
        first0, _ = family.derivatives(limit["inner"].eta, limit["inner"].mu)
        loss_score0 = X.T @ (model.weights*first0/family.variance(limit["inner"].mu)[0]
                            *(limit["inner"].mu-model.y))
        null_score = float(np.linalg.norm(null.T @ loss_score0, np.inf))
        assert null_score <= 1e-7*max(1, abs(finite_criterion))
        coefficient_bound = float(np.linalg.norm(positive.T @ loss_score0)/minimum_penalty)
        coefficient_error = float(np.linalg.norm(positive.T @ beta))
        assert coefficient_error <= coefficient_bound+1e-12
        covariance_root = null @ linalg.solve(A, null.T, assume_a="pos")
        covariance_bound = float(scale*np.linalg.norm(eliminated, 2)**2/minimum_penalty)
        covariance_error = float(np.linalg.norm(
            quotient_projection @ covariance @ quotient_projection-scale*covariance_root, 2))
        assert covariance_error <= covariance_bound+1e-12
        np.testing.assert_allclose(mu, limit["inner"].mu, rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(projection @ beta, projection @ beta0, rtol=1e-5, atol=1e-12)
        np.testing.assert_allclose(projection @ covariance @ projection,
                                   projection @ limit["covariance"] @ projection,
                                   rtol=1e-4, atol=1e-12)
        criterion_change = float(finite_criterion-limit["criterion"])
        assert abs(criterion_change) <= 1e-6
        data_loss_change = float(-family.selection_loglik(model.y, mu, model.weights, limit["phi"])
                                 + family.selection_loglik(model.y, limit["inner"].mu, model.weights, limit["phi"]))
        assert abs(data_loss_change) <= 1e-6
        positive_curve = max(float(np.max(np.abs(X @ positive @ (positive.T @ beta)))),
                             float(np.max(np.abs(new_X @ positive @ (positive.T @ beta)))))
        assert positive_curve <= 1e-6*noise_rms
        evidence[source] = {
            "criterion_change": criterion_change, "data_loss_change": data_loss_change,
            "minimum_positive_penalty": minimum_penalty,
            "coefficient_error": coefficient_error, "coefficient_bound": coefficient_bound,
            "covariance_error": covariance_error, "covariance_bound": covariance_bound,
            "positive_curve_relative_to_link_noise_rms": positive_curve/noise_rms,
            "null_score": null_score,
        }
    return {"sources": evidence, "joint_nullity": joint.shape[1],
            "positive_rank": positive.shape[1], "identified_null_dimension": null.shape[1],
            "nominal_unpenalized_dimension": nominal_null_dimension,
            "REML_slice_log_jacobian": measure_correction,
            "criterion_limit": limit["criterion"],
            "gradient": np.asarray(gradient).tolist(),
            "hessian_eigenvalues": curvature.tolist()}
