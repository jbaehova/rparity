"""Recomputed, case-indexed numerical limits for vanishing GAM directions.

This test-only module never changes model fitting. Its convexity and Schur
bounds use the observed data and both finite solutions on every invocation.
The indexed policy identifies fields, not cached expected passing values.
"""

import numpy as np
import pandas as pd
from scipy import linalg, optimize, special

from rparity.gam._fit import _Family, _irls


def _spectrum(matrix, tolerance=1e-10):
    values, vectors = linalg.eigh((matrix + matrix.T) / 2)
    positive = values > max(float(np.max(values)), 1) * tolerance
    return values[positive], vectors[:, positive], vectors[:, ~positive]


def _limit_fit(X, null, model, family, remainder, unpenalized_dimension):
    """Refit the unchanged loss on the constrained coefficient subspace."""
    reduced_penalty = null.T @ remainder @ null
    values, positive, _ = _spectrum(reduced_penalty, 1e-12)
    root = np.sqrt(values)[:, None] * positive.T
    inner = _irls(X @ null, model.y, model.weights, model.offset, family,
                  reduced_penalty, 1000, 1e-12, root)
    assert inner.converged
    beta = null @ inner.beta
    inverse = null @ inner.inverse @ null.T
    derivative, _ = family.derivatives(inner.eta, inner.mu)
    information = X.T @ ((model.weights * derivative**2
                         / family.variance(inner.mu)[0])[:, None] * X)
    influence = inverse @ information
    edf = float(np.trace(influence))
    if family.name in {"binomial", "poisson"}:
        phi, scale = 1., 1.
    else:
        variance, prime = family.variance(inner.mu)
        pearson = float(np.sum(model.weights * (model.y-inner.mu)**2 / variance))
        pearson /= len(model.y) - edf
        scale = pearson / (1 + float(np.mean(prime * (model.y-inner.mu) / variance)))
        if model.method == "GCV.Cp":
            phi = scale
        elif family.name == "gaussian":
            degrees = len(model.y)
            if model.method == "REML":
                degrees -= unpenalized_dimension
            phi = (inner.deviance + inner.penalty) / degrees
        else:
            def dispersion_score(log_phi):
                trial = np.exp(log_phi)
                shape = model.weights / trial
                correction = unpenalized_dimension if model.method == "REML" else 0
                return float(np.sum(model.weights * (special.digamma(shape)-np.log(shape)))
                             + (inner.deviance+inner.penalty)/2 + correction*trial/2)

            phi = float(np.exp(optimize.brentq(dispersion_score, -25, 10, xtol=1e-13)))
    if model.method == "GCV.Cp":
        if family.name in {"binomial", "poisson"}:
            criterion = inner.deviance / len(model.y) - 1 + 2*edf / len(model.y)
        else:
            criterion = len(model.y)*inner.deviance/(len(model.y)-edf)**2
    else:
        criterion = -family.selection_loglik(model.y, inner.mu, model.weights, phi)
        criterion += inner.penalty / (2*phi)
        observed = inner.observed + reduced_penalty
        if model.method == "ML":
            sign, determinant = (np.linalg.slogdet(positive.T @ observed @ positive)
                                 if len(values) else (1, 0))
            correction = 0
        else:
            sign, determinant = np.linalg.slogdet(observed)
            correction = unpenalized_dimension * np.log(2*np.pi*phi)
        assert sign > 0
        criterion += .5*(determinant-np.sum(np.log(values))-correction)
    return {"inner": inner, "beta": beta, "covariance": scale*inverse,
            "scale": scale, "criterion": float(criterion), "phi": phi}


def numerical_positive_penalty_limit(model, expected, transform, spec, policy, criterion):
    """Verify both finite fits against the same indexed vanishing directions.

    Only the positive penalty subspace may receive the listed exceptions.
    Its orthogonal complement retains every original relative tolerance.
    This proves a numerical perturbation bound, not global boundary optimality.
    """
    assert policy["kind"] == "gam_numerical_positive_penalty_limit"
    assert (model.family, model.link) in {
        ("gaussian", "identity"), ("binomial", "logit"),
        ("poisson", "log"), ("Gamma", "inverse"), ("Gamma", "log"),
    }
    assert model.converged
    inverse_transform = np.linalg.inv(transform)
    X = np.asarray(expected["X"], dtype=float)
    reference_penalties, labels = [], []
    for smooth in expected["smooths"]:
        indices = np.arange(smooth["first_para"]-1, smooth["last_para"])
        for local in smooth["S"]:
            penalty = np.zeros((len(model.beta),)*2)
            penalty[np.ix_(indices, indices)] = local
            reference_penalties.append(penalty)
            labels.append(smooth["label"])
    assert len(model.design.S) == len(reference_penalties)
    ratios = []
    for own, reference in zip(model.design.S, reference_penalties):
        mapped = inverse_transform.T @ own @ inverse_transform
        ratio = float(np.sum(mapped*reference)/np.sum(reference**2))
        assert ratio > 0
        np.testing.assert_allclose(mapped, ratio*reference, rtol=1e-8, atol=1e-10)
        ratios.append(ratio)
    python_sp = np.asarray(model.sp)*ratios
    reference_sp = np.atleast_1d(expected["sp"])
    collapsed = policy["collapsed_penalty_indices"]
    assert collapsed and len(set(collapsed)) == len(collapsed)
    assert all(0 <= index < len(reference_penalties) for index in collapsed)
    collapsed_labels = {labels[index] for index in collapsed}
    for field in ["term_fit", "term_se", "smooth_edf", "smooth_statistic"]:
        assert set(policy[field]) <= collapsed_labels
    base = sum((reference_penalties[index]/np.linalg.norm(reference_penalties[index], 2)
                for index in collapsed), np.zeros_like(X.T @ X))
    _, positive, null = _spectrum(base)
    assert positive.shape[1] == policy["positive_rank"]
    np.testing.assert_allclose(base @ null, 0, atol=1e-10)
    all_penalties = sum(reference_penalties, np.zeros_like(base))
    _, _, unpenalized = _spectrum(all_penalties)
    assert np.linalg.matrix_rank(np.vstack([X @ null, all_penalties @ null]),
                                tol=1e-10) == null.shape[1]
    projection = null @ null.T
    python_beta, reference_beta = transform @ model.beta, np.asarray(expected["beta"])
    python_covariance = transform @ model.cov_beta @ transform.T
    reference_covariance = np.asarray(expected["vcov"])
    np.testing.assert_allclose(projection @ python_beta, projection @ reference_beta,
                               rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(projection @ python_covariance @ projection,
                               projection @ reference_covariance @ projection,
                               rtol=1e-4, atol=1e-12)
    python_influence = transform @ (model._inner.inverse @ model._inner.information)
    python_influence = python_influence @ inverse_transform
    np.testing.assert_allclose(np.trace(python_influence), np.sum(expected["edf"]),
                               rtol=1e-3, atol=1e-12)
    gradient, hessian = model.smoothing_derivatives()
    assert np.max(np.abs(gradient), initial=0)/max(1, abs(criterion)) <= 1e-7
    curvature = np.linalg.eigvalsh((hessian+hessian.T)/2)
    assert np.min(curvature, initial=0) >= -1e-6*max(1, np.max(np.abs(curvature), initial=0))
    family = _Family(model.family, model.link)
    newdata = pd.DataFrame(spec["newdata"])
    new_X = model.design.predict(newdata) @ inverse_transform
    derivative, _ = family.derivatives(model.linear_predictor, model.fitted_values)
    noise = np.sqrt(model.scale*family.variance(model.fitted_values)[0])/np.abs(derivative)
    noise_rms = float(np.sqrt(np.average(noise**2, weights=model.weights)))
    assert noise_rms > 0 and np.isfinite(noise_rms)
    columns = np.atleast_1d(expected["terms"]["columns"]).tolist()
    terms = model.predict(newdata, type="terms", se_fit=True)
    smooths = model.summary_tables()["smooth"]
    statistic = "F" if "F" in expected["smooth_table"]["values"] else "Chi.sq"
    sources = [
        ("Python", python_sp, python_beta, python_covariance, model.scale, criterion),
        ("R", reference_sp, reference_beta, reference_covariance,
         expected["scale"], expected["criterion"]),
    ]
    evidence = {}
    for source, sp, beta, covariance, scale, finite_criterion in sources:
        large = sum((sp[index]*reference_penalties[index] for index in collapsed),
                    np.zeros_like(base))
        remainder = sum((sp[index]*reference_penalties[index]
                         for index in range(len(sp)) if index not in collapsed),
                        np.zeros_like(base))
        limit = _limit_fit(X, null, model, family, remainder, unpenalized.shape[1])
        inner, beta0 = limit["inner"], limit["beta"]
        eta = X @ beta + model.offset
        mu = family.inverse(eta)
        assert family.valid(mu)
        first, _ = family.derivatives(eta, mu)
        working = model.weights*first**2/family.variance(mu)[0]
        information = X.T @ (working[:, None]*X)
        H = information + remainder
        A, B = null.T @ H @ null, null.T @ H @ positive
        eliminated = positive - null @ np.linalg.solve(A, B)
        schur = positive.T @ H @ positive - B.T @ np.linalg.solve(A, B)
        assert np.min(np.linalg.eigvalsh((schur+schur.T)/2)) >= -1e-10*max(1, np.linalg.norm(H, 2))
        minimum_penalty = float(np.linalg.eigvalsh(positive.T @ large @ positive).min())
        assert minimum_penalty > 0
        first0, _ = family.derivatives(inner.eta, inner.mu)
        loss_gradient0 = X.T @ (model.weights*first0/family.variance(inner.mu)[0]
                                *(inner.mu-model.y)) + remainder @ beta0
        null_score = float(np.linalg.norm(null.T @ loss_gradient0, np.inf))
        assert null_score <= 1e-7*max(1, abs(finite_criterion))
        coefficient_bound = float(np.linalg.norm(positive.T @ loss_gradient0)/minimum_penalty)
        coefficient_error = float(np.linalg.norm(positive.T @ beta))
        assert coefficient_error <= coefficient_bound + 1e-12
        covariance_root = null @ np.linalg.inv(A) @ null.T
        covariance_bound = float(scale*np.linalg.norm(eliminated, 2)**2/minimum_penalty)
        covariance_error = float(np.linalg.norm(covariance-scale*covariance_root, 2))
        assert covariance_error <= covariance_bound + 1e-12
        np.testing.assert_allclose(mu, inner.mu, rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(projection @ beta, projection @ beta0,
                                   rtol=1e-5, atol=1e-12)
        np.testing.assert_allclose(projection @ covariance @ projection,
                                   projection @ limit["covariance"] @ projection,
                                   rtol=1e-4, atol=1e-12)
        criterion_change = float(finite_criterion-limit["criterion"])
        assert abs(criterion_change) <= 1e-6
        data_loss_change = float(-family.selection_loglik(model.y, mu, model.weights, limit["phi"])
                                 + family.selection_loglik(model.y, inner.mu, model.weights, limit["phi"]))
        assert abs(data_loss_change) <= 1e-6
        positive_curve = max(float(np.max(np.abs(X @ positive @ (positive.T @ beta)))),
                             float(np.max(np.abs(new_X @ positive @ (positive.T @ beta)))))
        assert positive_curve <= 1e-6*noise_rms
        diagonal_null = np.diag(covariance_root @ information)
        edf_bounds = (np.linalg.norm(eliminated, axis=1)
                      * np.linalg.norm(eliminated.T @ information, axis=0)/minimum_penalty)
        observed_edf = np.diag(python_influence) if source == "Python" else np.asarray(expected["edf"])
        vanishing_edf = policy["vanishing_coefficient_edf_indices"]
        assert np.all(np.abs(diagonal_null[vanishing_edf]) <= 1e-12)
        assert np.all(np.abs(observed_edf[vanishing_edf]) <= edf_bounds[vanishing_edf]+1e-12)
        term_limits, term_envelopes = {}, {}
        for index, label in enumerate(columns):
            if label not in collapsed_labels:
                continue
            own_indices = np.asarray(model.design.term_slices[label])
            reference_indices = np.flatnonzero(np.any(np.abs(transform[:, own_indices]) > 1e-10, axis=1))
            rows = np.zeros_like(new_X)
            rows[:, reference_indices] = new_X[:, reference_indices]
            fit0 = rows @ beta0
            null_shift = np.abs(rows @ null @ (null.T @ (beta-beta0)))
            fit_bound = np.linalg.norm(rows @ positive, axis=1)*coefficient_bound + null_shift
            var0 = np.maximum(0, np.einsum("ij,jk,ik->i", rows, limit["covariance"], rows))
            null_variance = scale*np.einsum("ij,jk,ik->i", rows, covariance_root, rows)
            variance_bound = scale*np.sum((rows @ eliminated)**2, axis=1)/minimum_penalty
            variance_bound += np.abs(null_variance-var0)
            se0, se_bound = np.sqrt(var0), np.sqrt(np.maximum(0, variance_bound))
            observed_fit = (np.asarray(terms["fit"][label]) if source == "Python"
                            else np.asarray(expected["terms"]["prediction"])[:, index])
            observed_se = (np.asarray(terms["se.fit"][label]) if source == "Python"
                           else np.asarray(expected["terms"]["se_fit"])[:, index])
            assert np.all(np.abs(observed_fit-fit0) <= fit_bound+1e-12)
            assert np.all(np.abs(observed_se-se0) <= se_bound+1e-12)
            term_limits[label] = {"fit": fit0, "se": se0}
            term_envelopes[label] = {
                "fit_error": float(np.max(np.abs(observed_fit-fit0))),
                "fit_bound": float(np.max(fit_bound)),
                "se_error": float(np.max(np.abs(observed_se-se0))),
                "se_bound": float(np.max(se_bound)),
            }
        smooth_bounds = {}
        for smooth in expected["smooths"]:
            label = smooth["label"]
            if label not in set(policy["smooth_edf"] + policy["smooth_statistic"]):
                continue
            indices = np.arange(smooth["first_para"]-1, smooth["last_para"])
            assert np.linalg.norm(null[indices], 2) <= 1e-10
            if source == "Python":
                row = smooths.loc[label]
            else:
                row_index = np.atleast_1d(expected["smooth_table"]["rows"]).tolist().index(label)
                row = {key: np.atleast_1d(value)[row_index]
                       for key, value in expected["smooth_table"]["values"].items()}
            smooth_edf_bound = float(np.sum(edf_bounds[indices]))
            statistic_bound = float(np.linalg.norm(H+large, 2)*coefficient_bound**2/scale)
            if statistic == "F":
                statistic_bound /= float(row["Ref.df"])
            if label in policy["smooth_edf"]:
                assert abs(float(row["edf"])) <= smooth_edf_bound+1e-12
            if label in policy["smooth_statistic"]:
                assert abs(float(row[statistic])) <= statistic_bound+1e-12
            smooth_bounds[label] = {"edf": smooth_edf_bound, "statistic": statistic_bound}
        evidence[source] = {
            "criterion_change": criterion_change, "data_loss_change": data_loss_change,
            "minimum_positive_penalty": minimum_penalty,
            "positive_curve_relative_to_link_noise_rms": positive_curve/noise_rms,
            "coefficient_error": coefficient_error, "coefficient_bound": coefficient_bound,
            "covariance_error": covariance_error, "covariance_bound": covariance_bound,
            "null_score": null_score, "diagonal_null": diagonal_null,
            "edf_bounds": edf_bounds,
            "vanishing_edf_observed": observed_edf[vanishing_edf].tolist(),
            "vanishing_edf_bounds": edf_bounds[vanishing_edf].tolist(),
            "term_limits": term_limits,
            "term_envelopes": term_envelopes, "smooth_bounds": smooth_bounds,
        }
    shared_identified = ((np.abs(evidence["Python"]["diagonal_null"]) > 1e-12)
                         | (np.abs(evidence["R"]["diagonal_null"]) > 1e-12))
    assert np.flatnonzero(~shared_identified).tolist() == policy["vanishing_coefficient_edf_indices"]
    np.testing.assert_allclose(np.diag(python_influence)[shared_identified],
                               np.asarray(expected["edf"])[shared_identified],
                               rtol=1e-3, atol=1e-12)
    for label in collapsed_labels:
        for field, tolerance in [("fit", 1e-6), ("se", 1e-4)]:
            np.testing.assert_allclose(evidence["Python"]["term_limits"][label][field],
                                       evidence["R"]["term_limits"][label][field],
                                       rtol=tolerance, atol=1e-12)
    return {"sources": evidence, "shared_identified_edf": shared_identified,
            "python_influence": python_influence}
