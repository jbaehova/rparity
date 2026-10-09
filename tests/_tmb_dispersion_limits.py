"""Independent public-R covariance references on proved dispersion limits.

These references preserve the original observation. A finite positive Hessian
at a regular point is extrapolated at three public score step sizes. Negative
binomial boundary references use a separately fitted model on the identified
Poisson stratum. The nominal original parameter count is always retained.
"""

import copy
import hashlib
import json

import numpy as np
from scipy.linalg import null_space
from scipy.stats import norm

_COMPONENTS = ("cond", "zi", "disp")


def fingerprint(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _covariance(result, component, names):
    matrix = result["vcov_components"].get(component)
    if matrix is None:
        full_names = np.atleast_1d(result["vcov_full_names"]).tolist()
        indices = [full_names.index(f"{component}~{name}") for name in names]
        matrix = np.asarray(result["vcov_full"], dtype=float)[np.ix_(indices, indices)]
    return np.atleast_2d(np.asarray(matrix, dtype=float))


def _positive_information(matrix):
    matrix = np.asarray(matrix, dtype=float)
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
    assert np.all(np.isfinite(matrix))
    np.testing.assert_allclose(matrix, matrix.T, rtol=0, atol=1e-12)
    eigenvalues = np.linalg.eigvalsh(matrix)
    assert eigenvalues[0] > np.sqrt(np.finfo(float).eps) * eigenvalues[-1]
    return eigenvalues


def _score(result, maximum=1e-7):
    diagnostic = result["covariance_diagnostics"]
    native = np.asarray(diagnostic["native_parameters"], dtype=float)
    score = np.asarray(diagnostic["score"], dtype=float)
    assert score.shape == native.shape and np.all(np.isfinite(score))
    assert np.max(np.abs(score)) <= maximum
    coefficients = np.concatenate(
        [np.atleast_1d(result["fixef"][component]).astype(float) for component in _COMPONENTS]
    )
    np.testing.assert_array_equal(native[: len(coefficients)], coefficients)
    return float(np.max(np.abs(score)))


def _wald(beta, covariance):
    assert np.all(np.diag(covariance) > 0)
    return 2 * norm.sf(np.abs(np.asarray(beta) / np.sqrt(np.diag(covariance))))


def _loglik(result):
    value = result["logLik"]
    return float(value if value is not None else -result["optimizer_diagnostics"]["objective"])


def poisson_density_bounds(counts, means, dispersions, weights, family):
    """Certified density and curvature bounds from integer-count recurrence.

    At a fixed mean mu and negative-binomial size r, log p_NB - log p_Pois
    equals sum_j log1p(j/r) - y log1p(mu/r) + r*(mu/r-log1p(mu/r)).
    An eight-term alternating series retains cancellation. Its geometric
    remainder and arithmetic forward error bound are included explicitly.
    Differentiating with respect to log dispersion multiplies each term by
    its power, giving independent first- and second-derivative bounds. A zero
    mixture contributes at most one quarter of the squared first derivative
    to curvature and is 1-Lipschitz in conditional log density.
    """
    assert family in {"nbinom1", "nbinom2", "poisson"}
    arrays = [
        np.asarray(value, dtype=np.longdouble)
        for value in [
            counts,
            means,
            dispersions,
            weights,
        ]
    ]
    y, mu, dispersion, weight = arrays
    assert y.shape == mu.shape == dispersion.shape == weight.shape
    assert np.all(np.isfinite(y)) and np.all(y >= 0) and np.all(y == np.floor(y))
    assert np.all(np.isfinite(mu)) and np.all(mu > 0)
    assert np.all(np.isfinite(dispersion)) and np.all(dispersion > 0)
    assert np.all(np.isfinite(weight)) and np.all(weight >= 0)
    if family == "poisson":
        return {"density": 0.0, "curvature": 0.0, "largest_expansion_ratio": 0.0}
    sizes = mu / dispersion if family == "nbinom1" else dispersion
    assert np.all(np.isfinite(sizes)) and np.all(sizes > 0)
    density_bound = np.longdouble(0)
    curvature_bound = np.longdouble(0)
    maximum_ratio = np.longdouble(0)
    degree = 8
    arithmetic_epsilon = np.finfo(np.longdouble).eps
    for count, mean, size, row_weight in zip(y, mu, sizes, weight, strict=True):
        a = np.arange(int(count), dtype=np.longdouble) / size
        b = mean / size
        ratio = max(np.max(a, initial=0), b)
        assert ratio < 0.1, "The Poisson-limit series must converge with resolved remainder"
        maximum_ratio = max(maximum_ratio, ratio)
        a_powers = [np.sum(a**k) for k in range(1, degree + 1)]
        b_powers = [b**k for k in range(1, degree + 1)]
        density_terms = []
        first_terms = []
        curvature_terms = []
        for k in range(1, degree + 1):
            alternating = np.longdouble((-1) ** (k + 1))
            density_terms.extend(
                [
                    alternating * a_powers[k - 1] / k,
                    -count * alternating * b_powers[k - 1] / k,
                ]
            )
            first_terms.extend(
                [
                    -alternating * a_powers[k - 1],
                    count * alternating * b_powers[k - 1],
                ]
            )
            curvature_terms.extend(
                [
                    alternating * k * a_powers[k - 1],
                    -count * alternating * k * b_powers[k - 1],
                ]
            )
            if k >= 2:
                term = size * (-alternating) * b_powers[k - 1] / k
                density_terms.append(term)
                first_terms.append((1 - k) * term)
                curvature_terms.append((1 - k) ** 2 * term)
        density_remainder = (np.sum(a ** (degree + 1)) + (count + size) * b ** (degree + 1)) / (
            degree + 1
        )
        first_remainder = np.sum(a ** (degree + 1) / (1 - a)) + (count + size) * b ** (
            degree + 1
        ) / (1 - b)

        def curvature_tail(value):
            return value ** (degree + 1) * ((degree + 1) - degree * value) / (1 - value) ** 2

        second_remainder = np.sum(curvature_tail(a)) + (count + size) * curvature_tail(b)
        operations = 128 * (degree + int(count) + 1)
        gamma = operations * arithmetic_epsilon / (1 - operations * arithmetic_epsilon)

        def bounded_sum(terms, remainder, rounding=gamma):
            terms = np.asarray(terms, dtype=np.longdouble)
            return abs(np.sum(terms)) + remainder + rounding * np.sum(np.abs(terms))

        row_density = bounded_sum(density_terms, density_remainder)
        row_first = bounded_sum(first_terms, first_remainder)
        row_curvature = bounded_sum(curvature_terms, second_remainder)
        density_bound += row_weight * row_density
        curvature_bound += row_weight * (row_curvature + row_first**2 / 4)
    inflation = 1 + 128 * np.finfo(float).eps
    return {
        "density": float(density_bound) * inflation,
        "curvature": float(curvature_bound) * inflation,
        "largest_expansion_ratio": float(maximum_ratio),
    }


def _richardson_reference(model, spec, observation, entry):
    assert entry["kind"] == "tmb_richardson_hessian_reference"
    assert fingerprint({"spec": spec, "oracle": observation}) == entry["reference_fingerprint"]
    expected = observation["result"]
    assert not observation["warnings"]
    assert expected["optimizer_diagnostics"]["convergence"] == 0
    assert expected["optimizer_diagnostics"]["pdHess"]
    assert np.max(np.abs(model.outer_gradient)) <= 1e-7
    own_eigenvalues = _positive_information(model.outer_information)
    matrices = {}
    point = None
    score_max = 0.0
    for step in (0.04, 0.08, 0.16):
        probe = entry["hessian_observations"][str(step)]
        assert fingerprint(probe) == entry["hessian_fingerprints"][str(step)]
        result = probe["result"]
        assert not probe["warnings"] and "error" not in result
        diagnostic = result["covariance_diagnostics"]
        assert diagnostic["finite_difference_step"] == step
        parameters = np.asarray(diagnostic["native_parameters"], dtype=float)
        if point is None:
            point = parameters
        else:
            np.testing.assert_array_equal(parameters, point)
        score_max = max(score_max, _score(result))
        # A step-size probe changes no reported fit field except diagnostics.
        for field in ["fixef", "VarCorr", "ranef", "logLik", "predictions", "df_resid"]:
            assert result[field] == expected[field]
        matrix = np.asarray(diagnostic["hessian"], dtype=float)
        assert matrix.shape == model.outer_information.shape
        _positive_information(matrix)
        matrices[step] = matrix
    low_information = (4 * matrices[0.04] - matrices[0.08]) / 3
    high_information = (4 * matrices[0.08] - matrices[0.16]) / 3
    _positive_information(low_information)
    eigenvalues = _positive_information(high_information)
    low_covariance = np.linalg.inv(low_information)
    high_covariance = np.linalg.inv(high_information)
    reference = copy.deepcopy(expected)
    cursor = 0
    stability = {}
    for component in _COMPONENTS:
        beta = np.atleast_1d(expected["fixef"][component]).astype(float)
        count = len(beta)
        block = high_covariance[cursor : cursor + count, cursor : cursor + count]
        low_block = low_covariance[cursor : cursor + count, cursor : cursor + count]
        cursor += count
        if not count:
            continue
        np.testing.assert_allclose(block, low_block, rtol=1e-4, atol=1e-12)
        np.testing.assert_allclose(model.vcov(component), block, rtol=1e-4, atol=1e-12)
        np.testing.assert_allclose(model.fixef()[component], beta, rtol=1e-5, atol=1e-12)
        reference["vcov_components"][component] = block.tolist()
        table = reference["coefficients"][component]
        if "Pr(>|z|)" in table.get("values", {}):
            p_values = _wald(beta, block)
            np.testing.assert_allclose(
                _wald(model.fixef()[component], model.vcov(component)),
                p_values,
                rtol=0,
                atol=1e-4,
            )
            table["values"]["Pr(>|z|)"] = p_values.tolist()
        stability[component] = float(np.max(np.abs(block - low_block)))
    reference["vcov"] = reference["vcov_components"]["cond"]
    return (
        reference,
        set(),
        {
            "reference": "Richardson extrapolated public R score Hessian at unchanged stationary point",
            "original_R_raw_covariance_preserved": True,
            "Hessian_step_sizes": [0.04, 0.08, 0.16],
            "component_consecutive_covariance_absolute_difference": stability,
            "R_information_minimum_eigenvalue": float(eigenvalues[0]),
            "Python_information_minimum_eigenvalue": float(own_eigenvalues[0]),
            "R_score_max": score_max,
            "Python_score_max": float(np.max(np.abs(model.outer_gradient))),
            "warning_or_boundary_exception": False,
        },
    )


def _fit_inputs(spec, auxiliary, collapsed, retained):
    for field in [
        "call",
        "formula",
        "link",
        "ziformula",
        "factors",
        "weights",
        "offset",
        "tmb_control",
        "score_polish",
    ]:
        assert spec.get(field) == auxiliary.get(field), f"unchanged original fit input {field}"
    for field, values in spec["data"].items():
        assert auxiliary["data"][field] == values, f"unchanged fit data {field}"
    own_args = {k: v for k, v in spec.get("args", {}).items() if k != "start"}
    aux_args = {k: v for k, v in auxiliary.get("args", {}).items() if k != "start"}
    assert own_args == aux_args
    if not retained:
        assert auxiliary["family"] == "poisson"
        assert spec["dispformula"] == auxiliary["dispformula"] == "~1"
        assert set(auxiliary["data"]) == set(spec["data"])
        return
    assert auxiliary["family"] == spec["family"]
    assert spec["dispformula"] == "~x + f"
    required = [f"lim_{level}" for level in retained] + ["lim_x", "lim_o"]
    assert set(auxiliary["data"]) == set(spec["data"]) | set(required)
    assert auxiliary["dispformula"] == "~0 + " + " + ".join(required[:-1]) + " + offset(lim_o)"
    mask = np.isin(spec["data"]["f"], collapsed)
    for level in retained:
        np.testing.assert_array_equal(
            auxiliary["data"][f"lim_{level}"],
            np.asarray(spec["data"]["f"]) == level,
        )
    np.testing.assert_array_equal(
        auxiliary["data"]["lim_x"],
        np.asarray(spec["data"]["x"]) * ~mask,
    )
    offset = np.asarray(auxiliary["data"]["lim_o"], dtype=float)
    assert np.all(offset[~mask] == 0) and np.all(offset[mask] == offset[mask][0])
    sign = -1 if spec["family"] == "nbinom1" else 1
    assert sign * offset[mask][0] >= 20


def _dispersion_chart(model, spec, collapsed, retained):
    names = model.disp_names
    if not retained:
        assert names == ["(Intercept)"] and len(model.disp_beta) == 1
        return np.zeros((0, 1)), np.eye(1)
    assert names == ["(Intercept)", "x", "fb", "fc"]
    factor = np.asarray(spec["data"]["f"])
    np.testing.assert_array_equal(
        model.disp_X,
        np.column_stack([np.ones(model.nobs), spec["data"]["x"], factor == "b", factor == "c"]),
    )
    transform = np.zeros((len(retained) + 1, len(names)))
    for i, level in enumerate(retained):
        transform[i, 0] = 1
        if level != "a":
            transform[i, names.index(f"f{level}")] = 1
    transform[-1, names.index("x")] = 1
    discarded = null_space(transform)
    assert discarded.shape[1] == len(collapsed)
    row_mask = np.isin(spec["data"]["f"], collapsed)
    assert np.linalg.norm(model.disp_X[~row_mask] @ discarded) <= 1e-12
    return transform, discarded


def _identified_information(model, transform, discarded):
    p = len(model.beta) + len(model.zi_beta)
    d = len(model.disp_beta)
    n = len(model.parameters)
    information = np.asarray(model.outer_information, dtype=float)
    assert information.shape == (n, n) and np.all(np.isfinite(information))
    np.testing.assert_allclose(information, information.T, rtol=0, atol=1e-12)
    weak = np.zeros((n, discarded.shape[1]))
    weak[p : p + d] = discarded
    relative_bound = np.sqrt(np.finfo(float).eps)
    norm_bound = max(1, np.linalg.norm(information, 2))
    assert np.linalg.norm(information @ weak, 2) <= relative_bound * norm_bound
    assert np.max(np.abs(model.outer_gradient)) <= 1e-7
    retained = len(transform)
    lift = np.zeros((n, n - d + retained))
    lift[:p, :p] = np.eye(p)
    lift[p : p + d, p : p + retained] = np.linalg.pinv(transform)
    lift[p + d :, p + retained :] = np.eye(n - p - d)
    tangent_information = lift.T @ information @ lift
    eigenvalues = _positive_information(tangent_information)
    tangent_covariance = np.linalg.inv(tangent_information)
    block = tangent_covariance[p : p + retained, p : p + retained]
    return block, {
        "discarded_dispersion_dimension": discarded.shape[1],
        "Python_weak_dispersion_information_norm": float(np.linalg.norm(information @ weak, 2)),
        "Python_information_norm": float(norm_bound),
        "Python_retained_information_minimum_eigenvalue": float(eigenvalues[0]),
        "Python_score_max": float(np.max(np.abs(model.outer_gradient))),
    }


def _boundary_reference(model, spec, observation, entry, caught):
    assert entry["kind"] == "tmb_independent_poisson_limit_reference"
    assert fingerprint({"spec": spec, "oracle": observation}) == entry["reference_fingerprint"]
    auxiliary = entry["auxiliary"]
    assert fingerprint(auxiliary) == entry["auxiliary_fingerprint"]
    original = observation["result"]
    result = auxiliary["oracle"]["result"]
    assert "error" not in result and not auxiliary["oracle"]["warnings"]
    assert result["optimizer_diagnostics"]["convergence"] == 0
    assert result["optimizer_diagnostics"]["pdHess"]
    reference_score = _score(result)
    reference_eigenvalues = _positive_information(result["covariance_diagnostics"]["hessian"])
    assert any(
        issubclass(warning.category, RuntimeWarning)
        and (
            "weakly identified" in str(warning.message)
            or "Poisson boundary" in str(warning.message)
        )
        for warning in caught
    ), "A proved dispersion limit must have a genuine measured Python dispersion warning"
    assert spec["family"] == model.family and model.family in {"nbinom1", "nbinom2"}
    assert model.link == "log"
    np.testing.assert_array_equal(model.y, spec["data"]["y"])
    expected_weights = spec.get("weights")
    if isinstance(expected_weights, str):
        expected_weights = spec["data"][expected_weights]
    if expected_weights is None:
        expected_weights = np.ones(model.nobs)
    np.testing.assert_array_equal(model.weights, expected_weights)
    collapsed = entry["collapsed_levels"]
    retained = entry["retained_levels"]
    assert sorted(collapsed + retained) == sorted(set(spec["data"]["f"]))
    assert set(collapsed).isdisjoint(retained)
    _fit_inputs(spec, auxiliary["spec"], collapsed, retained)
    transform, discarded = _dispersion_chart(model, spec, collapsed, retained)
    np.testing.assert_array_equal(
        transform,
        np.asarray(entry["dispersion_quotient_transform"]).reshape(transform.shape),
    )
    tangent_covariance, own_diagnostics = _identified_information(model, transform, discarded)
    mask = np.isin(spec["data"]["f"], collapsed)
    bounds = {}
    for label, mean, dispersion, family in [
        ("Python", model.predict(type="cond"), model.predict(type="disp"), model.family),
        (
            "original_R",
            original["predictions"]["cond"],
            original["predictions"]["disp"],
            model.family,
        ),
        (
            "independent_R",
            result["predictions"]["cond"],
            result["predictions"]["disp"],
            auxiliary["spec"]["family"],
        ),
    ]:
        bounds[label] = poisson_density_bounds(
            model.y[mask],
            np.asarray(mean)[mask],
            np.asarray(dispersion)[mask],
            model.weights[mask],
            family,
        )
        if label != "original_R" or not entry["better_optimum_reference"]:
            assert bounds[label]["density"] <= 1e-6, f"{label} collapsed density bound"
            assert bounds[label]["curvature"] <= 1e-6, f"{label} collapsed curvature bound"
    original_loglik = _loglik(original)
    reference_loglik = _loglik(result)
    assert abs(model.logLik() - reference_loglik) <= 1e-6
    if entry["better_optimum_reference"]:
        assert reference_loglik > original_loglik + 1e-6
        assert model.logLik() > original_loglik + 1e-6
    else:
        assert abs(reference_loglik - original_loglik) <= 1e-6
    # Start from the raw original structure, retaining its family, formulas,
    # section labels, nominal parameter count, df and boundary coordinates.
    reference = copy.deepcopy(original)
    changed = []
    for component in ("cond", "zi"):
        beta = np.atleast_1d(result["fixef"][component]).astype(float)
        assert len(model.fixef()[component]) == len(beta)
        if not len(beta):
            assert not len(np.atleast_1d(original["fixef"][component]))
            continue
        names = np.atleast_1d(result["fixef_names"][component]).tolist()
        assert names == np.atleast_1d(original["fixef_names"][component]).tolist()
        assert list(model.fixef()[component].index) == names
        np.testing.assert_allclose(model.fixef()[component], beta, rtol=1e-5, atol=1e-12)
        covariance = _covariance(result, component, names)
        np.testing.assert_allclose(model.vcov(component), covariance, rtol=1e-4, atol=1e-12)
        reference["vcov_components"][component] = covariance.tolist()
        changed.append(f"{component}_vcov")
        values = reference["coefficients"][component].get("values", {})
        if "Pr(>|z|)" in values:
            expected_p = np.asarray(
                result["coefficients"][component]["values"]["Pr(>|z|)"], dtype=float
            )
            np.testing.assert_allclose(
                _wald(model.fixef()[component], model.vcov(component)),
                expected_p,
                rtol=0,
                atol=1e-4,
            )
            values["Pr(>|z|)"] = expected_p.tolist()
            changed.append(f"{component}_wald_p")
        if entry["better_optimum_reference"]:
            reference["fixef"][component] = result["fixef"][component]
            reference["coefficients"][component] = copy.deepcopy(result["coefficients"][component])
            changed.append(f"{component}_beta")
        else:
            np.testing.assert_allclose(
                model.fixef()[component], original["fixef"][component], rtol=1e-5, atol=1e-12
            )
    dispersion_diagnostics = {}
    if retained:
        beta = transform @ model.disp_beta
        expected_beta = np.atleast_1d(result["fixef"]["disp"]).astype(float)
        names = [f"lim_{level}" for level in retained] + ["lim_x"]
        assert np.atleast_1d(result["fixef_names"]["disp"]).tolist() == names
        np.testing.assert_allclose(beta, expected_beta, rtol=1e-5, atol=1e-12)
        expected_covariance = _covariance(result, "disp", names)
        raw_covariance = transform @ model.vcov("disp").to_numpy() @ transform.T
        covariance = (
            tangent_covariance if entry["identified_tangent_covariance"] else raw_covariance
        )
        np.testing.assert_allclose(covariance, expected_covariance, rtol=1e-4, atol=1e-12)
        np.testing.assert_allclose(
            _wald(beta, covariance), _wald(expected_beta, expected_covariance), rtol=0, atol=1e-4
        )
        np.testing.assert_allclose(
            model.predict(type="disp")[~mask],
            np.asarray(result["predictions"]["disp"])[~mask],
            rtol=1e-6,
            atol=1e-12,
        )
        dispersion_diagnostics = {
            "identified_coefficients": beta.tolist(),
            "identified_covariance": covariance.tolist(),
            "independent_R_identified_covariance": expected_covariance.tolist(),
            "reported_raw_quotient_covariance": raw_covariance.tolist(),
            "covariance_reference": "inverse actual Python information on identified tangent"
            if entry["identified_tangent_covariance"]
            else "reported Python covariance projected to identified quotient",
            "original_raw_dispersion_covariance_parity_claimed": False,
        }
    for kind in ["response", "cond", "zprob"]:
        np.testing.assert_allclose(
            model.predict(type=kind), result["predictions"][kind], rtol=1e-6, atol=1e-12
        )
        if entry["better_optimum_reference"]:
            reference["predictions"][kind] = result["predictions"][kind]
            changed.append(f"{kind}_prediction")
        else:
            np.testing.assert_allclose(
                model.predict(type=kind), original["predictions"][kind], rtol=1e-6, atol=1e-12
            )
    for component, groups in result["VarCorr"].items():
        assert set(groups or {}) == set(original["VarCorr"][component] or {})
        for group, random in (groups or {}).items():
            names = np.atleast_1d(random["names"]).tolist()
            levels = np.atleast_1d(result["ranef"][component][group]["levels"]).tolist()
            assert names == np.atleast_1d(original["VarCorr"][component][group]["names"]).tolist()
            assert levels == np.atleast_1d(original["ranef"][component][group]["levels"]).tolist()
            np.testing.assert_allclose(
                model.VarCorr()[component][group], random["covariance"], rtol=1e-4, atol=1e-12
            )
            np.testing.assert_allclose(
                model.ranef()[component][group].loc[levels, names],
                result["ranef"][component][group]["values"],
                rtol=1e-4,
                atol=1e-12,
            )
            if entry["better_optimum_reference"]:
                reference["VarCorr"][component][group] = copy.deepcopy(random)
                reference["ranef"][component][group] = copy.deepcopy(
                    result["ranef"][component][group]
                )
                changed.extend([f"{group}_random_covariance", f"{group}_random_modes"])
            else:
                np.testing.assert_allclose(
                    model.VarCorr()[component][group],
                    original["VarCorr"][component][group]["covariance"],
                    rtol=1e-4,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    model.ranef()[component][group].loc[levels, names],
                    original["ranef"][component][group]["values"],
                    rtol=1e-4,
                    atol=1e-12,
                )
    nominal_count = model.nobs - original["df_resid"]
    assert nominal_count == len(model.parameters)
    np.testing.assert_allclose(
        original["AIC"], -2 * original_loglik + 2 * nominal_count, rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(
        original["BIC"],
        -2 * original_loglik + np.log(model.nobs) * nominal_count,
        rtol=0,
        atol=1e-10,
    )
    if entry["better_optimum_reference"]:
        reference["AIC"] = -2 * reference_loglik + 2 * nominal_count
        reference["BIC"] = -2 * reference_loglik + np.log(model.nobs) * nominal_count
        changed.extend(["AIC", "BIC"])
    np.testing.assert_allclose(model.AIC(), reference["AIC"], rtol=0, atol=2e-6)
    np.testing.assert_allclose(model.BIC(), reference["BIC"], rtol=0, atol=2e-6)
    assert reference["df_resid"] == model.df_resid == original["df_resid"]
    reference["vcov"] = reference["vcov_components"]["cond"]
    fields = {"disp_beta", "disp_vcov", "disp_wald_p", "disp_prediction", "sigma"}
    return (
        reference,
        fields,
        {
            "reference": "independently fitted public R negative-binomial Poisson stratum",
            "original_raw_observation_preserved": True,
            "collapsed_levels": collapsed,
            "retained_levels": retained,
            "collapsed_density_and_curvature_bounds": bounds,
            "identified_dispersion": dispersion_diagnostics,
            "better_optimum_reference": entry["better_optimum_reference"],
            "original_log_likelihood": original_loglik,
            "independent_R_log_likelihood": reference_loglik,
            "Python_log_likelihood": float(model.logLik()),
            "original_numeric_AIC_BIC_matched": not entry["better_optimum_reference"],
            "original_AIC": original["AIC"],
            "derived_reference_AIC": reference["AIC"],
            "original_BIC": original["BIC"],
            "derived_reference_BIC": reference["BIC"],
            "nominal_original_parameter_count": nominal_count,
            "original_df_resid": original["df_resid"],
            "independent_R_parameter_count": model.nobs - result["df_resid"],
            "changed_reference_fields": changed,
            "R_score_max": reference_score,
            "R_information_minimum_eigenvalue": float(reference_eigenvalues[0]),
            **own_diagnostics,
        },
    )


def dispersion_reference(model, spec, observation, entry, caught):
    """Return a guarded expected result, exact handled-field set and proof.

    The caller continues all its ordinary goldentest comparisons against the
    returned result, skipping only the five collapsed-dispersion fields that
    this helper has checked through identified charts and density bounds.
    A regular Richardson reference skips no field and needs no warning.
    """
    if entry["kind"] == "tmb_richardson_hessian_reference":
        return _richardson_reference(model, spec, observation, entry)
    return _boundary_reference(model, spec, observation, entry, caught)
