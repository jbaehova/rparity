"""Independent arithmetic and hostile-input checks for numerical references."""

import copy
import json
import warnings
from decimal import Decimal, localcontext
from pathlib import Path

import pandas as pd
import pytest
from _tmb_dispersion_limits import dispersion_reference, fingerprint, poisson_density_bounds
from _tmb_rank_one_limits import rank_one_reference

from rparity import glmmTMB

GOLDEN = Path(__file__).parent / "golden"


@pytest.fixture(scope="module")
def examples():
    """Reuse actual fits while each attack receives an independent copy."""
    result = {}
    for case in ["tmb_0848", "tmb_0367", "tmb_1047"]:
        fixture = json.loads((GOLDEN / "tmb" / f"{case}.json").read_text())
        spec = fixture["spec"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = glmmTMB(
                spec["formula"], pd.DataFrame(spec["data"]), family=spec["family"],
                link=spec["link"], ziformula=spec["ziformula"],
                dispformula=spec["dispformula"], weights=spec.get("weights"),
                offset=spec.get("offset"),
            )
        result[case] = model, spec, fixture["oracle"], caught
    return result


@pytest.mark.parametrize("mutation", [
    "changed_data", "missing_warning", "nonstationary_auxiliary",
    "indefinite_auxiliary", "identified_beta", "identified_covariance",
])
def test_dispersion_limit_rejects_invalid_identified_observations(examples, mutation):
    model, spec, observation, caught = copy.deepcopy(examples["tmb_0848"])
    entry = json.loads((GOLDEN / "tmb_dispersion_reference.json").read_text())["tmb_0848"]
    auxiliary = entry["auxiliary"]
    if mutation == "changed_data":
        auxiliary["spec"]["data"]["y"][0] += 1
    elif mutation == "missing_warning":
        caught = []
    elif mutation == "nonstationary_auxiliary":
        auxiliary["oracle"]["result"]["covariance_diagnostics"]["score"][0] = .01
    elif mutation == "indefinite_auxiliary":
        auxiliary["oracle"]["result"]["covariance_diagnostics"]["hessian"][0][0] = -1000
    elif mutation == "identified_beta":
        model.beta[0] += .01
    else:
        model.joint_cov[0, 0] *= 2
    # Resealing cannot make a mathematically invalid observation eligible.
    entry["auxiliary_fingerprint"] = fingerprint(auxiliary)
    with pytest.raises(AssertionError):
        dispersion_reference(model, spec, observation, entry, caught)


@pytest.mark.parametrize("mutation", [
    "unstable_hessian", "wrong_step", "nonstationary_point", "different_parameters",
])
def test_regular_information_reference_rejects_invalid_refinement(examples, mutation):
    model, spec, observation, caught = copy.deepcopy(examples["tmb_0367"])
    entry = json.loads((GOLDEN / "tmb_dispersion_reference.json").read_text())["tmb_0367"]
    probe = entry["hessian_observations"]["0.04"]
    diagnostics = probe["result"]["covariance_diagnostics"]
    if mutation == "unstable_hessian":
        diagnostics["hessian"][0][0] += 1
    elif mutation == "wrong_step":
        diagnostics["finite_difference_step"] = .02
    elif mutation == "nonstationary_point":
        diagnostics["score"][0] = .01
    else:
        diagnostics["native_parameters"][-1] += .01
    entry["hessian_fingerprints"]["0.04"] = fingerprint(probe)
    with pytest.raises(AssertionError):
        dispersion_reference(model, spec, observation, entry, caught)


@pytest.mark.parametrize("mutation", ["unstable_hessian", "different_parameters"])
def test_rank_one_information_reference_rejects_invalid_refinement(examples, mutation):
    model, spec, observation, caught = copy.deepcopy(examples["tmb_1047"])
    auxiliary = json.loads((GOLDEN / "tmb_rank_one_reference.json").read_text())["tmb_1047"]
    policy = json.loads((GOLDEN / "tmb_case_policy.json").read_text())["tmb_1047"]
    probe = auxiliary["regular_hessian_refinement"]["0.0005"]
    diagnostics = probe["oracle"]["result"]["covariance_diagnostics"]
    if mutation == "unstable_hessian":
        diagnostics["hessian"][0][0] += 1
    else:
        diagnostics["native_parameters"][-1] += .01
    policy["auxiliary_fingerprint"] = fingerprint(auxiliary)
    with pytest.raises(AssertionError):
        rank_one_reference(model, spec, observation, policy, auxiliary, caught)


@pytest.mark.parametrize("family", ["nbinom1", "nbinom2"])
@pytest.mark.parametrize("count,mean,size", [
    (0, 1.5, 1e8), (4, 3.5, 1e12), (20, 7.1, 1e4), (50, .7, 1e6),
])
def test_poisson_limit_bounds_against_high_precision_recurrence(family, count, mean, size):
    """An independent 80-digit density and derivative validate the envelope."""
    dispersion = mean / size if family == "nbinom1" else size
    bound = poisson_density_bounds([count], [mean], [dispersion], [1], family)
    with localcontext() as context:
        context.prec = 80
        y, mu, d = Decimal(count), Decimal(mean), Decimal(dispersion)
        r = mu / d if family == "nbinom1" else d
        t = mu / r
        density = (sum((1 + Decimal(j) / r).ln() for j in range(count))
                   - y * (1 + t).ln() + r * (t - (1 + t).ln()))
        first = (-sum((Decimal(j) / r) / (1 + Decimal(j) / r) for j in range(count))
                 + y * t / (1 + t) + r * (-(1 + t).ln() + t / (1 + t)))
        second = (sum((Decimal(j) / r) / (1 + Decimal(j) / r)**2 for j in range(count))
                  - y * t / (1 + t)**2
                  + r * (-(1 + t).ln() + t / (1 + t) + t * t / (1 + t)**2))
        assert float(abs(density)) <= bound["density"]
        assert float(abs(second) + first * first / 4) <= bound["curvature"]
