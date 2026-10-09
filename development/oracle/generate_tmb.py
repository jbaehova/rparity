"""Seeded extended GLMM designs, evaluated only by the approved R driver.

All family, zero-inflation, dispersion and random-effect crosses are generated
before either fitter runs. Errors and boundary fits remain in the fixture corpus.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("poisson", "nbinom1", "nbinom2", "binomial", "beta", "gaussian")
ZERO_FORMULAS = ("~0", "~1", "~z + x")
DISPERSION_FORMULAS = ("~1", "~x", "~x + f")
RANDOM_FORMULAS = ("(1|g)", "(x|g)")
REFERENCE_OVERRIDES = json.loads((ROOT / "development/oracle/stage2_overrides.json").read_text())


def make_case(index: int) -> dict[str, Any]:
    """Return a reproducible, independently generated mixed likelihood design."""
    rng = np.random.default_rng(2_013_857 + index)
    family = FAMILIES[index % len(FAMILIES)]
    zi = (index // len(FAMILIES)) % len(ZERO_FORMULAS)
    dispersion = (index // (len(FAMILIES) * len(ZERO_FORMULAS))) % len(DISPERSION_FORMULAS)
    random = (index // (len(FAMILIES) * len(ZERO_FORMULAS) * len(DISPERSION_FORMULAS))) % len(RANDOM_FORMULAS)
    replicate = index // (len(FAMILIES) * len(ZERO_FORMULAS) * len(DISPERSION_FORMULAS) * len(RANDOM_FORMULAS))
    ng = int(rng.integers(6, 10))
    sizes = rng.integers(10, 18, ng)
    if replicate % 2:
        sizes[0] += int(rng.integers(8, 16))
    group_numbers = np.repeat(np.arange(ng), sizes)
    n = len(group_numbers)
    g = np.array([f"g{number:02d}" for number in group_numbers])
    x = rng.uniform(-1.4, 1.4, n)
    z = rng.normal(0, 0.7, n)
    f = rng.choice(["a", "b", "c"], n, p=[0.45, 0.35, 0.2])
    f[:3] = ["a", "b", "c"]
    amplitude = 0.0 if replicate == 10 else float(rng.uniform(0.45, 0.85))
    intercept = rng.normal(0, amplitude, ng)
    slope = 0.3 * intercept + rng.normal(0, amplitude * 0.35, ng)
    offset = rng.uniform(-0.3, 0.3, n) if replicate % 3 == 1 else np.zeros(n)
    eta = 0.3 + 0.4 * x + 0.2 * (f == "b") - 0.15 * (f == "c") + intercept[group_numbers] + offset
    if random:
        eta += slope[group_numbers] * x
    disp_eta = (np.log(3.5) if family in {"nbinom2", "beta"} else np.log(0.6)) + (0.25 * x if dispersion else 0)
    if dispersion == 2:
        disp_eta += 0.2 * (f == "b") - 0.15 * (f == "c")
    phi = np.exp(disp_eta)
    link = "identity" if family == "gaussian" else "logit" if family in {"binomial", "beta"} else "log"
    mean = eta if family == "gaussian" else expit(eta) if family in {"binomial", "beta"} else np.exp(eta)
    response_formula = "y"
    trials = rng.integers(5, 13, n) if family == "binomial" and replicate % 3 else np.ones(n, dtype=int)
    if family == "poisson":
        response = rng.poisson(mean)
    elif family == "nbinom1":
        response = rng.negative_binomial(mean / phi, 1 / (1 + phi))
    elif family == "nbinom2":
        response = rng.negative_binomial(phi, phi / (phi + mean))
    elif family == "binomial":
        response = rng.binomial(trials, mean)
    elif family == "beta":
        # Floating-point random variates occasionally round an open-support draw
        # to exactly one. Keep every indexed case while enforcing its domain.
        response = np.clip(rng.beta(mean * phi, (1 - mean) * phi), 1e-12, 1 - 1e-12)
    else:
        response = rng.normal(mean, np.sqrt(phi))
    zi_probability = np.zeros(n) if zi == 0 else expit(-1.4 + (0.35 * z - 0.25 * x if zi == 2 else 0) + np.zeros(n))
    zero_inflated = rng.random(n) < zi_probability
    response = np.where(zero_inflated, 0, response)
    failures = trials - response if family == "binomial" else np.zeros(n)
    y = response / trials if family == "binomial" else response
    weights = np.ones(n)
    if family == "binomial" and replicate % 3 == 1:
        weights = trials.astype(float)
    elif family == "binomial" and replicate % 3 == 2:
        response_formula = "cbind(success, failure)"
    elif replicate % 3 == 2:
        weights = rng.integers(1, 4, n).astype(float)
    rhs = "x + f + " + RANDOM_FORMULAS[random]
    if np.any(offset):
        rhs += " + offset(o)"
    spec: dict[str, Any] = {
        "id": f"tmb_{index:04d}", "call": "glmmTMB", "formula": f"{response_formula} ~ {rhs}",
        "family": family, "link": link,
        "ziformula": ZERO_FORMULAS[zi], "dispformula": DISPERSION_FORMULAS[dispersion],
        "tmb_control": {"optCtrl": {"iter.max": 10000, "eval.max": 10000, "rel.tol": 1e-14, "x.tol": 1e-12, "sing.tol": 1e-16, "xf.tol": 1e-16}},
        "score_polish": True,
        "factors": ["f", "g"],
        "data": {
            "y": y.tolist(), "x": x.tolist(), "z": z.tolist(), "f": f.tolist(), "g": g.tolist(),
            "o": offset.tolist(), "weight": weights.tolist(), "trials": trials.tolist(),
            "success": response.tolist(), "failure": failures.tolist(),
        },
        "newdata": {
            "x": np.linspace(-1.2, 1.2, 12).tolist(), "z": rng.normal(0, 0.6, 12).tolist(),
            "f": ["a", "b", "c"] * 4,
            "g": [f"g{number % ng:02d}" for number in range(12)],
            "o": rng.uniform(-0.2, 0.2, 12).tolist() if np.any(offset) else [0.0] * 12,
        },
    }
    if spec["id"] in REFERENCE_OVERRIDES:
        override = REFERENCE_OVERRIDES[spec["id"]]
        spec["args"] = override["args"]
        for name in ("tmb_covariance_diagnostics", "tmb_hessian_step", "tmb_inner_control"):
            if name in override:
                spec[name] = override[name]
    if np.any(weights != 1):
        spec["weights"] = "weight"
    return {
        "id": spec["id"], "module": "tmb",
        "option": f"glmmTMB / {family} / zi {ZERO_FORMULAS[zi]} / disp {DISPERSION_FORMULAS[dispersion]} / {RANDOM_FORMULAS[random]}",
        "spec": spec,
        "design": {
            "seed": 2_013_857 + index, "replicate": replicate, "n": n,
            "groups": ng, "group_sizes": sizes.tolist(), "zero_true_random_variance": amplitude == 0,
            "response_form": "cbind" if response_formula.startswith("cbind") else "weighted-proportion" if family == "binomial" and replicate % 3 == 1 else "ordinary",
            "dispersion_applicable": family not in {"poisson", "binomial"},
            "zero_inflation": bool(zi), "integer_case_weights": bool(replicate % 3 == 2 and family != "binomial"),
        },
    }


def generate(count: int, start: int = 0, batch_size: int = 25, specs_only: bool = False) -> None:
    """Evaluate reproducible batches without dropping difficult oracle outcomes."""
    cases = [make_case(index) for index in range(start, start + count)]
    cache = ROOT / "development/oracle/cache"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / f"s2-design-tmb-{start:04d}-{count}.json").write_text(json.dumps(cases, indent=2) + "\n")
    if specs_only:
        print(f"Prepared {len(cases)} TMB designs: {dict(Counter(case['spec']['family'] for case in cases))}")
        return
    destination = ROOT / "tests/golden/tmb"
    destination.mkdir(parents=True, exist_ok=True)
    errors = 0
    for offset in range(0, len(cases), batch_size):
        batch = cases[offset:offset + batch_size]
        input_path = cache / f"s2-design-tmb-input-{start + offset:04d}.json"
        output_path = cache / f"s2-design-tmb-output-{start + offset:04d}.json"
        input_path.write_text(json.dumps({"cases": [case["spec"] for case in batch]}))
        subprocess.run(["Rscript", str(ROOT / "development/oracle/run_case.R"), str(input_path), str(output_path)], check=True, cwd=ROOT)
        outputs = json.loads(output_path.read_text())
        for case, output in zip(batch, outputs, strict=True):
            assert case["id"] == output["id"]
            existing_path = destination / f"{case['id']}.json"
            if existing_path.exists():
                previous = json.loads(existing_path.read_text())
                if previous.get("oracle_attempts"):
                    case["oracle_attempts"] = previous["oracle_attempts"]
                if previous["spec"] != case["spec"]:
                    case.setdefault("oracle_attempts", []).append({
                        "reason": "Uniform explicit high-precision nlminb controls with independent stationary-score Newton polish. Fourteen previously observed local optima use documented, independently R-confirmed public starts from development/oracle/stage2_overrides.json. Beta endpoints are projected to open support where needed. Original input and observation retained.",
                        "spec": previous["spec"], "oracle": previous["oracle"],
                    })
            case["oracle"] = output
            errors += "error" in output["result"]
            pending = destination / f".{case['id']}.pending"
            pending.write_text(json.dumps(case, indent=2) + "\n")
            pending.replace(existing_path)
        print(f"TMB observed {min(offset + batch_size, count)}/{count}; oracle errors={errors}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1200)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--specs-only", action="store_true")
    args = parser.parse_args()
    generate(args.count, args.start, args.batch_size, args.specs_only)
