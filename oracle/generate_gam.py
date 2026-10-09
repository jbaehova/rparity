"""Seeded GAM designs and black-box R fixtures for the Stage 2 parity gate.

No fitter is imported. The synthetic designs are fixed before observing either
implementation, and every oracle outcome is retained, including errors.
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

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_OVERRIDES = json.loads((ROOT / "oracle/stage2_overrides.json").read_text())
FAMILIES = ("gaussian", "binomial", "poisson", "Gamma")
METHODS = ("REML", "ML", "GCV.Cp")
ARCHETYPES = ("tp", "cr", "cs", "ps", "re", "te", "ti", "continuous-by", "factor-by", "additive")


def make_case(index: int) -> dict[str, Any]:
    """Cross all required family, method and smooth classes with seeded designs."""
    rng = np.random.default_rng(1_209_731 + index)
    archetype = ARCHETYPES[index % len(ARCHETYPES)]
    family = FAMILIES[(index // len(ARCHETYPES)) % len(FAMILIES)]
    method = METHODS[(index // (len(ARCHETYPES) * len(FAMILIES))) % len(METHODS)]
    replicate = index // (len(ARCHETYPES) * len(FAMILIES) * len(METHODS))
    n = int(rng.integers(95, 151))
    k = int(rng.integers(5, 9))
    x = rng.uniform(-1.8, 1.8, n)
    z = rng.uniform(-1.4, 1.4, n)
    w = rng.uniform(0.5, 1.8, n)
    probabilities = np.array([0.28, 0.36, 0.36]) if replicate % 2 == 0 else np.array([0.55, 0.3, 0.15])
    f = rng.choice(["a", "b", "c"], n, p=probabilities)
    # Fix a row per level before generating the response, so every design is valid.
    f[:3] = ["a", "b", "c"]
    ng = int(rng.integers(6, 11))
    groups = rng.choice(np.arange(ng), n)
    groups[:ng] = np.arange(ng)
    g = np.array([f"g{number:02d}" for number in groups])
    factor_effect = 0.25 * (f == "b") - 0.15 * (f == "c")
    amplitude = 0.0 if replicate == 9 else float(rng.uniform(0.45, 0.9))
    smooth = amplitude * (np.sin(1.6 * x) + 0.15 * x**2)
    rhs = "f"
    if archetype in {"tp", "cr", "cs", "ps"}:
        rhs += f' + s(x, bs="{archetype}", k={k})'
    elif archetype == "re":
        rhs += ' + x + s(g, bs="re")'
        smooth = rng.normal(0, amplitude, ng)[groups]
    elif archetype == "te":
        marginal_k = 4 + replicate % 2
        rhs += f' + te(x, z, bs=c("cr", "cr"), k=c({marginal_k}, {marginal_k}))'
        smooth += 0.3 * np.cos(1.4 * z) + 0.2 * x * z
    elif archetype == "ti":
        rhs += f' + s(x, bs="cr", k={k}) + s(z, bs="cr", k={k})'
        rhs += ' + ti(x, z, bs=c("cr", "cr"), k=c(4, 4))'
        smooth += 0.3 * np.cos(1.4 * z) + 0.2 * x * z
    elif archetype == "continuous-by":
        basis = ("tp", "cr", "cs", "ps")[replicate % 4]
        rhs += f' + w + s(x, by=w, bs="{basis}", k={k})'
        smooth *= w
    elif archetype == "factor-by":
        basis = ("tp", "cr", "cs", "ps")[replicate % 4]
        rhs += f' + s(x, by=f, bs="{basis}", k={k})'
        smooth *= 1 + 0.35 * (f == "b") - 0.2 * (f == "c")
    else:
        first, second = (("tp", "cr"), ("cs", "ps"))[replicate % 2]
        rhs += f' + s(x, bs="{first}", k={k}) + s(z, bs="{second}", k={k})'
        smooth += 0.3 * np.cos(1.4 * z)
    offset = rng.uniform(-0.25, 0.25, n) if replicate % 3 == 1 else np.zeros(n)
    if np.any(offset):
        rhs += " + offset(o)"
    eta = 0.25 + factor_effect + smooth + offset
    response_formula = "y"
    trial_count = rng.integers(3, 10, n)
    weights = rng.uniform(0.65, 1.8, n) if replicate % 3 == 2 else np.ones(n)
    if family == "gaussian":
        y = eta + rng.normal(0, rng.uniform(0.45, 0.9), n)
        successes = y
        failures = np.zeros(n)
    elif family == "poisson":
        y = rng.poisson(np.exp(eta))
        successes = y
        failures = np.zeros(n)
    elif family == "Gamma":
        shape = float(rng.uniform(3.5, 7))
        # The public default inverse link also receives valid positive predictors.
        mean = 1 / (2.0 + 0.5 * factor_effect + 0.35 * smooth + offset) if replicate % 2 == 0 else np.exp(eta)
        y = rng.gamma(shape, mean / shape)
        successes = y
        failures = np.zeros(n)
    else:
        mode = replicate % 3
        trials = trial_count if mode else np.ones(n, dtype=int)
        successes = rng.binomial(trials, expit(eta))
        failures = trials - successes
        y = successes / trials
        if mode == 1:
            weights = trials.astype(float)
        elif mode == 2:
            response_formula = "cbind(success, failure)"
            # Integer trial counts without fractional-prior-weight ambiguity.
            weights = np.ones(n)
    new_n = 15
    newdata = {
        "x": np.linspace(-1.6, 1.6, new_n).tolist(),
        "z": rng.uniform(-1.2, 1.2, new_n).tolist(),
        "w": rng.uniform(0.6, 1.7, new_n).tolist(),
        "f": ["a", "b", "c"] * 5,
        "g": [f"g{number % ng:02d}" for number in range(new_n)],
        "o": rng.uniform(-0.2, 0.2, new_n).tolist() if np.any(offset) else [0.0] * new_n,
    }
    spec: dict[str, Any] = {
        "id": f"gam_{index:04d}",
        "call": "gam",
        "formula": f"{response_formula} ~ {rhs}",
        "family": family,
        "link": "identity" if family == "gaussian" else "logit" if family == "binomial" else "inverse" if family == "Gamma" and replicate % 2 == 0 else "log",
        "args": {"method": method},
        "gam_control": {"epsilon": 1e-12, "maxit": 1000, "mgcv.tol": 1e-10, "newton": {"conv.tol": 1e-10}},
        "factors": ["f", "g"],
        "data": {
            "y": y.tolist(), "x": x.tolist(), "z": z.tolist(), "w": w.tolist(),
            "f": f.tolist(), "g": g.tolist(), "o": offset.tolist(),
            "weight": weights.tolist(), "success": successes.tolist(), "failure": failures.tolist(),
        },
        "newdata": newdata,
        "gam_prediction": True,
        "gam_check": True,
    }
    if spec["id"] in REFERENCE_OVERRIDES:
        override = REFERENCE_OVERRIDES[spec["id"]]
        spec["args"].update(override["args"])
        for name in ("gam_control", "gam_initial_sp"):
            if name in override:
                spec[name] = override[name]
    if np.any(weights != 1):
        spec["weights"] = "weight"
    return {
        "id": spec["id"], "module": "gam",
        "option": f"gam / {archetype} / {family} / {method}", "spec": spec,
        "design": {
            "seed": 1_209_731 + index, "archetype": archetype,
            "replicate": replicate, "n": n, "k": k, "groups": ng,
            "imbalanced_factor": bool(replicate % 2), "zero_true_smooth": amplitude == 0,
            "response_form": "cbind" if response_formula.startswith("cbind") else "weighted-proportion" if family == "binomial" and replicate % 3 == 1 else "ordinary",
        },
    }


def generate(count: int, start: int = 0, batch_size: int = 40, specs_only: bool = False) -> None:
    """Observe approved R batches and retain every indexed synthetic input."""
    cases = [make_case(index) for index in range(start, start + count)]
    cache = ROOT / "oracle/cache"
    cache.mkdir(parents=True, exist_ok=True)
    manifest = cache / f"s2-design-gam-{start:04d}-{count}.json"
    manifest.write_text(json.dumps(cases, indent=2) + "\n")
    if specs_only:
        print(f"Prepared {len(cases)} GAM designs: {dict(Counter(case['option'] for case in cases))}")
        return
    destination = ROOT / "tests/golden/gam"
    destination.mkdir(parents=True, exist_ok=True)
    errors = 0
    for offset in range(0, len(cases), batch_size):
        batch = cases[offset:offset + batch_size]
        input_path = cache / f"s2-design-gam-input-{start + offset:04d}.json"
        output_path = cache / f"s2-design-gam-output-{start + offset:04d}.json"
        input_path.write_text(json.dumps({"cases": [case["spec"] for case in batch]}))
        subprocess.run(["Rscript", str(ROOT / "oracle/run_case.R"), str(input_path), str(output_path)], check=True, cwd=ROOT)
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
                        "reason": "Uniform documented high-precision IRLS and smoothing controls. Explicit, independently R-confirmed in.out initialization overrides are recorded in oracle/stage2_overrides.json; original input and observation retained.",
                        "spec": previous["spec"], "oracle": previous["oracle"],
                    })
            case["oracle"] = output
            errors += "error" in output["result"]
            pending = destination / f".{case['id']}.pending"
            pending.write_text(json.dumps(case, indent=2) + "\n")
            pending.replace(existing_path)
        print(f"GAM observed {min(offset + batch_size, count)}/{count}; oracle errors={errors}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1200)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--specs-only", action="store_true")
    args = parser.parse_args()
    generate(args.count, args.start, args.batch_size, args.specs_only)
