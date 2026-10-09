"""Generate synthetic GLMM goldens through the approved R black-box driver."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from rparity.lmm._glmer_math import inverse_link

ROOT = Path(__file__).resolve().parents[1]


def make_case(seed: int) -> dict:
    """One reproducible design, varying families, response forms and grouping."""
    rng = np.random.default_rng(452000 + seed)
    family = "poisson" if seed % 4 == 3 else "binomial"
    link = "log" if family == "poisson" else ["logit", "probit", "cloglog"][seed % 3]
    ng = 6 + seed % 4
    per = 8 + seed % 5
    g = np.repeat(np.arange(ng), per)
    h = np.tile(np.arange(per), ng) % 4
    x = rng.normal(size=len(g))
    a = np.tile(["a", "b"], (len(g) + 1) // 2)[: len(g)]
    intercept = rng.normal(0, 0.45 + 0.12 * (seed % 3), ng)
    slope = rng.normal(0, 0.25, ng)
    structure = (seed // 4) % 6
    random = ["(1|g)", "(x|g)", "(x||g)", "(1|g)+(1|h)", "(1|g/h)", "(0+x|g)"][structure]
    eta = -0.25 + 0.55 * x + 0.22 * (a == "b")
    if structure != 5:
        eta += intercept[g]
    if structure in {1, 2, 5}:
        eta += slope[g] * x
    if structure == 3:
        eta += rng.normal(0, 0.35, 4)[h]
    if structure == 4:
        eta += rng.normal(0, 0.25, (ng, 4))[g, h]
    offset = rng.uniform(-0.3, 0.3, len(g)) if seed % 7 == 0 else np.zeros(len(g))
    eta += offset
    mean = inverse_link(eta.astype(float), link)[0]
    trials = rng.integers(5, 16, len(g)) if (seed // 3) % 3 != 0 else np.ones(len(g), dtype=int)
    if family == "poisson":
        response = rng.poisson(mean)
    else:
        response = rng.binomial(trials, mean)
    data = pd.DataFrame(
        {
            "x": x,
            "g": g,
            "h": h,
            "a": a,
            "offset": offset,
            "trials": trials,
            "success": response,
            "failure": trials - response,
            "y": response if family == "poisson" else response / trials,
        }
    )
    response_formula = "y"
    weights = None
    form = (seed // 3) % 3
    if family == "binomial" and form == 1:
        weights = "trials"
    elif family == "binomial" and form == 2:
        response_formula = "cbind(success,failure)"
    fixed = "x+a"
    if seed % 7 == 0:
        fixed += "+offset(offset)"
    spec = {
        "id": f"glmer_{seed:04d}",
        "call": "glmer",
        "formula": f"{response_formula}~{fixed}+{random}",
        "family": family,
        "link": link,
        "factors": ["a"],
        "glmer_tol": 1e-14,
        "glmer_nAGQ0initStep": False,
        "data": data.to_dict(orient="list"),
    }
    if weights:
        spec["weights"] = weights
    return spec


def fractional_cases() -> list[dict]:
    """R-valid fractional prior weights and fractional cbind counts."""
    rng = np.random.default_rng(1)
    group = np.repeat(np.arange(10), 20)
    x = rng.normal(size=len(group))
    random = rng.normal(0, 0.6, 10)
    response = rng.binomial(1, inverse_link(0.1 + 0.5 * x + random[group], "logit")[0])
    cases = []
    for i, mode in enumerate(["binary", "proportion", "cbind"]):
        y = response if mode == "binary" else 0.2 + 0.1 * (np.arange(len(group)) % 5)
        weight = np.full(len(group), 1.5) if mode == "binary" else 7.3 + np.arange(len(group)) % 3
        frame = pd.DataFrame(
            {"y": y, "x": x, "g": group, "weight": weight, "success": y * 8, "failure": (1 - y) * 8}
        )
        cases.append(
            {
                "id": f"glmer_{600 + i:04d}",
                "call": "glmer",
                "formula": "cbind(success,failure)~x+(1|g)" if mode == "cbind" else "y~x+(1|g)",
                "family": "binomial",
                "link": "logit",
                "weights": "weight",
                "glmer_tol": 1e-14,
                "glmer_nAGQ0initStep": False,
                "data": frame.to_dict(orient="list"),
                "weight_form": mode,
            }
        )
    return cases


def generate(count: int) -> None:
    """Execute one R batch and retain every observed result, including failures."""
    specs = [make_case(seed) for seed in range(count)] + fractional_cases()
    destination = ROOT / "tests/golden/glmer"
    destination.mkdir(parents=True, exist_ok=True)
    attempts: dict[str, list[dict]] = {}
    for spec in specs:
        existing_path = destination / f"{spec['id']}.json"
        if existing_path.exists():
            existing = json.loads(existing_path.read_text())
            if "oracle_attempts" in existing:
                attempts[spec["id"]] = existing["oracle_attempts"]
            if existing["spec"].get("glmer_tol") == 1e-15:
                attempts.setdefault(spec["id"], []).append(
                    {"glmer_tol": 1e-15, "glmer_nAGQ0initStep": False, "oracle": existing["oracle"]}
                )
    with tempfile.TemporaryDirectory() as directory:
        input_path = Path(directory) / "spec.json"
        output_path = Path(directory) / "result.json"

        def execute(batch: list[dict]) -> list[dict]:
            input_path.write_text(json.dumps({"cases": batch}))
            subprocess.run(
                ["Rscript", str(ROOT / "oracle/run_case.R"), str(input_path), str(output_path)],
                check=True,
            )
            return json.loads(output_path.read_text())

        results = execute(specs)
        for tolerance in [1e-13, 1e-12]:
            failed = [
                i
                for i, result in enumerate(results)
                if any(
                    label in result["result"].get("error", "").lower()
                    for label in ("pirls", "pwrss")
                )
            ]
            if not failed:
                break
            for i in failed:
                attempts.setdefault(specs[i]["id"], []).append(
                    {"glmer_tol": specs[i]["glmer_tol"], "oracle": results[i]}
                )
                specs[i]["glmer_tol"] = tolerance
            retry = execute([specs[i] for i in failed])
            for i, result in zip(failed, retry, strict=True):
                results[i] = result
    for spec, result in zip(specs, results, strict=True):
        fixture = {
            "id": spec["id"],
            "module": "glmer",
            "option": f"glmer / {spec['family']}-{spec['link']}",
            "spec": spec,
            "oracle": result,
        }
        if "weight_form" in spec:
            fixture["option"] += f" fractional {spec['weight_form']}"
        if spec["id"] in attempts:
            fixture["oracle_attempts"] = attempts[spec["id"]]
        (destination / f"{spec['id']}.json").write_text(json.dumps(fixture, indent=2) + "\n")
    errors = sum("error" in result["result"] for result in results)
    print(f"Generated {len(results)} GLMM fixtures; oracle errors={errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=600)
    generate(parser.parse_args().count)
