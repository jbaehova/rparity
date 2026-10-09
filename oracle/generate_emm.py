"""Generate synthetic marginal-mean goldens through the sole R black box."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def make_specs(count: int = 600) -> list[dict[str, Any]]:
    """Vary sample balance, interactions, links, averaging and comparisons."""
    cases = []
    adjustments = ["none", "bonferroni", "holm", "sidak", "fdr", "tukey"]
    methods: list[Any] = ["pairwise", "revpairwise", "trt.vs.ctrl", "consec", "poly",
                          {"first vs others": [1, -0.5, -0.5]}]
    for i in range(count):
        rng = np.random.default_rng(918200 + i)
        rows = []
        for ai, a in enumerate(["a", "b", "c"]):
            for bi, b in enumerate(["low", "high"]):
                n = int(rng.integers(8, 15))
                for _ in range(n):
                    x = float(rng.normal())
                    eta = -0.4 + 0.45 * ai + 0.25 * bi + 0.15 * ai * bi + 0.2 * x
                    rows.append({"a": a, "b": b, "x": x, "eta": eta})
        data = pd.DataFrame(rows)
        call = "lm" if i % 3 == 0 else "glm"
        family = "binomial" if i % 3 == 1 else "poisson"
        link = ["logit", "probit", "cloglog"][(i // 3) % 3]
        if call == "lm":
            data["y"] = data.eta + rng.normal(0, 0.7, len(data))
        elif family == "poisson":
            data["y"] = rng.poisson(np.exp(data.eta))
        else:
            from scipy.special import expit, ndtr

            probability = {"logit": expit, "probit": ndtr,
                           "cloglog": lambda x: -np.expm1(-np.exp(x))}[link](data.eta)
            data["y"] = rng.binomial(1, probability)
        del data["eta"]
        spec: dict[str, Any] = {
            "id": f"emm_{i:04d}", "call": call, "formula": "y ~ a * b + x",
            "data": data.to_dict(orient="list"), "factors": ["a", "b"],
            "family": family, "link": link, "operation": "emmeans",
            "emm_args": {"specs": "a"}, "ddf": "kenward-roger",
        }
        option_index = (i // 3) % 12
        if option_index < 5:
            weighting = ["equal", "proportional", "outer", "cells", "flat"][option_index]
            spec["emm_args"]["weights"] = weighting
            option = f"emmeans / weights {weighting}"
        elif option_index < 9:
            method = methods[(i // 36) % len(methods)]
            adjust = adjustments[(i // 12) % len(adjustments)]
            spec["emm_args"]["by"] = "b"
            spec["contrast"] = {"method": method, "adjust": adjust}
            spec["adjust"] = adjust
            option = f"contrast / {method if isinstance(method, str) else 'custom'} / {adjust}"
        elif option_index == 9:
            spec["emm_args"]["at"] = {"x": [-0.5, 0.5]}
            spec["emm_args"]["specs"] = "~ a * x"
            spec["response_type"] = "response"
            option = "emmeans / response / at"
        elif option_index == 10:
            spec["operation"] = "emtrends"
            spec["emm_args"]["var"] = "x"
            option = "emtrends / numeric slope"
        else:
            spec["operation"] = "joint_tests"
            spec["emm_args"] = {}
            option = "joint_tests / factorial"
        cases.append({"id": spec["id"], "module": "emm", "option": option, "spec": spec})
    return cases


def main() -> None:
    """Write real R outputs and all synthetic input observations into fixtures."""
    cases = make_specs() + make_native_specs() + make_extra_specs()
    cache = ROOT / "oracle/cache"
    cache.mkdir(parents=True, exist_ok=True)
    input_path = cache / "emm-golden-input.json"
    output_path = cache / "emm-golden-output.json"
    input_path.write_text(json.dumps({"cases": [case["spec"] for case in cases]}))
    subprocess.run(["Rscript", str(ROOT / "oracle/run_case.R"), str(input_path), str(output_path)], check=True)
    results = json.loads(output_path.read_text())
    boundaries = []
    for case in cases:
        spec = case["spec"]
        if spec["call"] != "glm":
            continue
        table = pd.DataFrame(spec["data"]).groupby(["a", "b"]).y.agg(["sum", "count"])
        separated = table["sum"] == 0
        if spec["family"] == "binomial":
            separated |= table["sum"] == table["count"]
        if separated.any():
            case["boundary"] = "An all-zero or all-one saturated factor cell has no finite GLM intercept MLE."
            boundaries.append({**spec, "operation": "fit"})
    if boundaries:
        boundary_input = cache / "emm-boundary-input.json"
        boundary_output = cache / "emm-boundary-output.json"
        boundary_input.write_text(json.dumps({"cases": boundaries}))
        subprocess.run(["Rscript", str(ROOT / "oracle/run_case.R"), str(boundary_input), str(boundary_output)], check=True)
        boundary_results = {result["id"]: result for result in json.loads(boundary_output.read_text())}
        for case in cases:
            if case["id"] in boundary_results:
                case["fit_oracle"] = boundary_results[case["id"]]
    output = ROOT / "tests/golden/emm"
    output.mkdir(parents=True, exist_ok=True)
    for case, result in zip(cases, results, strict=True):
        case["oracle"] = result
        (output / f"{case['id']}.json").write_text(json.dumps(case, indent=2) + "\n")
    errors = sum("error" in result["result"] for result in results)
    print(f"Created {len(cases)} emmeans goldens; oracle errors: {errors}")


def make_native_specs(count: int = 120) -> list[dict[str, Any]]:
    """Exercise each native adapter with explicit mixed-model df settings."""
    cases = []
    for j in range(count):
        i = 600 + j
        rng = np.random.default_rng(23910 + j)
        g = np.repeat(np.arange(8), 9)
        a = np.tile(np.repeat(["a", "b", "c"], 3), 8)
        x = rng.uniform(0.7, 1.5, len(g))
        random = rng.normal(0, 0.8, 8)[g]
        eta = 1.0 + 0.4 * (a == "b") + 0.9 * (a == "c") + 0.3 * x + random
        call = ["lmer", "glmer", "gls"][j % 3]
        y = rng.poisson(np.exp(eta)) if call == "glmer" else eta + rng.normal(0, 0.6, len(g))
        data = {"y": y.tolist(), "a": a.tolist(), "x": x.tolist(),
                "g": [f"g{value}" for value in g], "t": np.tile(np.arange(9), 8).tolist()}
        spec: dict[str, Any] = {
            "id": f"emm_{i:04d}", "call": call, "formula": "y ~ a + x" + (" + (1|g)" if call != "gls" else ""),
            "data": data, "factors": ["a", "g"], "family": "poisson", "link": "log",
            "operation": "emmeans", "emm_args": {"specs": "a"},
            "ddf": ["kenward-roger", "satterthwaite", "asymptotic"][(j // 3) % 3],
        }
        if call == "gls":
            spec["correlation"] = {"name": "corAR1", "args": {"form": "~ t | g"}}
            spec["gls_control"] = {"opt": "optim", "optimMethod": "BFGS", "msTol": 1e-14,
                                   "msMaxIter": 3000}
        if call == "glmer":
            spec["glmer_tol"] = 1e-14
            spec["glmer_nAGQ0initStep"] = False
        if (j // 3) % 4 == 1:
            spec["contrast"] = {"method": "pairwise", "adjust": "tukey"}
            spec["adjust"] = "tukey"
            operation = "pairs"
        elif (j // 3) % 4 == 2:
            spec["operation"] = "emtrends"
            spec["emm_args"]["var"] = "x"
            operation = "emtrends"
        elif (j // 3) % 4 == 3:
            spec["response_type"] = "response"
            operation = "response"
        else:
            operation = "emmeans"
        cases.append({"id": spec["id"], "module": "emm", "option": f"{operation} / {call} / {spec['ddf']}", "spec": spec})
    return cases


def make_extra_specs() -> list[dict[str, Any]]:
    """Verify transform-before-average and joint tests through native adapters."""
    cases = []
    for j, case in enumerate(make_specs(24)):
        spec = case["spec"]
        case["id"] = spec["id"] = f"emm_{720 + j:04d}"
        spec["operation"] = "emmeans"
        spec["emm_args"] = {"specs": "a", "regrid": "response"}
        spec["response_type"] = "response"
        case["option"] = f"regrid / response / {spec['call']}"
        cases.append(case)
    for j, case in enumerate(make_native_specs(9)):
        spec = case["spec"]
        case["id"] = spec["id"] = f"emm_{744 + j:04d}"
        spec["operation"] = "joint_tests"
        spec["emm_args"] = {}
        spec.pop("contrast", None)
        spec.pop("adjust", None)
        case["option"] = f"joint_tests / {spec['call']} / {spec['ddf']}"
        cases.append(case)
    return cases


if __name__ == "__main__":
    main()
