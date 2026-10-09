"""Generate seeded synthetic lmer cases using only the public R oracle.

Run ``uv run python oracle/generate_lmer.py --count 600`` to regenerate.
Covariance likelihoods follow Bates et al. (2015), equations 34 and 39-41.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STRUCTURES = ["(1|g)", "(x|g)", "(x||g)", "(1|g)+(1|h)", "(1|g/h)", "(0+x|g)"]


def make_case(number: int) -> dict:
    """Return an independently seeded irregular grouped Gaussian experiment."""
    rng = np.random.default_rng(821000 + number)
    kind = number % len(STRUCTURES)
    ng = int(rng.integers(7, 13))
    sizes = rng.integers(5, 9, ng)
    g = np.repeat(np.arange(ng), sizes)
    n = len(g)
    h = rng.integers(0, 4, n)
    x = rng.normal(size=n)
    a = np.asarray(["A", "B", "C"])[np.arange(n) % 3]
    weights = rng.uniform(.5, 2, n) if number % 7 == 0 else np.ones(n)
    offset = rng.normal(scale=.3, size=n) if number % 9 == 0 else np.zeros(n)
    random_intercept = rng.normal(scale=rng.uniform(.6, 1.8), size=ng)
    random_slope = rng.normal(scale=rng.uniform(.4, 1), size=ng)
    nested = rng.normal(scale=.6, size=(ng, 4))
    crossed = rng.normal(scale=.7, size=4)
    random = (random_intercept[g] if kind != 5 else 0)
    if kind in (1, 2, 5):
        random = random + random_slope[g] * x
    elif kind == 3:
        random = random + crossed[h]
    elif kind == 4:
        random = random + nested[g, h]
    categorical = number % 5 == 1
    y = 2 + 1.3*x + random + offset + rng.normal(scale=.55, size=n)/np.sqrt(weights)
    if categorical:
        y += np.where(a == "B", .7, np.where(a == "C", -.3, 0))
    # An exactly orthogonal balanced design exercises zero-variance solutions.
    boundary = number % 40 == 0
    if boundary:
        ng, n = 8, 48
        g = np.repeat(np.arange(ng), 6)
        h = np.tile(np.arange(3), 16)
        x = np.tile([-1, -.5, 0, .5, 1, 1.5], ng)
        a = np.asarray(["A", "B", "C"])[np.arange(n) % 3]
        weights, offset = np.ones(n), np.zeros(n)
        noise = rng.normal(size=6)
        noise -= np.column_stack([np.ones(6), x[:6]]) @ np.linalg.lstsq(
            np.column_stack([np.ones(6), x[:6]]), noise, rcond=None)[0]
        y = 2 + 1.3*x + np.tile(noise, ng)
        kind, categorical = 0, False
    data = {"y": y.tolist(), "x": x.tolist(), "g": [f"g{j:02}" for j in g],
            "h": [f"h{j}" for j in h], "a": a.tolist(), "w": weights.tolist(),
            "off": offset.tolist()}
    missing = number % 17 == 0 and not boundary
    if missing:
        for column, row in [("y", 2), ("x", 4), ("g", 6)]:
            data[column][row] = None
    fixed = "x + a" if categorical else "x"
    formula = f"y ~ {fixed} + {STRUCTURES[kind]}"
    spec = {"id": f"lmer_{number:04d}", "call": "lmer", "operation": "fit",
            "formula": formula, "data": data, "factors": ["g", "h", "a"],
            "args": {"REML": number % 2 == 0},
            "contrasts": "sum" if categorical and number % 10 == 6 else "treatment"}
    if number % 7 == 0 and not boundary:
        spec["weights"] = "w"
    if number % 9 == 0 and not boundary:
        # Formula offsets also have an unambiguous public predict(newdata) contract.
        spec["formula"] += " + offset(off)"
    labels = ["REML" if spec["args"]["REML"] else "ML", STRUCTURES[kind]]
    if categorical:
        labels.append(spec["contrasts"])
    if missing:
        labels.append("missingness")
    if "weights" in spec:
        labels.append("weights")
    if "offset(" in spec["formula"]:
        labels.append("offset")
    if boundary:
        labels.append("boundary")
    case = {"id": spec["id"], "module": "lmer", "option": "lmer / " + " / ".join(labels),
            "seed": 821000 + number, "spec": spec}
    if number % 10 == 0:
        # Every prediction experiment carries conditional, marginal and unseen levels.
        clean_rows = [j for j in range(n) if all(data[c][j] is not None for c in ("y", "x", "g"))][:4]
        newdata = {c: [data[c][j] for j in clean_rows] for c in data}
        unseen = {c: list(v) for c, v in newdata.items()}
        unseen["g"] = ["unseen"] * len(clean_rows)
        case["prediction_specs"] = [
            {**spec, "id": spec["id"] + "_conditional", "operation": "predict", "newdata": newdata},
            {**spec, "id": spec["id"] + "_population", "operation": "predict", "newdata": newdata, "population": True},
            {**spec, "id": spec["id"] + "_newlevels", "operation": "predict", "newdata": unseen, "allow_new_levels": True},
        ]
    return case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()
    destination = ROOT / "tests/golden/lmer"
    destination.mkdir(parents=True, exist_ok=True)
    for first in range(0, args.count, args.batch_size):
        cases = [make_case(i) for i in range(first, min(first+args.batch_size, args.count))]
        specs = [s for c in cases for s in [c["spec"], *c.get("prediction_specs", [])]]
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "input.json"
            output_path = Path(temporary) / "output.json"
            input_path.write_text(json.dumps({"cases": specs}))
            subprocess.run(["Rscript", str(ROOT / "oracle/run_case.R"), str(input_path), str(output_path)], check=True)
            observed = {o["id"]: o for o in json.loads(output_path.read_text())}
        for case in cases:
            case["oracle"] = observed[case["id"]]
            case["prediction_oracles"] = [observed[s["id"]] for s in case.get("prediction_specs", [])]
            if "error" in case["oracle"]["result"]:
                raise RuntimeError(f"{case['id']}: {case['oracle']['result']['error']}")
            (destination / (case["id"] + ".json")).write_text(json.dumps(case, indent=2) + "\n")
        print(f"generated {first + len(cases)} / {args.count}", flush=True)


if __name__ == "__main__":
    main()
