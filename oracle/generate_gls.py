"""Generate independently seeded synthetic GLS black-box golden fixtures.

Run ``uv run python oracle/generate_gls.py``. All R outputs are obtained through
oracle/run_case.R. No package datasets or R implementation code are copied.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.linalg import toeplitz
from statsmodels.tsa.arima_process import arma_acovf


def make_case(index: int) -> dict:
    """Construct one reproducible design and Gaussian response."""
    rng = np.random.default_rng(734_119 + index)
    family = index % 10
    n_groups = int(rng.integers(10, 19))
    dimension = 3 if family == 3 else int(rng.integers(3, 6))
    sizes = rng.integers(2, dimension + 1, size=n_groups)
    sizes[0] = dimension
    groups = np.repeat(np.arange(n_groups), sizes)
    times = np.concatenate([np.arange(1, size + 1) for size in sizes])
    n = len(groups)
    x = rng.normal(0.3, 1.3, n)
    f = np.where(groups % 2 == 0, "a", "b")
    fixed = index % 4 == 0
    rho = float(rng.uniform(0.15, 0.55))
    power = float(rng.uniform(0.12, 0.4))
    correlation = None
    variance = None
    name = "independent"
    if family in {1, 8}:
        name = "corAR1"
        correlation = {"name": name, "args": {"form": "~ t | g", "value": rho, "fixed": fixed}}
    elif family in {2, 9}:
        name = "corCompSymm"
        correlation = {"name": name, "args": {"form": "~ 1 | g", "value": rho, "fixed": fixed}}
    elif family == 3:
        name = "corSymm"
        correlation = {
            "name": name,
            "args": {"form": "~ t | g", "fixed": fixed, "value": [0.2, 0.1, 0.3]},
        }
    elif family == 4:
        name = "corARMA"
        p, q = [(1, 1), (2, 0), (0, 1), (0, 2), (1, 0)][(index // 10) % 5]
        value = {
            (1, 1): [0.45, 0.35],
            (2, 0): [0.4, 0.15],
            (0, 1): [0.45],
            (0, 2): [0.3, 0.15],
            (1, 0): [0.5],
        }[(p, q)]
        correlation = {
            "name": name,
            "args": {"form": "~ t | g", "p": p, "q": q, "value": value, "fixed": fixed},
        }
    if family in {5, 8}:
        variance = {"name": "varIdent", "args": {"form": "~ 1 | f"}}
        name += " + varIdent" if family == 8 else "varIdent"
    elif family == 6:
        variance = {"name": "varPower", "args": {"form": "~ t", "value": power}}
        name = "varPower"
    elif family in {7, 9}:
        variance = {"name": "varExp", "args": {"form": "~ t", "value": power / 2}}
        name += " + varExp" if family == 9 else "varExp"
    sd = np.ones(n)
    if variance:
        if variance["name"] == "varIdent":
            sd = np.where(f == "a", 1.0, rng.uniform(1.3, 2.0))
        elif variance["name"] == "varPower":
            sd = times**power
        else:
            sd = np.exp(times * power / 2)
    residual = np.zeros(n)
    for group in range(n_groups):
        block = np.flatnonzero(groups == group)
        time = times[block]
        if correlation is None:
            cov = np.eye(len(block))
        elif correlation["name"] == "corAR1":
            cov = rho ** np.abs(time[:, None] - time)
        elif correlation["name"] == "corCompSymm":
            cov = np.full((len(block), len(block)), rho)
            np.fill_diagonal(cov, 1)
        elif correlation["name"] == "corSymm":
            cov = np.array([[1, 0.2, 0.1], [0.2, 1, 0.3], [0.1, 0.3, 1]])[
                : len(block), : len(block)
            ]
        else:
            args = correlation["args"]
            ar = args["value"][: args["p"]]
            ma = args["value"][args["p"] :]
            acov = arma_acovf(np.r_[1, -np.array(ar)], np.r_[1, ma], nobs=len(block))
            cov = toeplitz(acov / acov[0])
        cov = sd[block, None] * cov * sd[None, block]
        residual[block] = rng.multivariate_normal(np.zeros(len(block)), cov)
    y = 3.5 + 1.2 * x + 0.7 * (f == "b") + rng.uniform(0.45, 0.9) * residual
    method = "ML" if (index // 10) % 2 else "REML"
    spec = {
        "id": f"gls_{index:04d}",
        "call": "gls",
        "formula": "y ~ x + f",
        "data": {
            "y": y.tolist(),
            "x": x.tolist(),
            "f": f.tolist(),
            "t": times.tolist(),
            "g": groups.tolist(),
        },
        "factors": ["f"],
        "args": {"method": method},
    }
    if correlation:
        spec["correlation"] = correlation
    if variance:
        spec["variance"] = variance
    # A uniform public optimizer improves convergence precision for tiny predictions.
    # Record it for every input, including retained boundary failures.
    spec["gls_control"] = {
        "opt": "optim", "optimMethod": "BFGS", "msTol": 1e-14, "msMaxIter": 3000
    }
    return {"id": spec["id"], "module": "gls", "option": f"gls / {name} / {method}", "spec": spec}


def main() -> None:
    """Write all cases and preserve any oracle failures as evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=600)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cache = root / "oracle/cache"
    cache.mkdir(parents=True, exist_ok=True)
    cases = [make_case(i) for i in range(args.count)]
    input_file, output_file = cache / "gls_input.json", cache / "gls_output.json"
    input_file.write_text(json.dumps({"cases": [case["spec"] for case in cases]}))
    subprocess.run(
        ["Rscript", str(root / "oracle/run_case.R"), str(input_file), str(output_file)],
        check=True,
        cwd=root,
    )
    outputs = json.loads(output_file.read_text())
    target = root / "tests/golden/gls"
    target.mkdir(parents=True, exist_ok=True)
    failures = 0
    for case, output in zip(cases, outputs, strict=True):
        case["oracle"] = output
        failures += "error" in output["result"]
        (target / f"{case['id']}.json").write_text(json.dumps(case, indent=2) + "\n")
    print(f"Generated {len(cases)} GLS fixtures; oracle errors: {failures}")


if __name__ == "__main__":
    main()
