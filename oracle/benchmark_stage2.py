"""Stage 2 fit timings on one machine, excluding R process and package startup."""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd

from rparity import gam, glmmTMB

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "oracle/cache"


def fit(spec: dict[str, Any]) -> Any:
    """Time the public fit with the same data and statistical specification."""
    common = {"family": spec["family"], "link": spec["link"],
              "weights": spec.get("weights"), "offset": spec.get("offset")}
    data = pd.DataFrame(spec["data"])
    if spec["call"] == "gam":
        return gam(spec["formula"], data, method=spec["args"]["method"], **common)
    return glmmTMB(spec["formula"], data, ziformula=spec["ziformula"],
                   dispformula=spec["dispformula"], **common)


def main() -> None:
    """Record medians of three warm fits, without saving local machine names."""
    candidates = [("gam", "gam_0000"), ("gam", "gam_0015"), ("gam", "gam_0033"),
                  ("tmb", "tmb_0001"), ("tmb", "tmb_0068")]
    selected = [json.loads((ROOT / f"tests/golden/{module}/{name}.json").read_text())
                for module, name in candidates]
    specs = [dict(case["spec"], id=f"warmup_{index}", benchmark=True)
             for index, case in enumerate(selected)]
    specs.extend(dict(case["spec"], id=f"benchmark_{index}_{repeat}", benchmark=True)
                 for repeat in range(3) for index, case in enumerate(selected))
    CACHE.mkdir(exist_ok=True)
    input_file, output_file = CACHE / "benchmark-s2-input.json", CACHE / "benchmark-s2-output.json"
    input_file.write_text(json.dumps({"cases": specs}))
    subprocess.run(["Rscript", "oracle/run_case.R", str(input_file), str(output_file)],
                   cwd=ROOT, check=True)
    observed = {case["id"]: case for case in json.loads(output_file.read_text())}
    rows = []
    for index, case in enumerate(selected):
        fit(case["spec"])
        times = []
        for _ in range(3):
            start = time.perf_counter()
            fit(case["spec"])
            times.append(time.perf_counter() - start)
        r_times = []
        for repeat in range(3):
            result = observed[f"benchmark_{index}_{repeat}"]["result"]
            if "error" in result:
                raise RuntimeError(result["error"])
            r_times.append(result["fit_seconds"])
        rows.append({"id": case["id"], "option": case["option"],
                     "observations": len(next(iter(case["spec"]["data"].values()))),
                     "python_seconds": statistics.median(times),
                     "r_seconds": statistics.median(r_times)})
    report = {"repetitions": 3, "statistic": "median", "machine_architecture": platform.machine(),
              "python_version": platform.python_version(),
              "scope": "Warm public fit calls on the same machine; R startup excluded.",
              "models": rows}
    (ROOT / "reports/benchmark_stage2.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
