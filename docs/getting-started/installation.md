---
title: "Install rparity for Python statistical modeling"
description: Install rparity from a GitHub release wheel or source checkout. Learn Python requirements, supported platform checks and how to run without R or rpy2.
---

# Installation

Use **Python 3.11 or newer**. You do not need R or `rpy2` to fit models,
predict or run inference.

## Install a release wheel

Download the wheel for your platform from the
[latest release](https://github.com/jbaehova/rparity/releases/latest).
The published wheel has been checked on **macOS ARM64**, including Apple
Silicon Macs. In the directory containing the downloaded wheel, run:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install ./rparity-*.whl
```

Select a compatible wheel rather than renaming a file for another platform.
If there is no wheel for your platform, use a source installation and check
your intended analysis locally. Execution on other platforms has not yet
been validated.

## Install from source

```sh
git clone https://github.com/jbaehova/rparity.git
cd rparity
python -m pip install .
```

This installs the library into your current Python environment. To isolate
it, create and activate a virtual environment first. Package builds obtain
the native OpenBLAS backend used for cubic shrinkage smooths; its libraries
and license notices are included in wheels.

## Check your installation

```sh
python -c "from rparity import lmer, gam, glmmTMB, emmeans; print('rparity is ready')"
```

The five declared Python dependencies are NumPy, SciPy, pandas, formulaic and
statsmodels. pandas data frames work throughout the public API. Polars input
is also accepted and converted to pandas internally; install Polars separately
if you use it.

Continue with the [quickstart](quickstart.md), or run an analysis from the
repository's [examples directory](https://github.com/jbaehova/rparity/tree/main/examples).
