# Stage 2 distribution verification

The final v0.2.0 wheel and source distribution passed local metadata, native
library, license, and clean-install checks. An isolated Python 3.11 rebuild
from the final source archive produced a byte-identical wheel. No GitHub
Actions test, PyPI upload, or R process was used for these checks.

## Final artifacts

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `rparity-0.2.0-py3-none-macosx_11_0_arm64.whl` | 6,911,059 | `ea945810b6810fee2082bc34548a3d0e8daac044703098486822423e6401318d` |
| `rparity-0.2.0.tar.gz` | 135,963 | `0f30fdd4ce7444c4ce60c58072accbec782f4dcd34a6b467b248516be103c6d2` |

Both archives declare version `0.2.0` and Python `>=3.11`. The wheel declares
`Root-Is-Purelib: false` and tag `py3-none-macosx_11_0_arm64`. All four bundled
Mach-O libraries contain the ARM64 architecture, including the single-slice
fat container used by `libgcc_s.1.1.dylib`.

The wheel and source archive include the final README in their package
metadata. Its SHA-256 is
`1f9395211f632a1e7b8015c32d983682a1079e05631557cd9fb484598885f572`.
Each archive declares exactly five direct runtime requirements:

| Requirement | Version in the clean Python 3.11 install |
| --- | --- |
| `numpy>=2.0` | 2.4.6 |
| `scipy>=1.14` | 1.17.1 |
| `pandas>=2.2` | 3.0.6 |
| `formulaic>=1.1` | 1.2.2 |
| `statsmodels>=0.14` | 0.15.0 |

Neither R, rpy2, nor scipy-openblas32 is a runtime requirement. Polars remains
optional and was absent from these clean environments.

## R-free installed execution

Two separate Python 3.11.15 virtual environments were created with `uv venv`.
One installed the release wheel. The other installed the independently
rebuilt wheel. Both contained the runtime dependencies and their transitive
dependencies, without an installed scipy-openblas32 provider or rpy2 package.
The imported rparity package came from the installed wheel in `site-packages`.

Before package import, the smoke script emptied `PATH` and removed `R_HOME`.
It checked that neither `R` nor `Rscript` could be located. A Python audit
hook rejected process-launch events, and subprocess entry points were
replaced by a rejecting guard. The guard's explicit rejection self-test
passed. Model fitting and inference then made zero process-launch attempts.

Both installations completed the following public API checks with no runtime
warnings:

- Both Python quick-start blocks in the final README, including the Stage 1
  correlated mixed model and the Stage 2 GAM and Poisson mixed model.
- A REML cubic shrinkage GAM using `bs='cs'`, with finite coefficients and
  covariance, response and term predictions with standard errors, smooth
  summary, partial effects, `gam_check()` and marginal means.
- All six glmmTMB families: Gaussian, Poisson, negative binomial 1, negative
  binomial 2, binomial and beta. Each checked coefficients, covariance,
  likelihood criteria, dispersion, fitted values, residuals, prediction,
  model matrix, summary, Type II Anova, marginal means and pairwise contrasts.
- A Poisson random-intercept model with covariate-dependent zero inflation,
  including random covariance and modes, the response mixture identity,
  component Type III Anova, conditional response-scale marginal means and
  zero-inflation probabilities.

The numerical backend loaded from `rparity/_openblas_libs` without searching
for a separately installed provider. Every installed native file matched the
SHA-256 and size recorded in the bundled manifest.

## Cubic shrinkage reproducibility

The cubic shrinkage penalty was bit-identical across the Python 3.12.13
checkout and both Python 3.11.15 wheel installs. The penalty SHA-256 was
`8e55dd8b237514f49b4679033329711eac9dadf7d9b7b5109c9099ae4ae3976a`.

| Comparison | Coefficients, maximum absolute difference | Predictions, maximum absolute difference | Prediction SE, maximum absolute difference |
| --- | ---: | ---: | ---: |
| Python 3.12 checkout versus Python 3.11 release wheel | 4.44e-16 | 4.44e-16 | 2.08e-17 |
| Release wheel versus isolated source-archive rebuild | 0 | 0 | 0 |

The independently rebuilt wheel also had the release wheel's exact bytes,
size and SHA-256. This checks the final source archive's build inputs as well
as installed execution.

## Bundled notices and build inputs

rparity's own code retains its MIT license. The wheel includes its project
license under `rparity-0.2.0.dist-info/licenses/LICENSE`. General numerical
libraries are vendored unmodified from the pinned build dependency
`scipy-openblas32==0.3.34.237.0`:

- `scipy_openblas32/lib/libscipy_openblas.dylib`
- `scipy_openblas32/.dylibs/libgcc_s.1.1.dylib`
- `scipy_openblas32/.dylibs/libgfortran.5.dylib`
- `scipy_openblas32/.dylibs/libquadmath.0.dylib`

These paths live below the wheel's `rparity/_openblas_libs` directory. The
manifest records package-relative paths and the original provider version.
The provider's complete retained notice matches
[`LICENSES/scipy-openblas32-LICENSE.txt`](../LICENSES/scipy-openblas32-LICENSE.txt)
byte for byte. It includes OpenBLAS and LAPACK BSD terms, the GCC runtime
exception and associated GPL text, and libquadmath terms. The complete
[`GNU LGPL 2.1 text`](../LICENSES/GNU-LGPL-2.1.txt) is bundled beside it.
See the [notice explanation](../LICENSES/README.md) for upstream source links.

The source archive includes `hatch_build.py`, the pinned build requirement,
all Python implementation modules, the README and all retained license
files. It contains neither oracle caches nor R files. The wheel contains
neither R files nor an rpy2 implementation.

## Local commands and scope

The following build and installation commands succeeded. Build isolation
was enabled for both wheel constructions; no global Python environment was
modified.

```sh
uv build --clear
uv venv --python 3.11 oracle/cache/s2-final-wheel-venv
uv pip install --python oracle/cache/s2-final-wheel-venv/bin/python \
  dist/rparity-0.2.0-py3-none-macosx_11_0_arm64.whl
uv build --wheel --python 3.11 \
  --out-dir oracle/cache/s2-final-wheel-rebuilt --no-create-gitignore \
  dist/rparity-0.2.0.tar.gz
uv venv --python 3.11 oracle/cache/s2-final-wheel-rebuilt-venv
uv pip install --python oracle/cache/s2-final-wheel-rebuilt-venv/bin/python \
  oracle/cache/s2-final-wheel-rebuilt/rparity-0.2.0-py3-none-macosx_11_0_arm64.whl
```

Local audit and smoke helpers and their JSON evidence are retained in the
ignored `oracle/cache/s2-final-wheel-*` paths. Each installed environment ran
`s2-final-wheel-smoke.py`; `s2-final-wheel-audit.py` checked both archives and
native hashes, and `s2-final-wheel-compare.py` checked the numerical and byte
comparisons reported above.

Execution evidence covers macOS ARM64 only. The wheel's macOS 11 minimum tag
comes from the pinned provider metadata; execution on every macOS version is
not claimed. Linux and Windows provider layouts were inspected during
development, but neither platform was executed in this release audit. The
native wheel must be built and verified on the target platform before
claiming equivalent execution coverage there.
