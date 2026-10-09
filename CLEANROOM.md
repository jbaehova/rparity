# Clean-room record

## Stage 1 implementation

No target R package source or function body was opened, read, or searched during this stage. The Python implementation derives from the statistical papers and textbooks recorded in REFERENCES.md, with permissively licensed NumPy, SciPy, formulaic and statsmodels dependencies.

Every numerical R observation passed through `oracle/run_case.R`. The oracle calls exported fitting, inference and prediction functions with explicit contrasts, optimizer controls and degrees-of-freedom options. It extracts public coefficients, covariance matrices, likelihoods, conditional modes, model summaries, warnings and inference tables. Function bodies and internal source files are not part of this interface.

Seeded synthetic inputs and public observations form the committed golden corpus. Precision attempts are observations rather than implementation templates. R's built-in example datasets remain only in ignored `oracle/cache/`; their tests require the development R installation. Package and R versions are recorded in `oracle/versions.json`.

The release wheel contains Python runtime modules and requires no R executable or rpy2. The same public oracle supplies representative-example checks and same-machine fit-time measurements. Required-field discrepancies remain visible in the numerical coverage and Stage 1 report.

## Post-release precision refinement

LMER stationary-score roots, GLMER implicit-mode score differentiation and GLS
factor/QR likelihood evaluation derive from the recorded mathematical references.
R was queried only through oracle/run_case.R for public fit outputs and optional
numerical optimizer gradient/Hessian fields. No target source or function body
was inspected. Original synthetic inputs, oracle results and comparison tolerances
were retained, including cases where tighter uniform R controls failed.

## Stage 2 GAM and distributional mixed models

No target R package source files or function bodies were intentionally
inspected for Stage 2 implementation. Mathematical methods were independently
implemented from the GAM and Laplace references listed in REFERENCES.md.
Public help pages and published papers supplied behavioral and mathematical
specifications. Their prose was not copied into docstrings or migration guides.

One external web-search response incidentally included a glmmTMB source
snippet. The source page was not opened, and the snippet was not used as an
implementation reference. Subsequent numerical questions were routed through
the oracle. This exception to a completely source-free search record is
recorded explicitly rather than omitted from the audit.

All numerical R fitting, inference, prediction, and diagnostic observations
pass through `oracle/run_case.R`. Explicit options select GAM smoothness
methods and diagnostic seeds. The oracle extracts public prediction matrices,
smooth penalties, coefficients, EDF, covariance, and fit criteria. Public
numerical spline objects are used to verify independently derived function
spaces and penalties. A change of coordinates maps Python and R bases for
coefficient comparison; the R matrix is not used by runtime fitting.

One-dimensional thin plate coordinates were reconstructed from public
`mgcv::slanczos()` results and fitted numerical spline objects. The inferred
starting sequence uses the independently specified linear congruential
recurrence `state = (106 * state + 1283) % 6075` from initial state 1.
Full reorthogonalization, tridiagonal eigensolves, and coordinate normalization
were implemented using mathematical references and SciPy. The public help
page describes the Lanczos method; no target function body was read.
The audit separates function-space equivalence from equality of native
coefficient coordinates. Multidimensional thin plate native coordinates can
still differ from the observed R coordinates.

Weighted Gamma smoothing selection and public conditional likelihood are
checked separately. Their precision-weight and frequency-weight densities
were independently evaluated with normalized Gamma formulas. Fletcher's
reporting scale remains separate from likelihood scale. For designs with a
joint design/penalty null space, coefficient comparisons use only identified
directions and REML records the selected coefficient measure. Numerical
observations establish these conventions; no R family function was inspected.

For mixed models, the oracle records all three fixed components and their
joint covariance. It also observes normalized likelihoods, conditional modes,
random covariance, component predictions, marginal means, and Wald tables.
Gaussian dispersion parameterization is checked against public results from
the recorded package version. No implementation of R's optimizer or TMB's
automatic differentiation is inspected or embedded in the Python package.

Mixed-model observed information is evaluated after a mathematical conversion
to the documented log-SD and scaled Cholesky coordinates. Centered score
differences use an unscaled `1e-3` increment. The public `stats::optim` manual
documents this default step, and black-box covariance observations establish
the matching convention. No source linked from that manual or the public
glmmTMB covariance vignette was opened.

Reference precision is refined through explicit optimizer controls and an
independent stationary-score Newton callback. The callback uses only the
objective and gradient supplied by the documented `glmmTMBControl` optimizer
interface. Additional starts use public `beta`, `betazi`, `betadisp`, and
`theta` arguments. Their likelihood improvements are independently confirmed
by R fits before becoming a reference refinement. Overrides are reproducible
from `oracle/stage2_overrides.json`; earlier specifications, outputs, and
warnings remain in each fixture's `oracle_attempts`. A revised stationary
reference does not remove the original failure observation or relax a
comparison tolerance.

The basis-dimension check uses public residual diagnostics and seeded
permutation observations. R seed states and sampled permutations are numerical
black-box observations, obtained through the same oracle. The Python
diagnostic implementation uses no R source or runtime R process.

Development R packages were installed through their public package installer.
Installer compilation output was not treated as implementation source.
Versions are recorded in `oracle/versions.json`. Synthetic inputs and oracle
results remain auditable in the golden corpus, including retained attempts
when a generator input requires a documented domain correction.

Tests run locally. The test workflow requires manual dispatch and is not
triggered for Stage 2 validation. Documentation deployment is limited to
changes affecting the documentation. Runtime GAM and mixed-model code uses
Python dependencies only; R remains a development oracle.

Two further delegated web-search responses incidentally returned target-source
snippets while investigating mixed-model covariance. Neither linked source
page was opened, and the workers reported that these snippets were not used
as implementation references. Subsequent probes used public numerical fields
through the oracle. These search incidents are retained in the audit record.

Cubic regression penalties were independently reconstructed from the natural
spline tridiagonal equations. Public numeric `smoothCon` matrices, knot
quantiles and hexadecimal floating-point outputs established operation-order
and single-rounding checks. The implementation uses an independently written
fused-contraction calculation and an OpenBLAS symmetric eigensolver, without
opening an mgcv function body. A build hook bundles the permissively licensed
OpenBLAS provider and its native dependencies. Third-party GCC runtime and
LGPL libquadmath notices are preserved in `LICENSES` and the distributions;
the MIT license does not replace those third-party terms.

Individual vanishing-penalty comparisons use fresh constrained refits of the
same independently specified likelihood. Convex-score and Schur-complement
bounds verify only the collapsed positive-penalty directions. A joint
design/penalty gauge is removed before computing these bounds, with its
nominal REML dimension and oblique-slice measure retained. Gaussian
zero-variance certificates prove a nonnegative objective derivative over
the whole variance axis. Identified fields remain subject to the original
comparison tolerances, and adverse-observation tests reject invalid inputs.

Twenty additional GAM references use documented public smoothing starts or
stronger controls. Lower minima or improved score and augmented-QR consistency
are independently verified. All prior specifications and observations remain
in the fixtures. Failed precision probes are retained separately rather than
being counted as passing references.

Mixed-model correlation boundaries are checked through public numeric Hessians
and independent, documented `rr(d=1)` auxiliary fits. The latter represent
the same physical rank-one random covariance in finite loading coordinates.
They supply identified limiting covariance and, for explicitly indexed fields,
predictions. Their reduced parameter count never replaces the original
unstructured model's AIC, BIC or residual degrees of freedom. Every original
observation and all auxiliary inputs and outputs are retained. Runtime fitting
does not receive these R matrices or reference coefficients.

Weak component information is also examined after profiling all nuisance
coordinates. This independently derived Schur-complement diagnostic detects
joint component contrasts that positive information diagonals can miss. It
changes warning diagnostics only. Quadratic likelihood and orthogonal-change
tests establish its joint-direction behavior without an R implementation
reference.

Public example reproduction exposed repeated-covariate knot selection: the
cubic basis must take type-7 quantiles over unique covariate values. Public
numeric spline objects verified the correction without a target source view.
The representative mcycle and Salamanders inputs stay in development cache.

Support-preserving GAM penalty roots factor each original smooth block before
rotating to the fitting space, preventing amplified rounding leakage between
independent blocks. Exact rank-one TMB tangents remove quadratic score-step
error by Richardson extrapolation. Independent public R loading-coordinate
Hessians verify consecutive refined covariance stability. Interior NB1
information refinement likewise records all original numeric Hessians and
requires independently stable positive information. Boundary dispersion
comparisons use separately derived normalized NB density and curvature bounds,
validated by 80-digit arithmetic. These are mathematical implementations and
numerical black-box observations; no target function body was read.
