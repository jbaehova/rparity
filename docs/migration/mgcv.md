# mgcv migration

Import `gam` and `gam_check` from `rparity`. The examples assume an existing
data frame `d`, a fitted model `m`, and prediction data `nd`. A formula is a
Python string. Smooth terms keep the familiar `s()`, `te()`, and `ti()` syntax.
Use pandas categorical columns for factors and preserve their levels in new
data. Polars input is converted to pandas internally.

The runtime requires Python 3.11 or newer and runs independently of R and
`rpy2`. The installed wheel includes the native backend needed for cubic
shrinkage penalties; its license notices are packaged with the library.

The default smoothing method is `"GCV.Cp"`. Set `method` explicitly when
reproducing an R analysis. The supported methods select smoothness with GCV
or UBRE, marginal ML, or REML. See the [validation coverage](../coverage.md)
for the recorded numerical results and remaining limitations. Public R
behavior is specified by the [mgcv GAM manual](https://stat.ethz.ch/R-manual/R-patched/library/mgcv/html/gam.html).
Stage 2 is complete in v0.2.0. The fixed GAM corpus passes 1,184/1,200 cases;
its 16 remaining differences count as failures in the combined Stage 2 result
of 2,356/2,400 (98.17%). See the
[Stage 2 report](https://github.com/jbaehova/rparity/blob/main/reports/STAGE_2_REPORT.md)
and [case-level evidence](https://github.com/jbaehova/rparity/blob/main/reports/STAGE_2_FAILURES.json).

## Model specification

| Use | R | Python |
| --- | --- | --- |
| Thin plate smooth | `gam(y ~ s(x,bs="tp",k=8),data=d,method="REML")` | `gam("y ~ s(x,bs='tp',k=8)",d,method="REML")` |
| Cubic regression spline | `gam(y ~ s(x,bs="cr",k=8),data=d,method="REML")` | `gam("y ~ s(x,bs='cr',k=8)",d,method="REML")` |
| Shrinkage cubic spline | `gam(y ~ s(x,bs="cs",k=8),data=d,method="REML")` | `gam("y ~ s(x,bs='cs',k=8)",d,method="REML")` |
| P-spline | `gam(y ~ s(x,bs="ps",k=8),data=d,method="REML")` | `gam("y ~ s(x,bs='ps',k=8)",d,method="REML")` |
| Random intercept smooth | `gam(y ~ x+s(g,bs="re"),data=d,method="REML")` | `gam("y ~ x+s(g,bs='re')",d,method="REML")` |
| Random slope smooth | `gam(y ~ x+s(x,g,bs="re"),data=d,method="REML")` | `gam("y ~ x+s(x,g,bs='re')",d,method="REML")` |
| Tensor product | `gam(y ~ te(x,z,k=c(5,5)),data=d,method="REML")` | `gam("y ~ te(x,z,k=[5,5])",d,method="REML")` |
| Smooth interaction | `gam(y ~ s(x)+s(z)+ti(x,z,k=c(5,5)),data=d,method="REML")` | `gam("y ~ s(x)+s(z)+ti(x,z,k=[5,5])",d,method="REML")` |
| Continuous multiplier | `gam(y ~ s(x,by=z,k=8),data=d,method="REML")` | `gam("y ~ s(x,by=z,k=8)",d,method="REML")` |
| Smooth by factor level | `gam(y ~ a+s(x,by=a,k=8),data=d,method="REML")` | `gam("y ~ a+s(x,by=a,k=8)",d,method="REML")` |
| Additive smooths | `gam(y ~ s(x,k=8)+s(z,k=8),data=d,method="REML")` | `gam("y ~ s(x,k=8)+s(z,k=8)",d,method="REML")` |
| Gaussian | `gam(f,data=d,family=gaussian(),method="REML")` | `gam(f,d,family="gaussian",method="REML")` |
| Binomial logit | `gam(f,data=d,family=binomial(),method="REML")` | `gam(f,d,family="binomial",method="REML")` |
| Binomial counts | `gam(cbind(success,failure) ~ s(x,k=8),data=d,family=binomial(),method="REML")` | `gam("cbind(success,failure) ~ s(x,k=8)",d,family="binomial",method="REML")` |
| Binomial proportions | `gam(f,data=d,family=binomial(),weights=trials,method="REML")` | `gam(f,d,family="binomial",weights="trials",method="REML")` |
| Poisson log | `gam(f,data=d,family=poisson(),method="REML")` | `gam(f,d,family="poisson",method="REML")` |
| Gamma inverse link | `gam(f,data=d,family=Gamma(),method="REML")` | `gam(f,d,family="Gamma",method="REML")` |
| Gamma log link | `gam(f,data=d,family=Gamma(link="log"),method="REML")` | `gam(f,d,family="Gamma",link="log",method="REML")` |
| REML smoothness | `gam(f,data=d,method="REML")` | `gam(f,d,method="REML")` |
| ML smoothness | `gam(f,data=d,method="ML")` | `gam(f,d,method="ML")` |
| GCV or UBRE | `gam(f,data=d,method="GCV.Cp")` | `gam(f,d,method="GCV.Cp")` |
| Prior weights | `gam(f,data=d,weights=w,method="REML")` | `gam(f,d,weights="w",method="REML")` |
| Formula offset | `gam(y ~ s(x,k=8)+offset(o),data=d,method="REML")` | `gam("y ~ s(x,k=8)+offset(o)",d,method="REML")` |
| Fixed smoothing parameter | `gam(f,data=d,sp=c(0.1),method="REML")` | `gam(f,d,sp=[0.1],method="REML")` |

Here `f` is an R formula in the R column and the corresponding formula string
in the Python column. For a binomial proportion, `trials` gives the denominator
of each observation. Count responses retain both successes and failures.

The basis size `k` limits the available function space. Smoothing chooses
effective degrees of freedom within that space. A factor `by` term creates a
separate smooth for each level. Include a parametric factor term when the
level-specific means should also differ. A numeric `by` variable multiplies
the smooth's contribution by its value.

## Results and prediction

| Use | R | Python |
| --- | --- | --- |
| Coefficients | `coef(m)` | `m.coef()` |
| Coefficient EDF | `m$edf` | `m.edf` |
| Smooth EDF | `summary(m)$s.table[,"edf"]` | `m.smooth_edf` |
| Estimated smoothness | `m$sp` | `m.sp` |
| Scale estimate | `summary(m)$scale` | `m.scale` |
| Summary text | `summary(m)` | `print(m.summary())` |
| Parametric tests | `summary(m)$p.table` | `m.summary_tables()["parametric"]` |
| Approximate smooth tests | `summary(m)$s.table` | `m.summary_tables()["smooth"]` |
| Bayesian covariance | `vcov(m)` | `m.vcov()` |
| Frequentist covariance | `vcov(m,freq=TRUE)` | `m.vcov(freq=True)` |
| Fitted response | `fitted(m)` | `m.fitted()` |
| Deviance residuals | `residuals(m,type="deviance")` | `m.residuals(type="deviance")` |
| Pearson residuals | `residuals(m,type="pearson")` | `m.residuals(type="pearson")` |
| Link prediction | `predict(m,newdata=nd,type="link")` | `m.predict(nd,type="link")` |
| Response prediction | `predict(m,newdata=nd,type="response")` | `m.predict(nd,type="response")` |
| Prediction standard error | `predict(m,newdata=nd,se.fit=TRUE)` | `m.predict(nd,se_fit=True)` |
| Response standard error | `predict(m,newdata=nd,type="response",se.fit=TRUE)` | `m.predict(nd,type="response",se_fit=True)` |
| Term contributions | `predict(m,newdata=nd,type="terms")` | `m.predict(nd,type="terms")` |
| Selected smooth | `predict(m,newdata=nd,type="terms",terms="s(x)")` | `m.predict(nd,type="terms",terms=["s(x)"])` |
| Term standard errors | `predict(m,newdata=nd,type="terms",se.fit=TRUE)` | `m.predict(nd,type="terms",se_fit=True)` |
| Remove a smooth | `predict(m,newdata=nd,exclude="s(g)")` | `m.predict(nd,exclude=["s(g)"])` |
| Prediction matrix | `predict(m,newdata=nd,type="lpmatrix")` | `m.predict(nd,type="lpmatrix")` |
| Partial-effect data | `predict(m,newdata=nd,type="terms",terms="s(x)",se.fit=TRUE)` | `m.partial_effects("s(x)",newdata=nd)` |
| Numerical diagnostics | `gam.check(m,k.rep=200,k.sample=5000)` | `gam_check(m,k_rep=200,k_sample=5000)` |
| Seeded basis check | `set.seed(17); k.check(m,n.rep=200,subsample=5000)` | `gam_check(m,seed=17,k_rep=200,k_sample=5000)["k.check"]` |

Standard-error predictions return a dictionary with `"fit"` and `"se.fit"`
entries. Term predictions return a data frame whose columns identify model
terms. The intercept is stored in `fit.attrs["constant"]`; term contributions
do not include it. Standard errors condition on fitted smoothing parameters.
Response standard errors use the inverse-link delta method.

```python
from rparity import gam, gam_check

m = gam("y ~ a+s(x,bs='cr',k=8)", d, method="REML")
prediction = m.predict(nd, type="response", se_fit=True)
effects = m.partial_effects("s(x)", n=100)
effects["lower"] = effects["fit"] - 1.96 * effects["se"]
effects["upper"] = effects["fit"] + 1.96 * effects["se"]
diagnostics = gam_check(m, seed=17, k_rep=200, k_sample=5000)
print(diagnostics["k.check"])
```

`partial_effects()` returns data for your plotting library. With no `term`,
it returns a dictionary of smooth labels and data frames. An automatic grid
varies each smooth's first variable while other predictors stay at their
median or an observed factor level. Supply `newdata` to construct a full
two-dimensional tensor surface or to control the values of `by` variables.
The `fit` column is a link-scale term effect.

`gam_check()` returns convergence information and numerical residual
diagnostics. Its `"k.check"` table contains the available basis dimension,
smooth EDF, a nearest-neighbor residual variance ratio, and a permutation
p-value. Set the seed and replication count when reproducing a comparison.
Unsupported factor-coordinate checks yield missing index and p-value entries.
The residual vectors are available for user-defined plots. Interpretation of
the test follows the [public basis-check specification](https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/gam.check.html).

## Scope and numerical coordinates

Smooth coefficients depend on the basis coordinates. `coef()` and `vcov()`
describe rparity's own basis. Golden validation maps the independently built
basis into the observed R prediction-matrix coordinates before comparing
coefficients and covariance. Predictions, function-space tests, and total EDF
do not require identical column coordinates. REML criterion comparisons also
account for the measure on unpenalized coefficients through
`criterion_in_basis(transform)`.

One-dimensional thin plate bases reproduce the observed native coordinate
orientation using a deterministic Lanczos construction. Multidimensional
thin plate bases can use different native coordinates while representing the
same functions. A numeric `by` smooth can share an unpenalized direction with
a parametric term, for example `w+s(x,by=w)`. Individual coefficients and
term decompositions then depend on the identifying constraint. Compare the
identified coefficient space and the complete predictor; a REML comparison
must also specify the coefficient measure through
`criterion_in_basis(transform, constraint=normals)`.

The reported scale uses Fletcher's improved Pearson estimator by default.
Likelihood scale is estimated separately for RE/ML selection. In a weighted
Gamma fit, the selection likelihood treats weights as precision while
`logLik()` preserves the ordinary Gamma frequency-weight convention. This
distinction matters when reproducing fit criteria with unequal weights.

`AIC()` uses conditional likelihood and influence EDF. It currently omits the
smoothing-uncertainty correction that mgcv can apply for RE/ML selection.
See the [public AIC specification](https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/logLik.gam.html)
before comparing these AIC values. Boundary tests involving multiple
random-effect smooths have an observed statistic discrepancy outside the
seeded golden corpus; their approximate p-values need independent checking.

This module uses dense linear algebra and targets ordinary GAMs with modest
basis sizes. Above 2000 unique thin plate locations, it chooses an evenly
indexed subset; this differs from mgcv's knot-sampling convention and has no
claim of native-coordinate parity. The Stage 2 scope does not include `bam`,
`gamm`, extra response families, or `bs="fs"`. Shared smooth parameters
through `id`, custom knots, and smoothing-selection uncertainty in prediction
are unavailable. Numerical
parity is reported per case in the coverage evidence.

Cubic shrinkage uses a bundled OpenBLAS eigensolver to preserve the very small
positive penalty eigenvalues across SciPy backends. It does not call R. The
native library and its dependency notices are included in platform wheels;
the Python implementation remains MIT licensed. OpenBLAS is supplied during
the build; the wheel retains the five declared Python runtime dependencies.
The current clean-wheel
execution check covers macOS ARM64. Other wheel platforms still need execution
validation.
