# Algorithm references

## Gaussian mixed models and formula blocks

Bates, Mächler, Bolker, and Walker (2015), *Fitting Linear Mixed-Effects Models
Using lme4*, Journal of Statistical Software 67(1).
[Publication](https://doi.org/10.18637/jss.v067.i01).
The implementation uses the relative covariance parameterization in equation
(4), profiled ML in equation (34), REML in equations (39) through (41), fixed
coefficient covariance in Section 5.1.1, and conditional random covariance in
Section 5.1.4. Dense marginal covariance algebra is mathematically equivalent
to the penalized least-squares formulation. Formulaic supplies the fixed
model matrices under its permissive license.

## Generalized mixed models

Ly, Christensen, Bates, Mächler, and Bolker (2026), *Fitting Generalized Linear
Mixed-Effects Models using lme4*.
[Public manuscript](https://arxiv.org/html/2607.12184v1).
Conditional modes use Newton updates from Appendix 6.1, including a stable
cloglog score. The standardized random effects follow Section 2.5, equations
(15) and (16). The Laplace likelihood follows Section 2.6, equations (18) and
(19). Expected Fisher weights for noncanonical links follow Appendix 6.2.
Python implements conditional modes and the
log-determinant independently using SciPy. Binomial normalization retains
trial counts; Poisson normalization retains log-factorial terms.

## Small-sample inference

Satterthwaite (1946), *An Approximate Distribution of Estimates of Variance
Components*, Biometrics Bulletin 2, 110-114. The denominator degrees of freedom
use the variance-of-variance delta method, including the observed parameter
information and derivatives of fixed-effect covariance.

Fai and Cornelius (1996), *Approximate F-tests of Multiple Degree of Freedom
Hypotheses in Generalized Least Squares Analyses of Unbalanced Split-Plot
Experiments*, Journal of Statistical Computation and Simulation 54, 363-378.
The multivariate denominator combines orthogonal one-dimensional contrasts.

Kenward and Roger (1997), *Small Sample Inference for Fixed Effects from
Restricted Maximum Likelihood*, Biometrics 53, 983-997, equations (4) through
(8). [Public article](https://people.math.aau.dk/~rw/Undervisning/Mat6F12/Handouts/kenwardroger97.pdf).
The implementation computes covariance bias adjustment and F-moment scaling
from linear marginal covariance components and expected REML information.

## ANOVA hypotheses

Kuznetsova, Brockhoff, and Christensen (2017), *lmerTest Package: Tests in
Linear Mixed Effects Models*, JSS 82(13), Appendix B.
[Publication](https://www.jstatsoft.org/article/view/v082i13).
Type I hypotheses use fixed-design QR and Forward-Doolittle row normalization.
Type II hypotheses apply categorical containment with identical continuous
factors. Type III uses marginal factor contrasts. The public `show_tests`
output was observed numerically to validate the independently derived bases.

Fox and Weisberg (2019), *An R Companion to Applied Regression*, third edition,
chapters on linear and generalized linear models. Type II/III car-style Wald
hypotheses retain the supplied coding; LR tests fit the reduced GLM.
[Behavioral reference](https://cran.r-project.org/web/packages/car/refman/car.html#Anova).
No implementation prose or package source was copied.

## Marginal means and contrasts

Searle, Speed, and Milliken (1980), *Population Marginal Means in the Linear
Model: An Alternative to Least Squares Means*, The American Statistician
34, 216-221. Estimates are averaged prediction rows, and their covariance is
obtained by applying those linear functions to fitted coefficient covariance.

Lenth's public documentation is used to specify reference grids, weighting,
contrast direction, multiplicity families, inverse links, regridding, and
trend definitions:
[Reference grids](https://rvlenth.github.io/emmeans/reference/ref_grid.html),
[contrast families](https://rvlenth.github.io/emmeans/reference/emmc-functions.html),
[comparisons](https://rvlenth.github.io/emmeans/articles/comparisons.html),
[model-specific degrees of freedom](https://rvlenth.github.io/emmeans/articles/models.html).
Sidak, Holm, Bonferroni, and Benjamini-Hochberg adjustments use their standard
probability/rank formulas; Tukey adjustment uses SciPy's studentized range.
GLS marginal-means degrees of freedom use the same variance delta method as
Satterthwaite above, applied to the unprofiled covariance-parameter likelihood.

## Generalized least squares

Pinheiro and Bates (2000), *Mixed-Effects Models in S and S-PLUS*, Chapter 5,
provides the correlated-error Gaussian likelihood, REML, variance functions,
and correlation structures. Pinheiro and Bates (1996), *Unconstrained
Parametrizations for Variance-Covariance Matrices*, Statistics and Computing
6, 289-296, supplies positive-definite covariance parameterization principles.

Public specifications for constructor behavior and interval/reporting options:
[GLS](https://stat.ethz.ch/R-manual/R-patched/library/nlme/html/gls.html),
[GLS controls](https://stat.ethz.ch/R-manual/R-devel/library/nlme/html/glsControl.html).
ARMA covariance is obtained from stationarity/invertibility-preserving partial
correlation coordinates and standard autocovariance recurrences. Variance
multipliers and group-specific scale ratios are independently evaluated.

## GAM bases and identifiability

Wood (2017), *Generalized Additive Models: An Introduction with R*, second
edition, Chapters 5 (*Smoothers*) and 6 (*GAM theory*), provides the common
penalized-regression framework and basis constraints.
[Publisher and contents](https://www.routledge.com/Generalized-Additive-Models-An-Introduction-with-R-Second-Edition/Wood/p/book/9781315370279).
Natural cubic regression splines use second-derivative quadratic penalties.
The shrinkage variant replaces the penalty's zero eigenvalues with positive
values. Factor random-effect smooths use indicator columns and an identity
penalty. Identifiability constraints are absorbed by matrix transformations.

Wood (2003), *Thin plate regression splines*, JRSS-B 65(1), 95-114.
[Publication](https://doi.org/10.1111/1467-9868.00374).
Thin plate bases combine a truncated radial-kernel eigenspace with the
unpenalized polynomial space. Python independently forms this eigenspace
using SciPy and preserves its associated quadratic penalty.

Demmel (1997), *Applied Numerical Linear Algebra*, supplies the symmetric
Lanczos construction. The public
[slanczos specification](https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/slanczos.html)
describes full reorthogonalization and LAPACK tridiagonal eigenproblems.
For one-dimensional thin plate bases, oracle observations identified a
deterministic starting sequence with `state = (106 * state + 1283) % 6075`,
initial state 1, and entries `2 * state / 6075 - 1`. The general recurrence
is the linear congruential method described in this
[numerical-methods lecture](https://farside.ph.utexas.edu/teaching/329/lectures/node107.html).
Its use and orientation in the spline calculation were inferred from public
numeric eigenvectors, iteration counts, and spline objects, rather than
target source. Python retains a dense eigenspace for accuracy and uses
independently implemented Lanczos, Householder QR, and RMS normalization to
choose its one-dimensional coordinates.

Eilers and Marx (1996), *Flexible smoothing with B-splines and penalties*,
Statistical Science 11(2), 89-121, supplies the coefficient-difference
penalty used by the P-spline basis.
[Publication](https://doi.org/10.1214/ss/1038425655).
SciPy constructs B-splines; finite differences form the penalty.

Wood (2006), *Low-Rank Scale-Invariant Tensor Product Smooths for Generalized
Additive Mixed Models*, Biometrics 62(4), 1025-1036.
[Publication](https://doi.org/10.1111/j.1541-0420.2006.00574.x).
Tensor design rows are products of marginal bases, and each marginal penalty
is extended with identity factors. Interaction smooths constrain the margins
before forming the product. Numeric multipliers and factor-level smooth
blocks follow the public [GAM model specification](https://stat.ethz.ch/R-manual/R-patched/library/mgcv/html/gam.models.html).

## GAM smoothing, scale, inference, and checks

Wood (2011), *Fast stable restricted maximum likelihood and marginal
likelihood estimation of semiparametric generalized linear models*, JRSS-B
73(1), 3-36.
[Publication](https://doi.org/10.1111/j.1467-9868.2010.00749.x).
The fitting criterion combines converged penalized likelihood with a Laplace
log-determinant correction. REML retains the unpenalized coefficient measure;
ML integrates the penalized subspace. Smoothing parameters are optimized
outside penalized IRLS. Coordinate changes transform both design and penalty,
including the additive REML measure constant. GCV and UBRE formulas are
specified in the [public GAM manual](https://stat.ethz.ch/R-manual/R-patched/library/mgcv/html/gam.html).
When a numeric `by` smooth shares an unpenalized direction with an explicit
parametric predictor, the design and every penalty identify a joint null
space. Coefficient comparisons then use the identifiable quotient. REML
also depends on the coefficient slice and its integration measure;
`criterion_in_basis(transform, constraint=...)` makes that choice explicit.

For weighted Gamma RE/ML selection, the exponential-dispersion likelihood
has observation shape `weight / phi` and scale `mu * phi / weight`.
Public conditional `logLik()` instead retains the ordinary Gamma family's
frequency-weight convention, multiplying each log density of shape
`1 / phi` by its weight. These distinct normalized likelihoods were checked
against public numerical oracle outputs and independently against SciPy
Gamma densities. Neither likelihood is substituted for the other.

Fletcher (2012), *Estimating overdispersion when fitting a generalized linear
model to sparse data*, Biometrika 99(1), 230-237.
[Publication](https://doi.org/10.1093/biomet/asr083).
The fitted GAM reporting scale uses the improved Pearson estimator. The
[public scale-estimation specification](https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/gam.scale.html)
also defines the Pearson and deviance alternatives. Likelihood estimation
scale and reporting scale are kept separate where the model requires it.
The improved reporting estimator divides Pearson dispersion by
`1 + mean(V'(mu) * (y - mu) / V(mu))`, with `V` the family variance function.

Wood, Pya, and Saefken (2016), *Smoothing parameter and model selection for
general smooth models*, JASA 111, 1548-1575, describes a smoothing-uncertainty
correction for conditional AIC.
[Publication](https://doi.org/10.1080/01621459.2016.1180986),
[public AIC specification](https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/logLik.gam.html).
The current Python `GamResult.AIC()` uses conditional influence EDF and an
estimated-scale parameter count; it does not implement that correction.

Wood (2013), *On p-values for smooth components of an extended generalized
additive model*, Biometrika 100(1), 221-228.
[Publication](https://doi.org/10.1093/biomet/ass048).
Approximate smooth tests operate in fitted function space, with Bayesian
coefficient covariance and a rank selected from influence degrees of freedom.
Wood (2013), *A simple test for random effects in regression models*,
Biometrika 100(4), 1005-1010, supplies the separate quadratic-form test for
fully penalized terms and a variance-component null on the boundary.
[Publication](https://doi.org/10.1093/biomet/ast038).
Prediction variances apply linear functions to fitted covariance; response
variances use inverse-link derivatives. Reporting follows the
[public summary specification](https://stat.ethz.ch/R-manual/R-patched/library/mgcv/html/summary.gam.html).

Wood (2017), Section 5.9, and the
[public basis-check manual](https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/gam.check.html)
specify nearest-neighbor residual differences for the k-index and residual
permutations for its null distribution. Seeded permutations are validated
through public numerical R observations. No target RNG source is used.

## Distributional mixed models and normalized Laplace likelihood

Kristensen, Nielsen, Berg, Skaug, and Bell (2016), *TMB: Automatic
Differentiation and Laplace Approximation*, JSS 70(5), 1-21, Section 2,
equations (1) through (5).
[Publication](https://doi.org/10.18637/jss.v070.i05),
[Public mathematical manuscript](https://arxiv.org/html/1509.00660#S2).
The conditional mode minimizes the joint negative log likelihood. The
Laplace approximation adds half the observed random-effect Hessian log
determinant and retains normalizing constants. Parameter covariance is the
inverse observed outer information. Python derives predictor derivatives and
implicit-mode derivatives from the model likelihoods and uses dense SciPy
linear algebra. Automatic differentiation from TMB is not part of the Python
runtime.

Brooks et al. (2017), *glmmTMB Balances Speed and Flexibility Among Packages
for Zero-inflated Generalized Linear Mixed Modeling*, The R Journal 9(2),
378-400, *Implementation of glmmTMB* and *The structure of glmmTMB*.
[Publication](https://journal.r-project.org/articles/RJ-2017-066/).
The model separates conditional means, zero-inflation probabilities, and
dispersion predictors. NB1 uses variance proportional to the mean, while NB2
uses a quadratic mean term. The normalized mixture density for a zero is the
sum of structural-zero mass and conditional zero mass. Positive outcomes
combine conditional density with the non-structural mixture probability.
The [public family manual](https://glmmTMB.github.io/glmmTMB/reference/nbinom2.html)
specifies current family parameterizations. Gaussian dispersion coordinates
are checked against public fit outputs from the version recorded in
`oracle/versions.json`.

The public
[unstructured-covariance mapping](https://glmmtmb.github.io/glmmTMB/articles/covstruct.html#unstructured-1)
specifies log standard deviations and scaled Cholesky correlation parameters.
For a unit-diagonal lower factor `L`, the correlation is
`D**(-1/2) @ L @ L.T @ D**(-1/2)`, where `D = diag(L @ L.T)`.
Python converts its physical covariance factors into these coordinates to
calculate observed outer information. Symmetric central differences of the
score use an unscaled increment of `1e-3`. This increment agrees with the
documented `ndeps` default in
[stats optim and optimHess](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/optim.html);
its agreement with glmmTMB's reported covariance was established by public
numerical observations. The calculation is independently implemented and
does not import an R optimizer or target implementation.

The development oracle may refine stationary scores using an independently
written Newton callback through the documented
[glmmTMBControl optimizer interface](https://glmmtmb.github.io/glmmTMB/reference/glmmTMBControl.html).
It receives the public objective and gradient as callable inputs and accepts
small steps only when the objective is preserved and the score norm improves.
This reference-precision step is separate from the Python fitting algorithm.

Ferrari and Cribari-Neto (2004), *Beta Regression for Modelling Rates and
Proportions*, Journal of Applied Statistics 31(7), 799-815.
[Publication](https://doi.org/10.1080/0266476042000214501).
The beta likelihood is parameterized by mean and precision, with shape
parameters `mu * phi` and `(1 - mu) * phi`. Log-gamma normalization and
polygamma derivatives are independently evaluated using SciPy. Binomial,
Poisson, and Gaussian components use their normalized standard densities.
Joint fixed-component covariance feeds the marginal-means and Wald
hypotheses documented above.

## Identified boundary references and profiled information

The public
[reduced-rank covariance specification](https://glmmtmb.github.io/glmmTMB/articles/covstruct.html#reduced-rank)
defines covariance through finite loadings. A rank-one two-dimensional random
covariance is the outer product of its loading vector. Independent public R
fits in this chart verify identified limiting inference without inverting the
divergent unstructured correlation coordinate. Nominal parameter-count
quantities still refer to the original fitted model.

For an information matrix partitioned into component and nuisance blocks,
the profiled component information is
`Hcc - Hcn @ inv(Hnn) @ Hnc`. Its weak eigenspace detects joint component
contrasts instead of relying on separate diagonal entries. This is the
block-inverse identity applied to the likelihood information specified by
Kristensen et al. (2016), Section 2. Orthogonal-coordinate tests independently
verify the calculation.

For a positive penalized block `K`, the norm of a profiled convex score at zero
divided by the smallest eigenvalue of `K` bounds its fitted coefficient norm.
The Schur complement bounds its contribution to Bayesian coefficient
covariance. The executable GAM guards recompute both quantities from each
actual fit. They compare retained null-space coefficients and covariance at
their original tolerances and separately retain the coefficient integration
measure in Wood's REML criterion.
