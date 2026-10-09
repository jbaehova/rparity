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
