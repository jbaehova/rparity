# glmmTMB migration

Import `glmmTMB`, `emmeans`, and `Anova` from `rparity`. The examples assume an
existing data frame `d`, a fitted model `m`, and prediction data `nd`. Formula
arguments are strings. Factors can be pandas categorical columns; their
training levels and contrast coding are retained for prediction. Polars
frames are converted to pandas internally.

The runtime requires Python 3.11 or newer. Its Python dependencies are the
five packages declared in `pyproject.toml`; the OpenBLAS backend used by GAM
shrinkage penalties is bundled at build time.

Conditional random effects are integrated with a normalized Laplace
approximation. Fitting uses maximum likelihood, including Gaussian models.
There is no runtime R or automatic-differentiation dependency. See the
[validation coverage](../coverage.md) for numerical evidence and the
[public model specification](https://glmmTMB.github.io/glmmTMB/reference/glmmTMB.html)
for the R interface.
Stage 2 is complete in v0.2.0. The fixed extended-model corpus passes
1,172/1,200 cases; its 28 remaining differences count as failures in the
combined Stage 2 result of 2,356/2,400 (98.17%). See the
[Stage 2 report](https://github.com/jbaehova/rparity/blob/main/reports/STAGE_2_REPORT.md)
and [case-level evidence](https://github.com/jbaehova/rparity/blob/main/reports/STAGE_2_FAILURES.json).

## Model specification

| Use | R | Python |
| --- | --- | --- |
| Gaussian random intercept | `glmmTMB(y ~ x+(1\|g),data=d,family=gaussian())` | `glmmTMB("y ~ x+(1\|g)",d,family="gaussian")` |
| Poisson | `glmmTMB(y ~ x+(1\|g),data=d,family=poisson())` | `glmmTMB("y ~ x+(1\|g)",d,family="poisson")` |
| Negative binomial 1 | `glmmTMB(y ~ x+(1\|g),data=d,family=nbinom1())` | `glmmTMB("y ~ x+(1\|g)",d,family="nbinom1")` |
| Negative binomial 2 | `glmmTMB(y ~ x+(1\|g),data=d,family=nbinom2())` | `glmmTMB("y ~ x+(1\|g)",d,family="nbinom2")` |
| Binomial logit | `glmmTMB(y ~ x+(1\|g),data=d,family=binomial())` | `glmmTMB("y ~ x+(1\|g)",d,family="binomial")` |
| Binomial counts | `glmmTMB(cbind(success,failure) ~ x+(1\|g),data=d,family=binomial())` | `glmmTMB("cbind(success,failure) ~ x+(1\|g)",d,family="binomial")` |
| Binomial proportions | `glmmTMB(y ~ x+(1\|g),data=d,family=binomial(),weights=trials)` | `glmmTMB("y ~ x+(1\|g)",d,family="binomial",weights="trials")` |
| Beta logit | `glmmTMB(y ~ x+(1\|g),data=d,family=beta_family())` | `glmmTMB("y ~ x+(1\|g)",d,family="beta")` |
| Correlated intercept and slope | `glmmTMB(y ~ x+(x\|g),data=d,family=poisson())` | `glmmTMB("y ~ x+(x\|g)",d,family="poisson")` |
| Fixed effects only | `glmmTMB(y ~ x+a,data=d,family=nbinom2())` | `glmmTMB("y ~ x+a",d,family="nbinom2")` |
| No zero inflation | `glmmTMB(f,data=d,family=poisson(),ziformula=~0)` | `glmmTMB(f,d,family="poisson",ziformula="~0")` |
| Constant zero inflation | `glmmTMB(f,data=d,family=poisson(),ziformula=~1)` | `glmmTMB(f,d,family="poisson",ziformula="~1")` |
| Zero-inflation covariates | `glmmTMB(f,data=d,family=nbinom2(),ziformula=~x+a)` | `glmmTMB(f,d,family="nbinom2",ziformula="~x+a")` |
| Constant dispersion | `glmmTMB(f,data=d,family=nbinom2(),dispformula=~1)` | `glmmTMB(f,d,family="nbinom2",dispformula="~1")` |
| Dispersion covariates | `glmmTMB(f,data=d,family=nbinom2(),dispformula=~x+a)` | `glmmTMB(f,d,family="nbinom2",dispformula="~x+a")` |
| Heterogeneous Gaussian SD | `glmmTMB(f,data=d,family=gaussian(),dispformula=~a)` | `glmmTMB(f,d,family="gaussian",dispformula="~a")` |
| Observation weights | `glmmTMB(f,data=d,family=poisson(),weights=w)` | `glmmTMB(f,d,family="poisson",weights="w")` |
| Formula offset | `glmmTMB(y ~ x+offset(o)+(1\|g),data=d,family=poisson())` | `glmmTMB("y ~ x+offset(o)+(1\|g)",d,family="poisson")` |
| Offset argument | `glmmTMB(f,data=d,family=poisson(),offset=o)` | `glmmTMB(f,d,family="poisson",offset="o")` |
| Sum coding | `glmmTMB(f,data=d,contrasts=list(a="contr.sum"))` | `glmmTMB(f,d,contrasts="sum")` |

Here `f` is an R formula in the R column and its formula string in the Python
column. Zero-inflation and dispersion formulas use fixed predictors. Random
effects belong to the conditional formula.

The six supported families use their default links. Poisson and both negative
binomial families use log links. Binomial and beta use logit links. Gaussian
uses the identity link. Other links are unavailable in this module.

The negative binomial parameterizations differ: `nbinom1` has conditional
variance `mu * (1 + phi)`, whereas `nbinom2` has conditional variance
`mu + mu**2 / phi`. Increasing `phi` therefore increases NB1 overdispersion and
reduces NB2 overdispersion. Beta regression uses precision `phi`, with variance
`mu * (1 - mu) / (1 + phi)`. These definitions follow the
[public family specifications](https://glmmTMB.github.io/glmmTMB/reference/nbinom2.html).

Dispersion coefficients are on a log scale. For the recorded glmmTMB
1.1.15.2 oracle, Gaussian dispersion models log standard deviation, so its
variance is `exp(2 * eta_disp)`. `predict(type="disp")` returns standard
deviation for Gaussian models and the family dispersion or precision for
the other families that estimate it. Poisson and binomial have no estimated
dispersion component.

## Extract results and predict

| Use | R | Python |
| --- | --- | --- |
| Summary | `summary(m)` | `print(m.summary())` |
| All fixed components | `fixef(m)` | `m.fixef()` |
| Conditional coefficients | `fixef(m)$cond` | `m.fixef()["cond"]` |
| Zero-inflation coefficients | `fixef(m)$zi` | `m.fixef()["zi"]` |
| Dispersion coefficients | `fixef(m)$disp` | `m.fixef()["disp"]` |
| Conditional covariance | `vcov(m)$cond` | `m.vcov("cond")` |
| Zero-inflation covariance | `vcov(m)$zi` | `m.vcov("zi")` |
| Dispersion covariance | `vcov(m)$disp` | `m.vcov("disp")` |
| Joint covariance | `vcov(m,full=TRUE)` | `m.vcov(full=True)` |
| Random modes | `ranef(m)$cond` | `m.ranef()["cond"]` |
| Conditional random covariance | `ranef(m,condVar=TRUE)` | `m.ranef(cond_var=True)` |
| Random variance components | `VarCorr(m)$cond` | `m.VarCorr()["cond"]` |
| Log likelihood | `logLik(m)` | `m.logLik()` |
| AIC | `AIC(m)` | `m.AIC()` |
| BIC | `BIC(m)` | `m.BIC()` |
| Intercept-only dispersion | `sigma(m)` | `m.sigma()` |
| Fitted response mean | `fitted(m)` | `m.fitted()` |
| Response residuals | `residuals(m,type="response")` | `m.residuals(type="response")` |
| Conditional mean | `predict(m,newdata=nd,type="conditional")` | `m.predict(nd,type="cond")` |
| Zero-inflation probability | `predict(m,newdata=nd,type="zprob")` | `m.predict(nd,type="zprob")` |
| Overall response mean | `predict(m,newdata=nd,type="response")` | `m.predict(nd,type="response")` |
| Conditional link predictor | `predict(m,newdata=nd,type="link")` | `m.predict(nd,type="link")` |
| Zero-inflation logit | `predict(m,newdata=nd,type="zlink")` | `m.predict(nd,type="zlink")` |
| Dispersion prediction | `predict(m,newdata=nd,type="disp")` | `m.predict(nd,type="disp")` |
| Population response | `predict(m,newdata=nd,type="response",re.form=NA)` | `m.predict(nd,type="response",re_form="NA")` |
| New grouping levels | `predict(m,newdata=nd,allow.new.levels=TRUE)` | `m.predict(nd,allow_new_levels=True)` |

`fixef()` returns a dictionary of named coefficient series. Empty `zi` and
`disp` entries indicate components without estimated coefficients. `ranef()`
and `VarCorr()` return component dictionaries keyed by grouping factor. The
conditional covariance of random effects is attached to each random-effect
data frame as `attrs["condVar"]`.
`vcov(full=True)` also includes the fitted random-covariance parameters.
Their `theta_*` coordinates are internal factor coordinates, so direct
comparison with R's covariance-parameter block requires a coordinate map.

For a zero-inflated model, let `mu` be the conditional mean and `p` the
structural-zero probability. A response prediction is `mu * (1 - p)`.
`type="cond"` is a Python alias of `type="conditional"`. Without zero
inflation, `zprob` is zero and response and conditional means agree. These
prediction definitions follow the
[public prediction specification](https://glmmTMB.github.io/glmmTMB/reference/predict.glmmTMB.html).

Population prediction sets random modes to zero. It does not integrate a
nonlinear inverse link over the random-effect distribution. The default
`re_form=None` includes the fitted random modes. With `allow_new_levels=True`,
new group levels contribute zero random effects. Formula offsets are
reevaluated in new data. Numeric offsets supplied during fitting require
explicit prediction offsets for new data.

## Marginal means and Wald tests

The Stage 1 inference functions accept conditional, zero-inflation, and
dispersion components. Pass `component` explicitly for clarity. Conditional
means on the response scale refer to the conditional mean, while
`component="zi", type="response"` reports structural-zero probabilities.

| Use | R | Python |
| --- | --- | --- |
| Conditional marginal means | `emmeans(m,"a",component="cond")` | `emmeans(m,"a",component="cond")` |
| Conditional response scale | `emmeans(m,"a",component="cond",type="response")` | `emmeans(m,"a",component="cond",type="response")` |
| Means at a covariate value | `emmeans(m,"a",component="cond",at=list(x=0))` | `emmeans(m,"a",component="cond",at={"x":[0]})` |
| Zero-inflation probabilities | `emmeans(m,"a",component="zi",type="response")` | `emmeans(m,"a",component="zi",type="response")` |
| Dispersion means | `emmeans(m,"a",component="disp",type="response")` | `emmeans(m,"a",component="disp",type="response")` |
| Pairwise conditional contrasts | `emmeans(m,pairwise ~ a,component="cond")` | `emmeans(m,"a",component="cond",pairwise=True)` |
| Conditional Type II | `car::Anova(m,type=2,component="cond")` | `Anova(m,type=2,component="cond")` |
| Conditional Type III | `car::Anova(m,type=3,component="cond")` | `Anova(m,type=3,component="cond")` |
| Zero-inflation Wald table | `car::Anova(m,type=2,component="zi")` | `Anova(m,type=2,component="zi")` |
| Dispersion Wald table | `car::Anova(m,type=2,component="disp")` | `Anova(m,type=2,component="disp")` |

```python
from rparity import Anova, emmeans, glmmTMB

m = glmmTMB(
    "y ~ x+a+(1|g)", d, family="nbinom2",
    ziformula="~a", dispformula="~x",
)
print(m.summary())
print(Anova(m, type=2, component="cond"))
means = emmeans(m, "a", component="cond", type="response")
print(means.summary())
overall = m.predict(nd, type="response", re_form="NA")
zero_probability = m.predict(nd, type="zprob")
```

`Anova()` uses component-specific Wald chi-squared tests. Type III hypotheses
depend on the fitted contrasts; use sum coding when an overall main effect
in an interaction model is intended. Marginal means generally use asymptotic
degrees of freedom. A Gaussian fixed-effects model with constant dispersion
and no zero inflation uses residual degrees of freedom. See the
[emmeans guide](emmeans.md) for weighting and contrast adjustments.

## Scope

This implementation uses dense matrices and is suited to small and medium
grouped models. It includes conditional random intercepts and correlated
slopes, fixed zero-inflation predictors, and fixed dispersion predictors.
Tweedie, truncated distributions, and hurdle models are deferred. REML,
random effects in zero-inflation or dispersion formulas, prediction standard
errors, and selecting a subset of random effects through `re_form` are
unavailable. A zero-inflated beta model admits structural zeros; its beta
component requires responses strictly between zero and one.

Coefficient covariance uses observed outer information in the documented
log-SD and scaled Cholesky coordinates. Regular fits use centered score
differences of step `1e-3`. At an exactly singular correlated random-effect
fit, a Richardson refinement resolves the remaining identifiable directions.
The coordinates follow the
[public unstructured-covariance mapping](https://glmmtmb.github.io/glmmTMB/articles/covstruct.html#unstructured-1).
Very small random variances, correlations at their limit, and an effectively
absent zero-inflation component can make individual covariance coordinates or
Wald tests unstable. Weak-information and singularity checks use the fitted
model's score, information, and physical random-effect covariance.
Convergence warnings and case-level discrepancies remain
visible in the validation evidence. Neither an optimizer success code nor a
finite likelihood alone establishes reliable inference.
