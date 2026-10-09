# lme4 migration

These examples assume an existing data frame `d` and fitted models. Import the
named functions from `rparity`. For statsmodels examples, import
`statsmodels.api as sm` and `statsmodels.formula.api as smf`.

Consult the validation coverage for numerical limitations in v0.1.0.

Population prediction uses `re_form="NA"`; `re_form=None` includes conditional
random effects. `deviance()` for Gaussian models returns the fitted ML/REML
criterion, consistent with the selected estimation method.

| Use | R | Python |
| --- | --- | --- |
| Random intercept | `lmer(y ~ x + (1\|g), data=d)` | `lmer("y ~ x + (1\|g)", d)` |
| Random slope | `lmer(y ~ x + (x\|g), data=d)` | `lmer("y ~ x + (x\|g)", d)` |
| Independent slopes | `lmer(y ~ x + (x\|\|g), data=d)` | `lmer("y ~ x + (x\|\|g)", d)` |
| Crossed intercepts | `lmer(y ~ x + (1\|g)+(1\|h), data=d)` | `lmer("y ~ x + (1\|g)+(1\|h)", d)` |
| Nested intercepts | `lmer(y ~ x + (1\|g/h), data=d)` | `lmer("y ~ x + (1\|g/h)", d)` |
| Slope without intercept | `lmer(y ~ x + (0+x\|g), data=d)` | `lmer("y ~ x + (0+x\|g)", d)` |
| REML | `lmer(f, data=d, REML=TRUE)` | `lmer(f, d, reml=True)` |
| ML | `lmer(f, data=d, REML=FALSE)` | `lmer(f, d, reml=False)` |
| Precision weights | `lmer(f, data=d, weights=w)` | `lmer(f, d, weights="w")` |
| Offset argument | `lmer(f, data=d, offset=o)` | `lmer(f, d, offset="o")` |
| Formula offset | `lmer(y ~ x+offset(o)+(1\|g), data=d)` | `lmer("y ~ x+offset(o)+(1\|g)", d)` |
| Fixed effects | `fixef(m)` | `m.fixef()` |
| Conditional modes | `ranef(m)` | `m.ranef()` |
| Conditional variance | `ranef(m, condVar=TRUE)` | `m.ranef(cond_var=True)` |
| Variance components | `VarCorr(m)` | `m.VarCorr()` |
| Fixed covariance | `vcov(m)` | `m.vcov()` |
| Log likelihood | `logLik(m)` | `m.logLik()` |
| AIC | `AIC(m)` | `m.AIC()` |
| BIC | `BIC(m)` | `m.BIC()` |
| Fit criterion | `deviance(m)` | `m.deviance()` |
| Fitted values | `fitted(m)` | `m.fitted()` |
| Residuals | `residuals(m)` | `m.residuals()` |
| Prediction | `predict(m, newdata=nd)` | `m.predict(newdata=nd)` |
| Population prediction | `predict(m, newdata=nd, re.form=NA)` | `m.predict(newdata=nd, re_form="NA")` |
| New groups | `predict(m,newdata=nd,allow.new.levels=TRUE)` | `m.predict(newdata=nd,allow_new_levels=True)` |
| Singularity | `isSingular(m)` | `m.is_singular()` |
| Satterthwaite summary | `summary(m)` | `m.summary()` |
| Likelihood comparison | `anova(m0,m1)` | `anova(m0,m1)` |
| Binomial logit | `glmer(f,data=d,family=binomial("logit"),nAGQ=1)` | `glmer(f,d,family="binomial",link="logit")` |
| Binomial probit | `glmer(f,data=d,family=binomial("probit"),nAGQ=1)` | `glmer(f,d,link="probit")` |
| Binomial cloglog | `glmer(f,data=d,family=binomial("cloglog"),nAGQ=1)` | `glmer(f,d,link="cloglog")` |
| Binomial counts | `glmer(cbind(s,f)~x+(1\|g),data=d,family=binomial)` | `glmer("cbind(s,f)~x+(1\|g)",d)` |
| Poisson | `glmer(f,data=d,family=poisson,nAGQ=1)` | `glmer(f,d,family="poisson")` |
