---
title: "Type II and III ANOVA in Python: car and lmerTest migration"
description: Translate car Anova and lmerTest ANOVA to Python for linear, generalized and mixed models. Learn contrast coding, Wald tests and degrees-of-freedom options.
---

# car migration

These examples assume an existing data frame `d` and fitted models. Import the
named functions from `rparity`. For statsmodels examples, import
`statsmodels.api as sm` and `statsmodels.formula.api as smf`.

See [numerical accuracy and limitations](../validation.md) and the
[option-level coverage](../coverage.md) when reproducing an R analysis.

Type III tests depend on contrast coding. **Use sum contrasts** when testing
overall main effects in interaction models. This also applies to statsmodels:
use `C(a, Sum)` in the fitted formula.

| Use | R | Python |
| --- | --- | --- |
| OLS Type II | `Anova(m,type=2)` | `Anova(m,type=2)` |
| OLS Type III | `Anova(m,type=3)` | `Anova(m,type=3)` |
| Type labels | `Anova(m,type="II")` | `Anova(m,type="II")` |
| OLS F | `Anova(m,type=2,test.statistic="F")` | `Anova(m,type=2,test_statistic="F")` |
| GLM LR II | `Anova(m,type=2,test.statistic="LR")` | `Anova(m,type=2,test_statistic="LR")` |
| GLM LR III | `Anova(m,type=3,test.statistic="LR")` | `Anova(m,type=3,test_statistic="LR")` |
| GLM Wald II | `Anova(m,type=2,test.statistic="Wald")` | `Anova(m,type=2,test_statistic="Wald")` |
| GLM Wald III | `Anova(m,type=3,test.statistic="Wald")` | `Anova(m,type=3,test_statistic="Wald")` |
| GLM F II | `Anova(m,type=2,test.statistic="F")` | `Anova(m,type=2,test_statistic="F")` |
| GLM F III | `Anova(m,type=3,test.statistic="F")` | `Anova(m,type=3,test_statistic="F")` |
| Mixed Wald II | `Anova(m,type=2,test.statistic="Chisq")` | `Anova(m,type=2,test_statistic="Chisq")` |
| Mixed Wald III | `Anova(m,type=3,test.statistic="Chisq")` | `Anova(m,type=3,test_statistic="Chisq")` |
| Mixed KR II | `Anova(m,type=2,test.statistic="F")` | `Anova(m,type=2,test_statistic="F")` |
| Mixed KR III | `Anova(m,type=3,test.statistic="F")` | `Anova(m,type=3,test_statistic="F")` |
| GLMM Wald II | `Anova(gm,type=2)` | `Anova(gm,type=2)` |
| GLMM Wald III | `Anova(gm,type=3)` | `Anova(gm,type=3)` |
| Satterthwaite I | `anova(m,type=1,ddf="Satterthwaite")` | `anova(m,type=1,ddf="satterthwaite")` |
| Satterthwaite II | `anova(m,type=2,ddf="Satterthwaite")` | `anova(m,type=2,ddf="satterthwaite")` |
| Satterthwaite III | `anova(m,type=3,ddf="Satterthwaite")` | `anova(m,type=3,ddf="satterthwaite")` |
| KR I | `anova(m,type=1,ddf="Kenward-Roger")` | `anova(m,type=1,ddf="kenward-roger")` |
| KR II | `anova(m,type=2,ddf="Kenward-Roger")` | `anova(m,type=2,ddf="kenward-roger")` |
| KR III | `anova(m,type=3,ddf="Kenward-Roger")` | `anova(m,type=3,ddf="kenward-roger")` |
| Default lmerTest III | `anova(m)` | `anova(m)` |
| ML comparison | `anova(m0,m1)` | `anova(m0,m1)` |
| Disable refitting | `anova(m0,m1,refit=FALSE)` | `anova(m0,m1,refit=False)` |
| Sum-coded mixed model | `lmer(y~a*b+(1\|g),d,contrasts=list(a=contr.sum,b=contr.sum))` | `lmer("y~a*b+(1\|g)",d,contrasts="sum")` |
| Treatment-coded model | `lmer(y~a*b+(1\|g),d)` | `lmer("y~a*b+(1\|g)",d,contrasts="treatment")` |
| Statsmodels formula OLS | `lm(y~a*b,data=d)` | `smf.ols("y ~ C(a, Sum)*C(b, Sum)",d).fit()` |
| Statsmodels Poisson | `glm(y~a,data=d,family=poisson)` | `smf.glm("y~a",d,family=sm.families.Poisson()).fit()` |
| Statsmodels binomial | `glm(y~a,data=d,family=binomial)` | `smf.glm("y~a",d,family=sm.families.Binomial()).fit()` |
| GLM Pearson dispersion | `Anova(m,test.statistic="F",error.estimate="pearson")` | `Anova(m,test_statistic="F",error_estimate="pearson")` |
