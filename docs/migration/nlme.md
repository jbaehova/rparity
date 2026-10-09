# nlme migration

These examples assume an existing data frame `d` and fitted models. Import the
named functions from `rparity`. For statsmodels examples, import
`statsmodels.api as sm` and `statsmodels.formula.api as smf`.

Consult the validation coverage for numerical limitations in v0.1.0.

| Use | R | Python |
| --- | --- | --- |
| Independent GLS | `gls(y~x,data=d)` | `gls("y~x",d)` |
| REML | `gls(y~x,data=d,method="REML")` | `gls("y~x",d,method="REML")` |
| ML | `gls(y~x,data=d,method="ML")` | `gls("y~x",d,method="ML")` |
| AR1 | `gls(f,d,correlation=corAR1(form=~t\|g))` | `gls(f,d,correlation=corAR1(form="~t\|g"))` |
| AR1 start | `corAR1(value=.3,form=~t\|g)` | `corAR1(value=.3,form="~t\|g")` |
| Fixed AR1 | `corAR1(value=.3,form=~t\|g,fixed=TRUE)` | `corAR1(value=.3,form="~t\|g",fixed=True)` |
| Compound symmetry | `gls(f,d,correlation=corCompSymm(form=~1\|g))` | `gls(f,d,correlation=corCompSymm(form="~1\|g"))` |
| Compound symmetry start | `corCompSymm(value=.2,form=~1\|g)` | `corCompSymm(value=.2,form="~1\|g")` |
| Unstructured | `gls(f,d,correlation=corSymm(form=~t\|g))` | `gls(f,d,correlation=corSymm(form="~t\|g"))` |
| ARMA11 | `corARMA(p=1,q=1,form=~t\|g)` | `corARMA(p=1,q=1,form="~t\|g")` |
| AR2 | `corARMA(p=2,q=0,form=~t\|g)` | `corARMA(p=2,q=0,form="~t\|g")` |
| MA1 | `corARMA(p=0,q=1,form=~t\|g)` | `corARMA(p=0,q=1,form="~t\|g")` |
| Group variance | `gls(f,d,weights=varIdent(form=~1\|a))` | `gls(f,d,weights=varIdent(form="~1\|a"))` |
| Variance ratios | `varIdent(value=c(b=2),form=~1\|a)` | `varIdent(value={"b":2},form="~1\|a")` |
| Power variance | `gls(f,d,weights=varPower(form=~v))` | `gls(f,d,weights=varPower(form="~v"))` |
| Fitted power variance | `varPower(form=~fitted(.))` | `varPower(form="~fitted(.)")` |
| Exponential variance | `gls(f,d,weights=varExp(form=~v))` | `gls(f,d,weights=varExp(form="~v"))` |
| Fitted exponential variance | `varExp(form=~fitted(.))` | `varExp(form="~fitted(.)")` |
| Combined covariance | `gls(f,d,correlation=corAR1(form=~t\|g),weights=varIdent(form=~1\|a))` | `gls(f,d,correlation=corAR1(form="~t\|g"),weights=varIdent(form="~1\|a"))` |
| Coefficients | `coef(m)` | `m.coef()` |
| Covariance | `vcov(m)` | `m.vcov()` |
| Likelihood | `logLik(m)` | `m.logLik()` |
| AIC | `AIC(m)` | `m.AIC()` |
| BIC | `BIC(m)` | `m.BIC()` |
| Summary | `summary(m)` | `m.summary()` |
| Sequential ANOVA | `anova(m,type="sequential")` | `m.anova(type="sequential")` |
| Marginal ANOVA | `anova(m,type="marginal")` | `m.anova(type="marginal")` |
| Intervals | `intervals(m)` | `m.intervals()` |
| Coefficients intervals | `intervals(m,which="coef")` | `m.intervals(which="coef")` |
| Predict | `predict(m,newdata=nd)` | `m.predict(newdata=nd)` |
| Residuals | `resid(m,type="normalized")` | `m.residuals(type="normalized")` |
| Fitted | `fitted(m)` | `m.fitted()` |
