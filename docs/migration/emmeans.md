# emmeans migration

These examples assume an existing data frame `d` and fitted models. Import the
named functions from `rparity`. For statsmodels examples, import
`statsmodels.api as sm` and `statsmodels.formula.api as smf`.

Consult the validation coverage for numerical limitations in v0.1.0.

Control contrasts require an explicit supported adjustment. The R `dunnettx`
default is unavailable; Python defaults to `none`.

| Use | R | Python |
| --- | --- | --- |
| Marginal means | `emmeans(m,"a")` | `emmeans(m,"a")` |
| Two factors | `emmeans(m,c("a","b"))` | `emmeans(m,["a","b"])` |
| Conditioning | `emmeans(m,"a",by="b")` | `emmeans(m,"a",by="b")` |
| Formula conditioning | `emmeans(m,~a\|b)` | `emmeans(m,"~a\|b")` |
| At a numeric value | `emmeans(m,"a",at=list(x=2))` | `emmeans(m,"a",at={"x":[2]})` |
| At several values | `emmeans(m,"x",at=list(x=c(0,1,2)))` | `emmeans(m,"x",at={"x":[0,1,2]})` |
| Equal weights | `emmeans(m,"a",weights="equal")` | `emmeans(m,"a",weights="equal")` |
| Proportional weights | `emmeans(m,"a",weights="proportional")` | `emmeans(m,"a",weights="proportional")` |
| Cell weights | `emmeans(m,"a",weights="cells")` | `emmeans(m,"a",weights="cells")` |
| Outer weights | `emmeans(m,"a",weights="outer")` | `emmeans(m,"a",weights="outer")` |
| Flat weights | `emmeans(m,"a",weights="flat")` | `emmeans(m,"a",weights="flat")` |
| Pairwise | `pairs(e)` | `pairs(e)` |
| Reverse pairs | `pairs(e,reverse=TRUE)` | `pairs(e,reverse=True)` |
| Pairwise contrast | `contrast(e,"pairwise")` | `contrast(e,"pairwise")` |
| Control contrasts | `contrast(e,"trt.vs.ctrl",ref=1,adjust="bonferroni")` | `contrast(e,"trt.vs.ctrl",ref=1,adjust="bonferroni")` |
| Consecutive contrasts | `contrast(e,"consec")` | `contrast(e,"consec")` |
| Polynomial contrasts | `contrast(e,"poly")` | `contrast(e,"poly")` |
| Custom contrasts | `contrast(e,list(diff=c(1,-1,0)))` | `contrast(e,{"diff":[1,-1,0]})` |
| Tukey adjustment | `pairs(e,adjust="tukey")` | `pairs(e,adjust="tukey")` |
| Bonferroni | `pairs(e,adjust="bonferroni")` | `pairs(e,adjust="bonferroni")` |
| Holm | `pairs(e,adjust="holm")` | `pairs(e,adjust="holm")` |
| Sidak | `pairs(e,adjust="sidak")` | `pairs(e,adjust="sidak")` |
| No adjustment | `pairs(e,adjust="none")` | `pairs(e,adjust="none")` |
| FDR | `pairs(e,adjust="fdr")` | `pairs(e,adjust="fdr")` |
| Response scale | `summary(e,type="response")` | `e.summary(type="response")` |
| Regrid | `regrid(e,transform="response")` | `regrid(e,transform="response")` |
| KR degrees of freedom | `emmeans(m,"a",lmer.df="kenward-roger")` | `emmeans(m,"a",lmer_df="kenward-roger")` |
| Satterthwaite | `emmeans(m,"a",lmer.df="satterthwaite")` | `emmeans(m,"a",lmer_df="satterthwaite")` |
| Asymptotic | `emmeans(m,"a",lmer.df="asymptotic")` | `emmeans(m,"a",lmer_df="asymptotic")` |
| Trends | `emtrends(m,"a",var="x")` | `emtrends(m,"a",var="x")` |
| Joint tests | `joint_tests(m)` | `joint_tests(m)` |
| Confidence limits | `confint(e,level=.90)` | `e.confint(level=.90)` |
| Tests and intervals | `summary(e,infer=c(TRUE,TRUE))` | `e.summary(infer=(True,True))` |
