# Validation coverage

Stage 1 remains in progress until every required option and acceptance check passes.

Counts below refer to committed synthetic R observations. Passing status comes from
the latest recorded pytest run; unexecuted or skipped cases are not passes.

| Module | R function and options | Implementation | Golden cases | Pass rate | Notes |
| --- | --- | --- | ---: | ---: | --- |
| anova | anova / lmer / ML model comparison | implemented, verification ongoing | 50 | unverified | 0 failures; 50 unverified |
| anova | car::Anova / glm / Type 2 / F / sum | implemented, verification ongoing | 16 | unverified | 0 failures; 16 unverified |
| anova | car::Anova / glm / Type 2 / F / treatment | implemented, verification ongoing | 17 | unverified | 0 failures; 17 unverified |
| anova | car::Anova / glm / Type 2 / LR / sum | implemented, verification ongoing | 17 | unverified | 0 failures; 17 unverified |
| anova | car::Anova / glm / Type 2 / LR / treatment | implemented, verification ongoing | 17 | unverified | 0 failures; 17 unverified |
| anova | car::Anova / glm / Type 2 / Wald / sum | implemented, verification ongoing | 17 | unverified | 0 failures; 17 unverified |
| anova | car::Anova / glm / Type 2 / Wald / treatment | implemented, verification ongoing | 16 | unverified | 0 failures; 16 unverified |
| anova | car::Anova / glm / Type 3 / F / sum | implemented, verification ongoing | 17 | unverified | 0 failures; 17 unverified |
| anova | car::Anova / glm / Type 3 / F / treatment | implemented, verification ongoing | 17 | unverified | 0 failures; 17 unverified |
| anova | car::Anova / glm / Type 3 / LR / sum | implemented, verification ongoing | 17 | unverified | 0 failures; 17 unverified |
| anova | car::Anova / glm / Type 3 / LR / treatment | implemented, verification ongoing | 16 | unverified | 0 failures; 16 unverified |
| anova | car::Anova / glm / Type 3 / Wald / sum | implemented, verification ongoing | 16 | unverified | 0 failures; 16 unverified |
| anova | car::Anova / glm / Type 3 / Wald / treatment | implemented, verification ongoing | 17 | unverified | 0 failures; 17 unverified |
| anova | car::Anova / glmer / Type 2 / Chisq / sum | implemented, verification ongoing | 25 | unverified | 0 failures; 25 unverified |
| anova | car::Anova / glmer / Type 2 / Chisq / treatment | implemented, verification ongoing | 25 | unverified | 0 failures; 25 unverified |
| anova | car::Anova / glmer / Type 3 / Chisq / sum | implemented, verification ongoing | 25 | unverified | 0 failures; 25 unverified |
| anova | car::Anova / glmer / Type 3 / Chisq / treatment | implemented, verification ongoing | 25 | unverified | 0 failures; 25 unverified |
| anova | car::Anova / lm / Type 2 / F / sum | implemented, verification ongoing | 50 | unverified | 0 failures; 50 unverified |
| anova | car::Anova / lm / Type 2 / F / treatment | implemented, verification ongoing | 50 | unverified | 0 failures; 50 unverified |
| anova | car::Anova / lm / Type 3 / F / sum | implemented, verification ongoing | 50 | unverified | 0 failures; 50 unverified |
| anova | car::Anova / lm / Type 3 / F / treatment | implemented, verification ongoing | 50 | unverified | 0 failures; 50 unverified |
| anova | car::Anova / lmer / Type 2 / Chisq / sum | implemented, verification ongoing | 25 | unverified | 0 failures; 25 unverified |
| anova | car::Anova / lmer / Type 2 / Chisq / treatment | implemented, verification ongoing | 25 | unverified | 0 failures; 25 unverified |
| anova | car::Anova / lmer / Type 2 / F / sum | implemented, verification ongoing | 25 | unverified | 0 failures; 25 unverified |
| anova | car::Anova / lmer / Type 3 / Chisq / treatment | implemented, verification ongoing | 25 | unverified | 0 failures; 25 unverified |
| anova | car::Anova / lmer / Type 3 / F / sum | implemented, verification ongoing | 25 | unverified | 0 failures; 25 unverified |
| anova | car::Anova / lmer / Type 3 / F / treatment | implemented, verification ongoing | 25 | unverified | 0 failures; 25 unverified |
| emm | contrast / consec / fdr | implemented, verification ongoing | 27 | unverified | 0 failures; 27 unverified |
| emm | contrast / consec / tukey | implemented, verification ongoing | 9 | unverified | 0 failures; 9 unverified |
| emm | contrast / custom / fdr | implemented, verification ongoing | 18 | unverified | 0 failures; 18 unverified |
| emm | contrast / custom / tukey | implemented, verification ongoing | 6 | unverified | 0 failures; 6 unverified |
| emm | contrast / pairwise / bonferroni | implemented, verification ongoing | 27 | unverified | 0 failures; 27 unverified |
| emm | contrast / pairwise / holm | implemented, verification ongoing | 9 | unverified | 0 failures; 9 unverified |
| emm | contrast / poly / bonferroni | implemented, verification ongoing | 27 | unverified | 0 failures; 27 unverified |
| emm | contrast / poly / holm | implemented, verification ongoing | 6 | unverified | 0 failures; 6 unverified |
| emm | contrast / revpairwise / fdr | implemented, verification ongoing | 27 | unverified | 0 failures; 27 unverified |
| emm | contrast / revpairwise / tukey | implemented, verification ongoing | 9 | unverified | 0 failures; 9 unverified |
| emm | contrast / trt.vs.ctrl / bonferroni | implemented, verification ongoing | 27 | unverified | 0 failures; 27 unverified |
| emm | contrast / trt.vs.ctrl / holm | implemented, verification ongoing | 9 | unverified | 0 failures; 9 unverified |
| emm | emmeans / glmer / asymptotic | implemented, verification ongoing | 3 | unverified | 0 failures; 3 unverified |
| emm | emmeans / glmer / kenward-roger | implemented, verification ongoing | 4 | unverified | 0 failures; 4 unverified |
| emm | emmeans / glmer / satterthwaite | implemented, verification ongoing | 3 | unverified | 0 failures; 3 unverified |
| emm | emmeans / gls / asymptotic | implemented, verification ongoing | 3 | unverified | 0 failures; 3 unverified |
| emm | emmeans / gls / kenward-roger | implemented, verification ongoing | 4 | unverified | 0 failures; 4 unverified |
| emm | emmeans / gls / satterthwaite | implemented, verification ongoing | 3 | unverified | 0 failures; 3 unverified |
| emm | emmeans / lmer / asymptotic | implemented, verification ongoing | 3 | unverified | 0 failures; 3 unverified |
| emm | emmeans / lmer / kenward-roger | implemented, verification ongoing | 4 | unverified | 0 failures; 4 unverified |
| emm | emmeans / lmer / satterthwaite | implemented, verification ongoing | 3 | unverified | 0 failures; 3 unverified |
| emm | emmeans / response / at | implemented, verification ongoing | 48 | unverified | 0 failures; 48 unverified |
| emm | emmeans / weights cells | implemented, verification ongoing | 51 | unverified | 0 failures; 51 unverified |
| emm | emmeans / weights equal | implemented, verification ongoing | 51 | unverified | 0 failures; 51 unverified |
| emm | emmeans / weights flat | implemented, verification ongoing | 51 | unverified | 0 failures; 51 unverified |
| emm | emmeans / weights outer | implemented, verification ongoing | 51 | unverified | 0 failures; 51 unverified |
| emm | emmeans / weights proportional | implemented, verification ongoing | 51 | unverified | 0 failures; 51 unverified |
| emm | emtrends / glmer / asymptotic | implemented, verification ongoing | 4 | unverified | 0 failures; 4 unverified |
| emm | emtrends / glmer / kenward-roger | implemented, verification ongoing | 3 | unverified | 0 failures; 3 unverified |
| emm | emtrends / glmer / satterthwaite | implemented, verification ongoing | 3 | unverified | 0 failures; 3 unverified |
| emm | emtrends / gls / asymptotic | implemented, verification ongoing | 4 | unverified | 0 failures; 4 unverified |
| emm | emtrends / gls / kenward-roger | implemented, verification ongoing | 3 | unverified | 0 failures; 3 unverified |
| emm | emtrends / gls / satterthwaite | implemented, verification ongoing | 3 | unverified | 0 failures; 3 unverified |
| emm | emtrends / lmer / asymptotic | implemented, verification ongoing | 4 | unverified | 0 failures; 4 unverified |
| emm | emtrends / lmer / kenward-roger | implemented, verification ongoing | 3 | unverified | 0 failures; 3 unverified |
| emm | emtrends / lmer / satterthwaite | implemented, verification ongoing | 3 | unverified | 0 failures; 3 unverified |
| emm | emtrends / numeric slope | implemented, verification ongoing | 48 | unverified | 0 failures; 48 unverified |
| emm | joint_tests / factorial | implemented, verification ongoing | 48 | unverified | 0 failures; 48 unverified |
| emm | joint_tests / glmer / asymptotic | implemented, verification ongoing | 1 | unverified | 0 failures; 1 unverified |
| emm | joint_tests / glmer / kenward-roger | implemented, verification ongoing | 1 | unverified | 0 failures; 1 unverified |
| emm | joint_tests / glmer / satterthwaite | implemented, verification ongoing | 1 | unverified | 0 failures; 1 unverified |
| emm | joint_tests / gls / asymptotic | implemented, verification ongoing | 1 | unverified | 0 failures; 1 unverified |
| emm | joint_tests / gls / kenward-roger | implemented, verification ongoing | 1 | unverified | 0 failures; 1 unverified |
| emm | joint_tests / gls / satterthwaite | implemented, verification ongoing | 1 | unverified | 0 failures; 1 unverified |
| emm | joint_tests / lmer / asymptotic | implemented, verification ongoing | 1 | unverified | 0 failures; 1 unverified |
| emm | joint_tests / lmer / kenward-roger | implemented, verification ongoing | 1 | unverified | 0 failures; 1 unverified |
| emm | joint_tests / lmer / satterthwaite | implemented, verification ongoing | 1 | unverified | 0 failures; 1 unverified |
| emm | pairs / glmer / asymptotic | implemented, verification ongoing | 3 | unverified | 0 failures; 3 unverified |
| emm | pairs / glmer / kenward-roger | implemented, verification ongoing | 3 | unverified | 0 failures; 3 unverified |
| emm | pairs / glmer / satterthwaite | implemented, verification ongoing | 4 | unverified | 0 failures; 4 unverified |
| emm | pairs / gls / asymptotic | implemented, verification ongoing | 3 | unverified | 0 failures; 3 unverified |
| emm | pairs / gls / kenward-roger | implemented, verification ongoing | 3 | unverified | 0 failures; 3 unverified |
| emm | pairs / gls / satterthwaite | implemented, verification ongoing | 4 | unverified | 0 failures; 4 unverified |
| emm | pairs / lmer / asymptotic | implemented, verification ongoing | 3 | unverified | 0 failures; 3 unverified |
| emm | pairs / lmer / kenward-roger | implemented, verification ongoing | 3 | unverified | 0 failures; 3 unverified |
| emm | pairs / lmer / satterthwaite | implemented, verification ongoing | 4 | unverified | 0 failures; 4 unverified |
| emm | regrid / response / glm | implemented, verification ongoing | 16 | unverified | 0 failures; 16 unverified |
| emm | regrid / response / lm | implemented, verification ongoing | 8 | unverified | 0 failures; 8 unverified |
| emm | response / glmer / asymptotic | implemented, verification ongoing | 3 | unverified | 0 failures; 3 unverified |
| emm | response / glmer / kenward-roger | implemented, verification ongoing | 4 | unverified | 0 failures; 4 unverified |
| emm | response / glmer / satterthwaite | implemented, verification ongoing | 3 | unverified | 0 failures; 3 unverified |
| emm | response / gls / asymptotic | implemented, verification ongoing | 3 | unverified | 0 failures; 3 unverified |
| emm | response / gls / kenward-roger | implemented, verification ongoing | 4 | unverified | 0 failures; 4 unverified |
| emm | response / gls / satterthwaite | implemented, verification ongoing | 3 | unverified | 0 failures; 3 unverified |
| emm | response / lmer / asymptotic | implemented, verification ongoing | 3 | unverified | 0 failures; 3 unverified |
| emm | response / lmer / kenward-roger | implemented, verification ongoing | 4 | unverified | 0 failures; 4 unverified |
| emm | response / lmer / satterthwaite | implemented, verification ongoing | 3 | unverified | 0 failures; 3 unverified |
| glmer | glmer / binomial-cloglog | implemented, verification ongoing | 150 | unverified | 0 failures; 150 unverified |
| glmer | glmer / binomial-logit | implemented, verification ongoing | 150 | unverified | 0 failures; 150 unverified |
| glmer | glmer / binomial-probit | implemented, verification ongoing | 150 | unverified | 0 failures; 150 unverified |
| glmer | glmer / poisson-log | implemented, verification ongoing | 150 | unverified | 0 failures; 150 unverified |
| gls | gls / corAR1 + varIdent / ML | implemented, verification ongoing | 30 | unverified | 0 failures; 30 unverified |
| gls | gls / corAR1 + varIdent / REML | implemented, verification ongoing | 30 | unverified | 0 failures; 30 unverified |
| gls | gls / corAR1 / ML | implemented, verification ongoing | 30 | unverified | 0 failures; 30 unverified |
| gls | gls / corAR1 / REML | implemented, verification ongoing | 30 | unverified | 0 failures; 30 unverified |
| gls | gls / corARMA / ML | implemented, verification ongoing | 30 | unverified | 0 failures; 30 unverified |
| gls | gls / corARMA / REML | implemented, verification ongoing | 30 | unverified | 0 failures; 30 unverified |
| gls | gls / corCompSymm + varExp / ML | implemented, verification ongoing | 30 | unverified | 0 failures; 30 unverified |
| gls | gls / corCompSymm + varExp / REML | implemented, verification ongoing | 30 | unverified | 0 failures; 30 unverified |
| gls | gls / corCompSymm / ML | implemented, verification ongoing | 30 | unverified | 0 failures; 30 unverified |
| gls | gls / corCompSymm / REML | implemented, verification ongoing | 30 | unverified | 0 failures; 30 unverified |
| gls | gls / corSymm / ML | implemented, verification ongoing | 30 | unverified | 0 failures; 30 unverified |
| gls | gls / corSymm / REML | implemented, verification ongoing | 30 | unverified | 0 failures; 30 unverified |
| gls | gls / independent / ML | implemented, verification ongoing | 30 | unverified | 0 failures; 30 unverified |
| gls | gls / independent / REML | implemented, verification ongoing | 30 | unverified | 0 failures; 30 unverified |
| gls | gls / independentvarExp / ML | implemented, verification ongoing | 30 | unverified | 0 failures; 30 unverified |
| gls | gls / independentvarExp / REML | implemented, verification ongoing | 30 | unverified | 0 failures; 30 unverified |
| gls | gls / independentvarIdent / ML | implemented, verification ongoing | 30 | unverified | 0 failures; 30 unverified |
| gls | gls / independentvarIdent / REML | implemented, verification ongoing | 30 | unverified | 0 failures; 30 unverified |
| gls | gls / varPower / ML | implemented, verification ongoing | 30 | unverified | 0 failures; 30 unverified |
| gls | gls / varPower / REML | implemented, verification ongoing | 30 | unverified | 0 failures; 30 unverified |
| inference | lmerTest / Kenward-Roger / anova / Type 1 | implemented, verification ongoing | 68 | unverified | 0 failures; 68 unverified |
| inference | lmerTest / Kenward-Roger / anova / Type 2 | implemented, verification ongoing | 66 | unverified | 0 failures; 66 unverified |
| inference | lmerTest / Kenward-Roger / anova / Type 3 | implemented, verification ongoing | 66 | unverified | 0 failures; 66 unverified |
| inference | lmerTest / Satterthwaite / anova / Type 1 | implemented, verification ongoing | 68 | unverified | 0 failures; 68 unverified |
| inference | lmerTest / Satterthwaite / anova / Type 2 | implemented, verification ongoing | 66 | unverified | 0 failures; 66 unverified |
| inference | lmerTest / Satterthwaite / anova / Type 3 | implemented, verification ongoing | 66 | unverified | 0 failures; 66 unverified |
| inference | lmerTest / Satterthwaite / fit / Type 1 | implemented, verification ongoing | 68 | unverified | 0 failures; 68 unverified |
| inference | lmerTest / Satterthwaite / fit / Type 2 | implemented, verification ongoing | 66 | unverified | 0 failures; 66 unverified |
| inference | lmerTest / Satterthwaite / fit / Type 3 | implemented, verification ongoing | 66 | unverified | 0 failures; 66 unverified |
| lmer | lmer / ML / (0+x\|g) | implemented, verification ongoing | 65 | unverified | 0 failures; 65 unverified |
| lmer | lmer / ML / (0+x\|g) / missingness | implemented, verification ongoing | 4 | unverified | 0 failures; 4 unverified |
| lmer | lmer / ML / (0+x\|g) / missingness / weights | implemented, verification ongoing | 1 | unverified | 0 failures; 1 unverified |
| lmer | lmer / ML / (0+x\|g) / treatment | implemented, verification ongoing | 16 | unverified | 0 failures; 16 unverified |
| lmer | lmer / ML / (0+x\|g) / treatment / missingness | implemented, verification ongoing | 1 | unverified | 0 failures; 1 unverified |
| lmer | lmer / ML / (0+x\|g) / treatment / weights | implemented, verification ongoing | 3 | unverified | 0 failures; 3 unverified |
| lmer | lmer / ML / (0+x\|g) / weights | implemented, verification ongoing | 10 | unverified | 0 failures; 10 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) | implemented, verification ongoing | 45 | unverified | 0 failures; 45 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / missingness | implemented, verification ongoing | 1 | unverified | 0 failures; 1 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / missingness / offset | implemented, verification ongoing | 2 | unverified | 0 failures; 2 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / missingness / weights | implemented, verification ongoing | 1 | unverified | 0 failures; 1 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / offset | implemented, verification ongoing | 21 | unverified | 0 failures; 21 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / treatment | implemented, verification ongoing | 10 | unverified | 0 failures; 10 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / treatment / missingness | implemented, verification ongoing | 2 | unverified | 0 failures; 2 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / treatment / offset | implemented, verification ongoing | 5 | unverified | 0 failures; 5 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / treatment / weights | implemented, verification ongoing | 2 | unverified | 0 failures; 2 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / treatment / weights / offset | implemented, verification ongoing | 1 | unverified | 0 failures; 1 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / weights | implemented, verification ongoing | 6 | unverified | 0 failures; 6 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / weights / offset | implemented, verification ongoing | 4 | unverified | 0 failures; 4 unverified |
| lmer | lmer / ML / (x\|g) | implemented, verification ongoing | 64 | unverified | 0 failures; 64 unverified |
| lmer | lmer / ML / (x\|g) / missingness | implemented, verification ongoing | 4 | unverified | 0 failures; 4 unverified |
| lmer | lmer / ML / (x\|g) / missingness / weights | implemented, verification ongoing | 1 | unverified | 0 failures; 1 unverified |
| lmer | lmer / ML / (x\|g) / treatment | implemented, verification ongoing | 16 | unverified | 0 failures; 16 unverified |
| lmer | lmer / ML / (x\|g) / treatment / missingness | implemented, verification ongoing | 1 | unverified | 0 failures; 1 unverified |
| lmer | lmer / ML / (x\|g) / treatment / weights | implemented, verification ongoing | 3 | unverified | 0 failures; 3 unverified |
| lmer | lmer / ML / (x\|g) / weights | implemented, verification ongoing | 11 | unverified | 0 failures; 11 unverified |
| lmer | lmer / REML / (1\|g) | implemented, verification ongoing | 38 | unverified | 0 failures; 38 unverified |
| lmer | lmer / REML / (1\|g) / boundary | implemented, verification ongoing | 15 | unverified | 0 failures; 15 unverified |
| lmer | lmer / REML / (1\|g) / missingness | implemented, verification ongoing | 4 | unverified | 0 failures; 4 unverified |
| lmer | lmer / REML / (1\|g) / offset | implemented, verification ongoing | 22 | unverified | 0 failures; 22 unverified |
| lmer | lmer / REML / (1\|g) / sum | implemented, verification ongoing | 11 | unverified | 0 failures; 11 unverified |
| lmer | lmer / REML / (1\|g) / sum / missingness / offset | implemented, verification ongoing | 1 | unverified | 0 failures; 1 unverified |
| lmer | lmer / REML / (1\|g) / sum / offset | implemented, verification ongoing | 5 | unverified | 0 failures; 5 unverified |
| lmer | lmer / REML / (1\|g) / sum / weights | implemented, verification ongoing | 2 | unverified | 0 failures; 2 unverified |
| lmer | lmer / REML / (1\|g) / sum / weights / offset | implemented, verification ongoing | 1 | unverified | 0 failures; 1 unverified |
| lmer | lmer / REML / (1\|g) / weights | implemented, verification ongoing | 8 | unverified | 0 failures; 8 unverified |
| lmer | lmer / REML / (1\|g) / weights / offset | implemented, verification ongoing | 3 | unverified | 0 failures; 3 unverified |
| lmer | lmer / REML / (1\|g/h) | implemented, verification ongoing | 60 | unverified | 0 failures; 60 unverified |
| lmer | lmer / REML / (1\|g/h) / missingness | implemented, verification ongoing | 4 | unverified | 0 failures; 4 unverified |
| lmer | lmer / REML / (1\|g/h) / missingness / weights | implemented, verification ongoing | 1 | unverified | 0 failures; 1 unverified |
| lmer | lmer / REML / (1\|g/h) / sum | implemented, verification ongoing | 17 | unverified | 0 failures; 17 unverified |
| lmer | lmer / REML / (1\|g/h) / sum / missingness | implemented, verification ongoing | 1 | unverified | 0 failures; 1 unverified |
| lmer | lmer / REML / (1\|g/h) / sum / weights | implemented, verification ongoing | 2 | unverified | 0 failures; 2 unverified |
| lmer | lmer / REML / (1\|g/h) / weights | implemented, verification ongoing | 10 | unverified | 0 failures; 10 unverified |
| lmer | lmer / REML / (x\|\|g) | implemented, verification ongoing | 60 | unverified | 0 failures; 60 unverified |
| lmer | lmer / REML / (x\|\|g) / missingness | implemented, verification ongoing | 5 | unverified | 0 failures; 5 unverified |
| lmer | lmer / REML / (x\|\|g) / sum | implemented, verification ongoing | 17 | unverified | 0 failures; 17 unverified |
| lmer | lmer / REML / (x\|\|g) / sum / missingness / weights | implemented, verification ongoing | 1 | unverified | 0 failures; 1 unverified |
| lmer | lmer / REML / (x\|\|g) / sum / weights | implemented, verification ongoing | 2 | unverified | 0 failures; 2 unverified |
| lmer | lmer / REML / (x\|\|g) / weights | implemented, verification ongoing | 10 | unverified | 0 failures; 10 unverified |

## Module totals

| Module | Golden cases | Passed | Failed |
| --- | ---: | ---: | ---: |
| anova | 700 | 0 | 0 |
| emm | 753 | 0 | 0 |
| glmer | 600 | 0 | 0 |
| gls | 600 | 0 | 0 |
| inference | 600 | 0 | 0 |
| lmer | 600 | 0 | 0 |

Total: 3853 synthetic cases, 0 recorded passes.

Each module requires at least 300 cases; Stage 1 requires at least 3,000 and 98% passing.

## R package examples

Built-in R example data is not committed. The following `needs_r` cases
are included in coverage and require the development oracle.

| Representative example | Result |
| --- | --- |
| Representative examples | unverified |
