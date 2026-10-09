---
title: R numerical compatibility and supported options in Python
description: Recorded R comparisons for supported rparity model and inference options in Python.
---

# Validation coverage

These results document numerical compatibility for the supported Python APIs.

Counts below refer to committed synthetic R observations. Passing status comes from
the latest recorded pytest run; unexecuted or skipped cases are not passes.

| Module | R function and options | Implementation | Golden cases | Pass rate | Notes |
| --- | --- | --- | ---: | ---: | --- |
| anova | anova / lmer / ML model comparison | implemented; validation recorded | 50 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / glm / Type 2 / F / sum | implemented; validation recorded | 16 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / glm / Type 2 / F / treatment | implemented; validation recorded | 17 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / glm / Type 2 / LR / sum | implemented; validation recorded | 17 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / glm / Type 2 / LR / treatment | implemented; validation recorded | 17 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / glm / Type 2 / Wald / sum | implemented; validation recorded | 17 | 94.12% | 1 failures; 0 unverified |
| anova | car::Anova / glm / Type 2 / Wald / treatment | implemented; validation recorded | 16 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / glm / Type 3 / F / sum | implemented; validation recorded | 17 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / glm / Type 3 / F / treatment | implemented; validation recorded | 17 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / glm / Type 3 / LR / sum | implemented; validation recorded | 17 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / glm / Type 3 / LR / treatment | implemented; validation recorded | 16 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / glm / Type 3 / Wald / sum | implemented; validation recorded | 16 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / glm / Type 3 / Wald / treatment | implemented; validation recorded | 17 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / glmer / Type 2 / Chisq / sum | implemented; validation recorded | 25 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / glmer / Type 2 / Chisq / treatment | implemented; validation recorded | 25 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / glmer / Type 3 / Chisq / sum | implemented; validation recorded | 25 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / glmer / Type 3 / Chisq / treatment | implemented; validation recorded | 25 | 96.00% | 1 failures; 0 unverified |
| anova | car::Anova / lm / Type 2 / F / sum | implemented; validation recorded | 50 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / lm / Type 2 / F / treatment | implemented; validation recorded | 50 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / lm / Type 3 / F / sum | implemented; validation recorded | 50 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / lm / Type 3 / F / treatment | implemented; validation recorded | 50 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / lmer / Type 2 / Chisq / sum | implemented; validation recorded | 25 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / lmer / Type 2 / Chisq / treatment | implemented; validation recorded | 25 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / lmer / Type 2 / F / sum | implemented; validation recorded | 25 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / lmer / Type 3 / Chisq / treatment | implemented; validation recorded | 25 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / lmer / Type 3 / F / sum | implemented; validation recorded | 25 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / lmer / Type 3 / F / treatment | implemented; validation recorded | 25 | 96.00% | 1 failures; 0 unverified |
| emm | contrast / consec / fdr | implemented; validation recorded | 27 | 100.00% | 0 failures; 0 unverified |
| emm | contrast / consec / tukey | implemented; validation recorded | 9 | 100.00% | 0 failures; 0 unverified |
| emm | contrast / custom / fdr | implemented; validation recorded | 18 | 100.00% | 0 failures; 0 unverified |
| emm | contrast / custom / tukey | implemented; validation recorded | 6 | 100.00% | 0 failures; 0 unverified |
| emm | contrast / pairwise / bonferroni | implemented; validation recorded | 27 | 100.00% | 0 failures; 0 unverified |
| emm | contrast / pairwise / holm | implemented; validation recorded | 9 | 100.00% | 0 failures; 0 unverified |
| emm | contrast / poly / bonferroni | implemented; validation recorded | 27 | 100.00% | 0 failures; 0 unverified |
| emm | contrast / poly / holm | implemented; validation recorded | 6 | 100.00% | 0 failures; 0 unverified |
| emm | contrast / revpairwise / fdr | implemented; validation recorded | 27 | 100.00% | 0 failures; 0 unverified |
| emm | contrast / revpairwise / tukey | implemented; validation recorded | 9 | 100.00% | 0 failures; 0 unverified |
| emm | contrast / trt.vs.ctrl / bonferroni | implemented; validation recorded | 27 | 100.00% | 0 failures; 0 unverified |
| emm | contrast / trt.vs.ctrl / holm | implemented; validation recorded | 9 | 100.00% | 0 failures; 0 unverified |
| emm | emmeans / glmer / asymptotic | implemented; validation recorded | 3 | 100.00% | 0 failures; 0 unverified |
| emm | emmeans / glmer / kenward-roger | implemented; validation recorded | 4 | 100.00% | 0 failures; 0 unverified |
| emm | emmeans / glmer / satterthwaite | implemented; validation recorded | 3 | 100.00% | 0 failures; 0 unverified |
| emm | emmeans / gls / asymptotic | implemented; validation recorded | 3 | 100.00% | 0 failures; 0 unverified |
| emm | emmeans / gls / kenward-roger | implemented; validation recorded | 4 | 100.00% | 0 failures; 0 unverified |
| emm | emmeans / gls / satterthwaite | implemented; validation recorded | 3 | 100.00% | 0 failures; 0 unverified |
| emm | emmeans / lmer / asymptotic | implemented; validation recorded | 3 | 100.00% | 0 failures; 0 unverified |
| emm | emmeans / lmer / kenward-roger | implemented; validation recorded | 4 | 100.00% | 0 failures; 0 unverified |
| emm | emmeans / lmer / satterthwaite | implemented; validation recorded | 3 | 100.00% | 0 failures; 0 unverified |
| emm | emmeans / response / at | implemented; validation recorded | 48 | 100.00% | 0 failures; 0 unverified |
| emm | emmeans / weights cells | implemented; validation recorded | 51 | 100.00% | 0 failures; 0 unverified |
| emm | emmeans / weights equal | implemented; validation recorded | 51 | 100.00% | 0 failures; 0 unverified |
| emm | emmeans / weights flat | implemented; validation recorded | 51 | 100.00% | 0 failures; 0 unverified |
| emm | emmeans / weights outer | implemented; validation recorded | 51 | 100.00% | 0 failures; 0 unverified |
| emm | emmeans / weights proportional | implemented; validation recorded | 51 | 100.00% | 0 failures; 0 unverified |
| emm | emtrends / glmer / asymptotic | implemented; validation recorded | 4 | 100.00% | 0 failures; 0 unverified |
| emm | emtrends / glmer / kenward-roger | implemented; validation recorded | 3 | 100.00% | 0 failures; 0 unverified |
| emm | emtrends / glmer / satterthwaite | implemented; validation recorded | 3 | 100.00% | 0 failures; 0 unverified |
| emm | emtrends / gls / asymptotic | implemented; validation recorded | 4 | 100.00% | 0 failures; 0 unverified |
| emm | emtrends / gls / kenward-roger | implemented; validation recorded | 3 | 100.00% | 0 failures; 0 unverified |
| emm | emtrends / gls / satterthwaite | implemented; validation recorded | 3 | 100.00% | 0 failures; 0 unverified |
| emm | emtrends / lmer / asymptotic | implemented; validation recorded | 4 | 100.00% | 0 failures; 0 unverified |
| emm | emtrends / lmer / kenward-roger | implemented; validation recorded | 3 | 100.00% | 0 failures; 0 unverified |
| emm | emtrends / lmer / satterthwaite | implemented; validation recorded | 3 | 100.00% | 0 failures; 0 unverified |
| emm | emtrends / numeric slope | implemented; validation recorded | 48 | 100.00% | 0 failures; 0 unverified |
| emm | joint_tests / factorial | implemented; validation recorded | 48 | 100.00% | 0 failures; 0 unverified |
| emm | joint_tests / glmer / asymptotic | implemented; validation recorded | 1 | 100.00% | 0 failures; 0 unverified |
| emm | joint_tests / glmer / kenward-roger | implemented; validation recorded | 1 | 100.00% | 0 failures; 0 unverified |
| emm | joint_tests / glmer / satterthwaite | implemented; validation recorded | 1 | 100.00% | 0 failures; 0 unverified |
| emm | joint_tests / gls / asymptotic | implemented; validation recorded | 1 | 100.00% | 0 failures; 0 unverified |
| emm | joint_tests / gls / kenward-roger | implemented; validation recorded | 1 | 100.00% | 0 failures; 0 unverified |
| emm | joint_tests / gls / satterthwaite | implemented; validation recorded | 1 | 100.00% | 0 failures; 0 unverified |
| emm | joint_tests / lmer / asymptotic | implemented; validation recorded | 1 | 100.00% | 0 failures; 0 unverified |
| emm | joint_tests / lmer / kenward-roger | implemented; validation recorded | 1 | 100.00% | 0 failures; 0 unverified |
| emm | joint_tests / lmer / satterthwaite | implemented; validation recorded | 1 | 100.00% | 0 failures; 0 unverified |
| emm | pairs / glmer / asymptotic | implemented; validation recorded | 3 | 100.00% | 0 failures; 0 unverified |
| emm | pairs / glmer / kenward-roger | implemented; validation recorded | 3 | 100.00% | 0 failures; 0 unverified |
| emm | pairs / glmer / satterthwaite | implemented; validation recorded | 4 | 100.00% | 0 failures; 0 unverified |
| emm | pairs / gls / asymptotic | implemented; validation recorded | 3 | 66.67% | 1 failures; 0 unverified |
| emm | pairs / gls / kenward-roger | implemented; validation recorded | 3 | 100.00% | 0 failures; 0 unverified |
| emm | pairs / gls / satterthwaite | implemented; validation recorded | 4 | 100.00% | 0 failures; 0 unverified |
| emm | pairs / lmer / asymptotic | implemented; validation recorded | 3 | 100.00% | 0 failures; 0 unverified |
| emm | pairs / lmer / kenward-roger | implemented; validation recorded | 3 | 100.00% | 0 failures; 0 unverified |
| emm | pairs / lmer / satterthwaite | implemented; validation recorded | 4 | 100.00% | 0 failures; 0 unverified |
| emm | regrid / response / glm | implemented; validation recorded | 16 | 100.00% | 0 failures; 0 unverified |
| emm | regrid / response / lm | implemented; validation recorded | 8 | 100.00% | 0 failures; 0 unverified |
| emm | response / glmer / asymptotic | implemented; validation recorded | 3 | 100.00% | 0 failures; 0 unverified |
| emm | response / glmer / kenward-roger | implemented; validation recorded | 4 | 100.00% | 0 failures; 0 unverified |
| emm | response / glmer / satterthwaite | implemented; validation recorded | 3 | 100.00% | 0 failures; 0 unverified |
| emm | response / gls / asymptotic | implemented; validation recorded | 3 | 100.00% | 0 failures; 0 unverified |
| emm | response / gls / kenward-roger | implemented; validation recorded | 4 | 100.00% | 0 failures; 0 unverified |
| emm | response / gls / satterthwaite | implemented; validation recorded | 3 | 100.00% | 0 failures; 0 unverified |
| emm | response / lmer / asymptotic | implemented; validation recorded | 3 | 100.00% | 0 failures; 0 unverified |
| emm | response / lmer / kenward-roger | implemented; validation recorded | 4 | 100.00% | 0 failures; 0 unverified |
| emm | response / lmer / satterthwaite | implemented; validation recorded | 3 | 100.00% | 0 failures; 0 unverified |
| gam | gam / additive / Gamma / GCV.Cp | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / additive / Gamma / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / additive / Gamma / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / additive / binomial / GCV.Cp | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / additive / binomial / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / additive / binomial / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / additive / gaussian / GCV.Cp | implemented; validation recorded | 10 | 90.00% | 1 failures; 0 unverified |
| gam | gam / additive / gaussian / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / additive / gaussian / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / additive / poisson / GCV.Cp | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / additive / poisson / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / additive / poisson / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / continuous-by / Gamma / GCV.Cp | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / continuous-by / Gamma / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / continuous-by / Gamma / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / continuous-by / binomial / GCV.Cp | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / continuous-by / binomial / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / continuous-by / binomial / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / continuous-by / gaussian / GCV.Cp | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / continuous-by / gaussian / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / continuous-by / gaussian / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / continuous-by / poisson / GCV.Cp | implemented; validation recorded | 10 | 90.00% | 1 failures; 0 unverified |
| gam | gam / continuous-by / poisson / ML | implemented; validation recorded | 10 | 90.00% | 1 failures; 0 unverified |
| gam | gam / continuous-by / poisson / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cr / Gamma / GCV.Cp | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cr / Gamma / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cr / Gamma / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cr / binomial / GCV.Cp | implemented; validation recorded | 10 | 90.00% | 1 failures; 0 unverified |
| gam | gam / cr / binomial / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cr / binomial / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cr / gaussian / GCV.Cp | implemented; validation recorded | 10 | 90.00% | 1 failures; 0 unverified |
| gam | gam / cr / gaussian / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cr / gaussian / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cr / poisson / GCV.Cp | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cr / poisson / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cr / poisson / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cs / Gamma / GCV.Cp | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cs / Gamma / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cs / Gamma / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cs / binomial / GCV.Cp | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cs / binomial / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cs / binomial / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cs / gaussian / GCV.Cp | implemented; validation recorded | 10 | 90.00% | 1 failures; 0 unverified |
| gam | gam / cs / gaussian / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cs / gaussian / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cs / poisson / GCV.Cp | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cs / poisson / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cs / poisson / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / factor-by / Gamma / GCV.Cp | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / factor-by / Gamma / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / factor-by / Gamma / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / factor-by / binomial / GCV.Cp | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / factor-by / binomial / ML | implemented; validation recorded | 10 | 90.00% | 1 failures; 0 unverified |
| gam | gam / factor-by / binomial / REML | implemented; validation recorded | 10 | 80.00% | 2 failures; 0 unverified |
| gam | gam / factor-by / gaussian / GCV.Cp | implemented; validation recorded | 10 | 80.00% | 2 failures; 0 unverified |
| gam | gam / factor-by / gaussian / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / factor-by / gaussian / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / factor-by / poisson / GCV.Cp | implemented; validation recorded | 10 | 90.00% | 1 failures; 0 unverified |
| gam | gam / factor-by / poisson / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / factor-by / poisson / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ps / Gamma / GCV.Cp | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ps / Gamma / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ps / Gamma / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ps / binomial / GCV.Cp | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ps / binomial / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ps / binomial / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ps / gaussian / GCV.Cp | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ps / gaussian / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ps / gaussian / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ps / poisson / GCV.Cp | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ps / poisson / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ps / poisson / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / re / Gamma / GCV.Cp | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / re / Gamma / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / re / Gamma / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / re / binomial / GCV.Cp | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / re / binomial / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / re / binomial / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / re / gaussian / GCV.Cp | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / re / gaussian / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / re / gaussian / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / re / poisson / GCV.Cp | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / re / poisson / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / re / poisson / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / te / Gamma / GCV.Cp | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / te / Gamma / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / te / Gamma / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / te / binomial / GCV.Cp | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / te / binomial / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / te / binomial / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / te / gaussian / GCV.Cp | implemented; validation recorded | 10 | 80.00% | 2 failures; 0 unverified |
| gam | gam / te / gaussian / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / te / gaussian / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / te / poisson / GCV.Cp | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / te / poisson / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / te / poisson / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ti / Gamma / GCV.Cp | implemented; validation recorded | 10 | 90.00% | 1 failures; 0 unverified |
| gam | gam / ti / Gamma / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ti / Gamma / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ti / binomial / GCV.Cp | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ti / binomial / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ti / binomial / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ti / gaussian / GCV.Cp | implemented; validation recorded | 10 | 90.00% | 1 failures; 0 unverified |
| gam | gam / ti / gaussian / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ti / gaussian / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ti / poisson / GCV.Cp | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ti / poisson / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ti / poisson / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / tp / Gamma / GCV.Cp | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / tp / Gamma / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / tp / Gamma / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / tp / binomial / GCV.Cp | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / tp / binomial / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / tp / binomial / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / tp / gaussian / GCV.Cp | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / tp / gaussian / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / tp / gaussian / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / tp / poisson / GCV.Cp | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / tp / poisson / ML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / tp / poisson / REML | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| glmer | glmer / binomial-cloglog | implemented; validation recorded | 150 | 95.33% | 7 failures; 0 unverified |
| glmer | glmer / binomial-logit | implemented; validation recorded | 150 | 97.33% | 4 failures; 0 unverified |
| glmer | glmer / binomial-logit fractional binary | implemented; validation recorded | 1 | 100.00% | 0 failures; 0 unverified |
| glmer | glmer / binomial-logit fractional cbind | implemented; validation recorded | 1 | 100.00% | 0 failures; 0 unverified |
| glmer | glmer / binomial-logit fractional proportion | implemented; validation recorded | 1 | 100.00% | 0 failures; 0 unverified |
| glmer | glmer / binomial-probit | implemented; validation recorded | 150 | 96.67% | 5 failures; 0 unverified |
| glmer | glmer / poisson-log | implemented; validation recorded | 150 | 98.67% | 2 failures; 0 unverified |
| gls | gls / corAR1 + varIdent / ML | implemented; validation recorded | 30 | 100.00% | 0 failures; 0 unverified |
| gls | gls / corAR1 + varIdent / REML | implemented; validation recorded | 30 | 100.00% | 0 failures; 0 unverified |
| gls | gls / corAR1 / ML | implemented; validation recorded | 30 | 100.00% | 0 failures; 0 unverified |
| gls | gls / corAR1 / REML | implemented; validation recorded | 30 | 100.00% | 0 failures; 0 unverified |
| gls | gls / corARMA / ML | implemented; validation recorded | 30 | 96.67% | 1 failures; 0 unverified |
| gls | gls / corARMA / REML | implemented; validation recorded | 30 | 100.00% | 0 failures; 0 unverified |
| gls | gls / corCompSymm + varExp / ML | implemented; validation recorded | 30 | 93.33% | 2 failures; 0 unverified |
| gls | gls / corCompSymm + varExp / REML | implemented; validation recorded | 30 | 93.33% | 2 failures; 0 unverified |
| gls | gls / corCompSymm / ML | implemented; validation recorded | 30 | 100.00% | 0 failures; 0 unverified |
| gls | gls / corCompSymm / REML | implemented; validation recorded | 30 | 100.00% | 0 failures; 0 unverified |
| gls | gls / corSymm / ML | implemented; validation recorded | 30 | 86.67% | 4 failures; 0 unverified |
| gls | gls / corSymm / REML | implemented; validation recorded | 30 | 96.67% | 1 failures; 0 unverified |
| gls | gls / independent / ML | implemented; validation recorded | 30 | 100.00% | 0 failures; 0 unverified |
| gls | gls / independent / REML | implemented; validation recorded | 30 | 100.00% | 0 failures; 0 unverified |
| gls | gls / independentvarExp / ML | implemented; validation recorded | 30 | 90.00% | 3 failures; 0 unverified |
| gls | gls / independentvarExp / REML | implemented; validation recorded | 30 | 100.00% | 0 failures; 0 unverified |
| gls | gls / independentvarIdent / ML | implemented; validation recorded | 30 | 100.00% | 0 failures; 0 unverified |
| gls | gls / independentvarIdent / REML | implemented; validation recorded | 30 | 100.00% | 0 failures; 0 unverified |
| gls | gls / varPower / ML | implemented; validation recorded | 30 | 100.00% | 0 failures; 0 unverified |
| gls | gls / varPower / REML | implemented; validation recorded | 30 | 100.00% | 0 failures; 0 unverified |
| inference | lmerTest / Kenward-Roger / anova / Type 1 | implemented; validation recorded | 68 | 100.00% | 0 failures; 0 unverified |
| inference | lmerTest / Kenward-Roger / anova / Type 2 | implemented; validation recorded | 66 | 96.97% | 2 failures; 0 unverified |
| inference | lmerTest / Kenward-Roger / anova / Type 3 | implemented; validation recorded | 66 | 100.00% | 0 failures; 0 unverified |
| inference | lmerTest / Satterthwaite / anova / Type 1 | implemented; validation recorded | 68 | 100.00% | 0 failures; 0 unverified |
| inference | lmerTest / Satterthwaite / anova / Type 2 | implemented; validation recorded | 66 | 96.97% | 2 failures; 0 unverified |
| inference | lmerTest / Satterthwaite / anova / Type 3 | implemented; validation recorded | 66 | 100.00% | 0 failures; 0 unverified |
| inference | lmerTest / Satterthwaite / fit / Type 1 | implemented; validation recorded | 68 | 100.00% | 0 failures; 0 unverified |
| inference | lmerTest / Satterthwaite / fit / Type 2 | implemented; validation recorded | 66 | 96.97% | 2 failures; 0 unverified |
| inference | lmerTest / Satterthwaite / fit / Type 3 | implemented; validation recorded | 66 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (0+x\|g) | implemented; validation recorded | 65 | 95.38% | 3 failures; 0 unverified |
| lmer | lmer / ML / (0+x\|g) / missingness | implemented; validation recorded | 4 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (0+x\|g) / missingness / weights | implemented; validation recorded | 1 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (0+x\|g) / treatment | implemented; validation recorded | 16 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (0+x\|g) / treatment / missingness | implemented; validation recorded | 1 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (0+x\|g) / treatment / weights | implemented; validation recorded | 3 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (0+x\|g) / weights | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) | implemented; validation recorded | 45 | 97.78% | 1 failures; 0 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / missingness | implemented; validation recorded | 1 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / missingness / offset | implemented; validation recorded | 2 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / missingness / weights | implemented; validation recorded | 1 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / offset | implemented; validation recorded | 21 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / treatment | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / treatment / missingness | implemented; validation recorded | 2 | 50.00% | 1 failures; 0 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / treatment / offset | implemented; validation recorded | 5 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / treatment / weights | implemented; validation recorded | 2 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / treatment / weights / offset | implemented; validation recorded | 1 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / weights | implemented; validation recorded | 6 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / weights / offset | implemented; validation recorded | 4 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (x\|g) | implemented; validation recorded | 64 | 96.88% | 2 failures; 0 unverified |
| lmer | lmer / ML / (x\|g) / missingness | implemented; validation recorded | 4 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (x\|g) / missingness / weights | implemented; validation recorded | 1 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (x\|g) / treatment | implemented; validation recorded | 16 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (x\|g) / treatment / missingness | implemented; validation recorded | 1 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (x\|g) / treatment / weights | implemented; validation recorded | 3 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (x\|g) / weights | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (1\|g) | implemented; validation recorded | 38 | 97.37% | 1 failures; 0 unverified |
| lmer | lmer / REML / (1\|g) / boundary | implemented; validation recorded | 15 | 93.33% | 1 failures; 0 unverified |
| lmer | lmer / REML / (1\|g) / missingness | implemented; validation recorded | 4 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (1\|g) / offset | implemented; validation recorded | 22 | 90.91% | 2 failures; 0 unverified |
| lmer | lmer / REML / (1\|g) / sum | implemented; validation recorded | 11 | 90.91% | 1 failures; 0 unverified |
| lmer | lmer / REML / (1\|g) / sum / missingness / offset | implemented; validation recorded | 1 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (1\|g) / sum / offset | implemented; validation recorded | 5 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (1\|g) / sum / weights | implemented; validation recorded | 2 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (1\|g) / sum / weights / offset | implemented; validation recorded | 1 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (1\|g) / weights | implemented; validation recorded | 8 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (1\|g) / weights / offset | implemented; validation recorded | 3 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (1\|g/h) | implemented; validation recorded | 60 | 96.67% | 2 failures; 0 unverified |
| lmer | lmer / REML / (1\|g/h) / missingness | implemented; validation recorded | 4 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (1\|g/h) / missingness / weights | implemented; validation recorded | 1 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (1\|g/h) / sum | implemented; validation recorded | 17 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (1\|g/h) / sum / missingness | implemented; validation recorded | 1 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (1\|g/h) / sum / weights | implemented; validation recorded | 2 | 50.00% | 1 failures; 0 unverified |
| lmer | lmer / REML / (1\|g/h) / weights | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (x\|\|g) | implemented; validation recorded | 60 | 98.33% | 1 failures; 0 unverified |
| lmer | lmer / REML / (x\|\|g) / missingness | implemented; validation recorded | 5 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (x\|\|g) / sum | implemented; validation recorded | 17 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (x\|\|g) / sum / missingness / weights | implemented; validation recorded | 1 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (x\|\|g) / sum / weights | implemented; validation recorded | 2 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (x\|\|g) / weights | implemented; validation recorded | 10 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~0 / disp ~1 / (1\|g) | implemented; validation recorded | 12 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~0 / disp ~1 / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~0 / disp ~x + f / (1\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~0 / disp ~x + f / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~0 / disp ~x / (1\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~0 / disp ~x / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~1 / disp ~1 / (1\|g) | implemented; validation recorded | 12 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~1 / disp ~1 / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~1 / disp ~x + f / (1\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~1 / disp ~x + f / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~1 / disp ~x / (1\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~1 / disp ~x / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~z + x / disp ~1 / (1\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~z + x / disp ~1 / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~z + x / disp ~x + f / (1\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~z + x / disp ~x + f / (x\|g) | implemented; validation recorded | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~z + x / disp ~x / (1\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~z + x / disp ~x / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~0 / disp ~1 / (1\|g) | implemented; validation recorded | 12 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~0 / disp ~1 / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~0 / disp ~x + f / (1\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~0 / disp ~x + f / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~0 / disp ~x / (1\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~0 / disp ~x / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~1 / disp ~1 / (1\|g) | implemented; validation recorded | 12 | 91.67% | 1 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~1 / disp ~1 / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~1 / disp ~x + f / (1\|g) | implemented; validation recorded | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~1 / disp ~x + f / (x\|g) | implemented; validation recorded | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~1 / disp ~x / (1\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~1 / disp ~x / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~z + x / disp ~1 / (1\|g) | implemented; validation recorded | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~z + x / disp ~1 / (x\|g) | implemented; validation recorded | 11 | 72.73% | 3 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~z + x / disp ~x + f / (1\|g) | implemented; validation recorded | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~z + x / disp ~x + f / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~z + x / disp ~x / (1\|g) | implemented; validation recorded | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~z + x / disp ~x / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~0 / disp ~1 / (1\|g) | implemented; validation recorded | 12 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~0 / disp ~1 / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~0 / disp ~x + f / (1\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~0 / disp ~x + f / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~0 / disp ~x / (1\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~0 / disp ~x / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~1 / disp ~1 / (1\|g) | implemented; validation recorded | 12 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~1 / disp ~1 / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~1 / disp ~x + f / (1\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~1 / disp ~x + f / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~1 / disp ~x / (1\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~1 / disp ~x / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~z + x / disp ~1 / (1\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~z + x / disp ~1 / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~z + x / disp ~x + f / (1\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~z + x / disp ~x + f / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~z + x / disp ~x / (1\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~z + x / disp ~x / (x\|g) | implemented; validation recorded | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~0 / disp ~1 / (1\|g) | implemented; validation recorded | 12 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~0 / disp ~1 / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~0 / disp ~x + f / (1\|g) | implemented; validation recorded | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~0 / disp ~x + f / (x\|g) | implemented; validation recorded | 11 | 81.82% | 2 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~0 / disp ~x / (1\|g) | implemented; validation recorded | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~0 / disp ~x / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~1 / disp ~1 / (1\|g) | implemented; validation recorded | 12 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~1 / disp ~1 / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~1 / disp ~x + f / (1\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~1 / disp ~x + f / (x\|g) | implemented; validation recorded | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~1 / disp ~x / (1\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~1 / disp ~x / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~z + x / disp ~1 / (1\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~z + x / disp ~1 / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~z + x / disp ~x + f / (1\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~z + x / disp ~x + f / (x\|g) | implemented; validation recorded | 11 | 81.82% | 2 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~z + x / disp ~x / (1\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~z + x / disp ~x / (x\|g) | implemented; validation recorded | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~0 / disp ~1 / (1\|g) | implemented; validation recorded | 12 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~0 / disp ~1 / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~0 / disp ~x + f / (1\|g) | implemented; validation recorded | 11 | 81.82% | 2 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~0 / disp ~x + f / (x\|g) | implemented; validation recorded | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~0 / disp ~x / (1\|g) | implemented; validation recorded | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~0 / disp ~x / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~1 / disp ~1 / (1\|g) | implemented; validation recorded | 12 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~1 / disp ~1 / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~1 / disp ~x + f / (1\|g) | implemented; validation recorded | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~1 / disp ~x + f / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~1 / disp ~x / (1\|g) | implemented; validation recorded | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~1 / disp ~x / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~z + x / disp ~1 / (1\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~z + x / disp ~1 / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~z + x / disp ~x + f / (1\|g) | implemented; validation recorded | 11 | 81.82% | 2 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~z + x / disp ~x + f / (x\|g) | implemented; validation recorded | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~z + x / disp ~x / (1\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~z + x / disp ~x / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~0 / disp ~1 / (1\|g) | implemented; validation recorded | 12 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~0 / disp ~1 / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~0 / disp ~x + f / (1\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~0 / disp ~x + f / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~0 / disp ~x / (1\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~0 / disp ~x / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~1 / disp ~1 / (1\|g) | implemented; validation recorded | 12 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~1 / disp ~1 / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~1 / disp ~x + f / (1\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~1 / disp ~x + f / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~1 / disp ~x / (1\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~1 / disp ~x / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~z + x / disp ~1 / (1\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~z + x / disp ~1 / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~z + x / disp ~x + f / (1\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~z + x / disp ~x + f / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~z + x / disp ~x / (1\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~z + x / disp ~x / (x\|g) | implemented; validation recorded | 11 | 100.00% | 0 failures; 0 unverified |

## Module totals

| Module | Golden cases | Passed | Failed |
| --- | ---: | ---: | ---: |
| anova | 700 | 697 | 3 |
| emm | 753 | 752 | 1 |
| gam | 1200 | 1184 | 16 |
| glmer | 603 | 585 | 18 |
| gls | 600 | 587 | 13 |
| inference | 600 | 594 | 6 |
| lmer | 600 | 584 | 16 |
| tmb | 1200 | 1172 | 28 |

Total: 6256 synthetic cases, 6155 recorded passes and 101 failures.

## Validation groups

| Model group | Golden cases | Passed | Failed | Pass rate |
| --- | ---: | ---: | ---: | ---: |
| Mixed models and inference | 3856 | 3799 | 57 | 98.52% |
| GAMs and distributional models | 2400 | 2356 | 44 | 98.17% |

Known differences remain counted as failures. The option-level results above
describe the tested scope; they do not establish parity for unsupported options.

## R package examples

Built-in R example data is not committed. The following `needs_r` cases
are included in coverage and require the development oracle.

| Representative example | Result |
| --- | --- |
| test_builtin_example[Orthodont-nlme-gls-distance ~ age] | passed |
| test_builtin_example[cbpp-lme4-glmer-cbind(incidence, size-incidence) ~ period + (1 \| herd)] | passed |
| test_builtin_example[sleepstudy-lme4-lmer-Reaction ~ Days + (Days \| Subject)] | passed |
| test_car_duncan_anova[2] | passed |
| test_car_duncan_anova[3] | passed |
| test_emmeans_sleepstudy_trends[asymptotic-5] | passed |
| test_emmeans_sleepstudy_trends[kenward-roger-4] | passed |
| test_emmeans_sleepstudy_trends[satterthwaite-3] | passed |
| test_emmeans_warpbreaks_tukey[emmeans] | passed |
| test_emmeans_warpbreaks_tukey[pairs] | passed |
| test_lmertest_sleepstudy_anova[kenward-roger-2] | passed |
| test_lmertest_sleepstudy_anova[satterthwaite-1] | passed |
| test_lmertest_sleepstudy_coefficients | passed |
| test_nlme_orthodont_inference | passed |
| test_public_mcycle_gaussian_gam | passed |
| test_public_salamanders_zero_inflated_poisson | passed |
