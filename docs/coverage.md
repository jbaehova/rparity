# Validation coverage

Stage 1 acceptance checks are complete for v0.1.0.

Stage 2 acceptance checks are complete for v0.2.0.

Counts below refer to committed synthetic R observations. Passing status comes from
the latest recorded pytest run; unexecuted or skipped cases are not passes.

| Module | R function and options | Implementation | Golden cases | Pass rate | Notes |
| --- | --- | --- | ---: | ---: | --- |
| anova | anova / lmer / ML model comparison | implemented and validated | 50 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / glm / Type 2 / F / sum | implemented and validated | 16 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / glm / Type 2 / F / treatment | implemented and validated | 17 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / glm / Type 2 / LR / sum | implemented and validated | 17 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / glm / Type 2 / LR / treatment | implemented and validated | 17 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / glm / Type 2 / Wald / sum | implemented and validated | 17 | 94.12% | 1 failures; 0 unverified |
| anova | car::Anova / glm / Type 2 / Wald / treatment | implemented and validated | 16 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / glm / Type 3 / F / sum | implemented and validated | 17 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / glm / Type 3 / F / treatment | implemented and validated | 17 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / glm / Type 3 / LR / sum | implemented and validated | 17 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / glm / Type 3 / LR / treatment | implemented and validated | 16 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / glm / Type 3 / Wald / sum | implemented and validated | 16 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / glm / Type 3 / Wald / treatment | implemented and validated | 17 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / glmer / Type 2 / Chisq / sum | implemented and validated | 25 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / glmer / Type 2 / Chisq / treatment | implemented and validated | 25 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / glmer / Type 3 / Chisq / sum | implemented and validated | 25 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / glmer / Type 3 / Chisq / treatment | implemented and validated | 25 | 96.00% | 1 failures; 0 unverified |
| anova | car::Anova / lm / Type 2 / F / sum | implemented and validated | 50 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / lm / Type 2 / F / treatment | implemented and validated | 50 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / lm / Type 3 / F / sum | implemented and validated | 50 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / lm / Type 3 / F / treatment | implemented and validated | 50 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / lmer / Type 2 / Chisq / sum | implemented and validated | 25 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / lmer / Type 2 / Chisq / treatment | implemented and validated | 25 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / lmer / Type 2 / F / sum | implemented and validated | 25 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / lmer / Type 3 / Chisq / treatment | implemented and validated | 25 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / lmer / Type 3 / F / sum | implemented and validated | 25 | 100.00% | 0 failures; 0 unverified |
| anova | car::Anova / lmer / Type 3 / F / treatment | implemented and validated | 25 | 96.00% | 1 failures; 0 unverified |
| emm | contrast / consec / fdr | implemented and validated | 27 | 100.00% | 0 failures; 0 unverified |
| emm | contrast / consec / tukey | implemented and validated | 9 | 100.00% | 0 failures; 0 unverified |
| emm | contrast / custom / fdr | implemented and validated | 18 | 100.00% | 0 failures; 0 unverified |
| emm | contrast / custom / tukey | implemented and validated | 6 | 100.00% | 0 failures; 0 unverified |
| emm | contrast / pairwise / bonferroni | implemented and validated | 27 | 100.00% | 0 failures; 0 unverified |
| emm | contrast / pairwise / holm | implemented and validated | 9 | 100.00% | 0 failures; 0 unverified |
| emm | contrast / poly / bonferroni | implemented and validated | 27 | 100.00% | 0 failures; 0 unverified |
| emm | contrast / poly / holm | implemented and validated | 6 | 100.00% | 0 failures; 0 unverified |
| emm | contrast / revpairwise / fdr | implemented and validated | 27 | 100.00% | 0 failures; 0 unverified |
| emm | contrast / revpairwise / tukey | implemented and validated | 9 | 100.00% | 0 failures; 0 unverified |
| emm | contrast / trt.vs.ctrl / bonferroni | implemented and validated | 27 | 100.00% | 0 failures; 0 unverified |
| emm | contrast / trt.vs.ctrl / holm | implemented and validated | 9 | 100.00% | 0 failures; 0 unverified |
| emm | emmeans / glmer / asymptotic | implemented and validated | 3 | 100.00% | 0 failures; 0 unverified |
| emm | emmeans / glmer / kenward-roger | implemented and validated | 4 | 100.00% | 0 failures; 0 unverified |
| emm | emmeans / glmer / satterthwaite | implemented and validated | 3 | 100.00% | 0 failures; 0 unverified |
| emm | emmeans / gls / asymptotic | implemented and validated | 3 | 100.00% | 0 failures; 0 unverified |
| emm | emmeans / gls / kenward-roger | implemented and validated | 4 | 100.00% | 0 failures; 0 unverified |
| emm | emmeans / gls / satterthwaite | implemented and validated | 3 | 100.00% | 0 failures; 0 unverified |
| emm | emmeans / lmer / asymptotic | implemented and validated | 3 | 100.00% | 0 failures; 0 unverified |
| emm | emmeans / lmer / kenward-roger | implemented and validated | 4 | 100.00% | 0 failures; 0 unverified |
| emm | emmeans / lmer / satterthwaite | implemented and validated | 3 | 100.00% | 0 failures; 0 unverified |
| emm | emmeans / response / at | implemented and validated | 48 | 100.00% | 0 failures; 0 unverified |
| emm | emmeans / weights cells | implemented and validated | 51 | 100.00% | 0 failures; 0 unverified |
| emm | emmeans / weights equal | implemented and validated | 51 | 100.00% | 0 failures; 0 unverified |
| emm | emmeans / weights flat | implemented and validated | 51 | 100.00% | 0 failures; 0 unverified |
| emm | emmeans / weights outer | implemented and validated | 51 | 100.00% | 0 failures; 0 unverified |
| emm | emmeans / weights proportional | implemented and validated | 51 | 100.00% | 0 failures; 0 unverified |
| emm | emtrends / glmer / asymptotic | implemented and validated | 4 | 100.00% | 0 failures; 0 unverified |
| emm | emtrends / glmer / kenward-roger | implemented and validated | 3 | 100.00% | 0 failures; 0 unverified |
| emm | emtrends / glmer / satterthwaite | implemented and validated | 3 | 100.00% | 0 failures; 0 unverified |
| emm | emtrends / gls / asymptotic | implemented and validated | 4 | 100.00% | 0 failures; 0 unverified |
| emm | emtrends / gls / kenward-roger | implemented and validated | 3 | 100.00% | 0 failures; 0 unverified |
| emm | emtrends / gls / satterthwaite | implemented and validated | 3 | 100.00% | 0 failures; 0 unverified |
| emm | emtrends / lmer / asymptotic | implemented and validated | 4 | 100.00% | 0 failures; 0 unverified |
| emm | emtrends / lmer / kenward-roger | implemented and validated | 3 | 100.00% | 0 failures; 0 unverified |
| emm | emtrends / lmer / satterthwaite | implemented and validated | 3 | 100.00% | 0 failures; 0 unverified |
| emm | emtrends / numeric slope | implemented and validated | 48 | 100.00% | 0 failures; 0 unverified |
| emm | joint_tests / factorial | implemented and validated | 48 | 100.00% | 0 failures; 0 unverified |
| emm | joint_tests / glmer / asymptotic | implemented and validated | 1 | 100.00% | 0 failures; 0 unverified |
| emm | joint_tests / glmer / kenward-roger | implemented and validated | 1 | 100.00% | 0 failures; 0 unverified |
| emm | joint_tests / glmer / satterthwaite | implemented and validated | 1 | 100.00% | 0 failures; 0 unverified |
| emm | joint_tests / gls / asymptotic | implemented and validated | 1 | 100.00% | 0 failures; 0 unverified |
| emm | joint_tests / gls / kenward-roger | implemented and validated | 1 | 100.00% | 0 failures; 0 unverified |
| emm | joint_tests / gls / satterthwaite | implemented and validated | 1 | 100.00% | 0 failures; 0 unverified |
| emm | joint_tests / lmer / asymptotic | implemented and validated | 1 | 100.00% | 0 failures; 0 unverified |
| emm | joint_tests / lmer / kenward-roger | implemented and validated | 1 | 100.00% | 0 failures; 0 unverified |
| emm | joint_tests / lmer / satterthwaite | implemented and validated | 1 | 100.00% | 0 failures; 0 unverified |
| emm | pairs / glmer / asymptotic | implemented and validated | 3 | 100.00% | 0 failures; 0 unverified |
| emm | pairs / glmer / kenward-roger | implemented and validated | 3 | 100.00% | 0 failures; 0 unverified |
| emm | pairs / glmer / satterthwaite | implemented and validated | 4 | 100.00% | 0 failures; 0 unverified |
| emm | pairs / gls / asymptotic | implemented and validated | 3 | 66.67% | 1 failures; 0 unverified |
| emm | pairs / gls / kenward-roger | implemented and validated | 3 | 100.00% | 0 failures; 0 unverified |
| emm | pairs / gls / satterthwaite | implemented and validated | 4 | 100.00% | 0 failures; 0 unverified |
| emm | pairs / lmer / asymptotic | implemented and validated | 3 | 100.00% | 0 failures; 0 unverified |
| emm | pairs / lmer / kenward-roger | implemented and validated | 3 | 100.00% | 0 failures; 0 unverified |
| emm | pairs / lmer / satterthwaite | implemented and validated | 4 | 100.00% | 0 failures; 0 unverified |
| emm | regrid / response / glm | implemented and validated | 16 | 100.00% | 0 failures; 0 unverified |
| emm | regrid / response / lm | implemented and validated | 8 | 100.00% | 0 failures; 0 unverified |
| emm | response / glmer / asymptotic | implemented and validated | 3 | 100.00% | 0 failures; 0 unverified |
| emm | response / glmer / kenward-roger | implemented and validated | 4 | 100.00% | 0 failures; 0 unverified |
| emm | response / glmer / satterthwaite | implemented and validated | 3 | 100.00% | 0 failures; 0 unverified |
| emm | response / gls / asymptotic | implemented and validated | 3 | 100.00% | 0 failures; 0 unverified |
| emm | response / gls / kenward-roger | implemented and validated | 4 | 100.00% | 0 failures; 0 unverified |
| emm | response / gls / satterthwaite | implemented and validated | 3 | 100.00% | 0 failures; 0 unverified |
| emm | response / lmer / asymptotic | implemented and validated | 3 | 100.00% | 0 failures; 0 unverified |
| emm | response / lmer / kenward-roger | implemented and validated | 4 | 100.00% | 0 failures; 0 unverified |
| emm | response / lmer / satterthwaite | implemented and validated | 3 | 100.00% | 0 failures; 0 unverified |
| gam | gam / additive / Gamma / GCV.Cp | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / additive / Gamma / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / additive / Gamma / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / additive / binomial / GCV.Cp | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / additive / binomial / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / additive / binomial / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / additive / gaussian / GCV.Cp | implemented and validated | 10 | 90.00% | 1 failures; 0 unverified |
| gam | gam / additive / gaussian / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / additive / gaussian / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / additive / poisson / GCV.Cp | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / additive / poisson / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / additive / poisson / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / continuous-by / Gamma / GCV.Cp | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / continuous-by / Gamma / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / continuous-by / Gamma / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / continuous-by / binomial / GCV.Cp | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / continuous-by / binomial / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / continuous-by / binomial / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / continuous-by / gaussian / GCV.Cp | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / continuous-by / gaussian / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / continuous-by / gaussian / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / continuous-by / poisson / GCV.Cp | implemented and validated | 10 | 90.00% | 1 failures; 0 unverified |
| gam | gam / continuous-by / poisson / ML | implemented and validated | 10 | 90.00% | 1 failures; 0 unverified |
| gam | gam / continuous-by / poisson / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cr / Gamma / GCV.Cp | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cr / Gamma / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cr / Gamma / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cr / binomial / GCV.Cp | implemented and validated | 10 | 90.00% | 1 failures; 0 unverified |
| gam | gam / cr / binomial / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cr / binomial / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cr / gaussian / GCV.Cp | implemented and validated | 10 | 90.00% | 1 failures; 0 unverified |
| gam | gam / cr / gaussian / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cr / gaussian / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cr / poisson / GCV.Cp | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cr / poisson / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cr / poisson / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cs / Gamma / GCV.Cp | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cs / Gamma / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cs / Gamma / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cs / binomial / GCV.Cp | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cs / binomial / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cs / binomial / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cs / gaussian / GCV.Cp | implemented and validated | 10 | 90.00% | 1 failures; 0 unverified |
| gam | gam / cs / gaussian / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cs / gaussian / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cs / poisson / GCV.Cp | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cs / poisson / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / cs / poisson / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / factor-by / Gamma / GCV.Cp | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / factor-by / Gamma / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / factor-by / Gamma / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / factor-by / binomial / GCV.Cp | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / factor-by / binomial / ML | implemented and validated | 10 | 90.00% | 1 failures; 0 unverified |
| gam | gam / factor-by / binomial / REML | implemented and validated | 10 | 80.00% | 2 failures; 0 unverified |
| gam | gam / factor-by / gaussian / GCV.Cp | implemented and validated | 10 | 80.00% | 2 failures; 0 unverified |
| gam | gam / factor-by / gaussian / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / factor-by / gaussian / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / factor-by / poisson / GCV.Cp | implemented and validated | 10 | 90.00% | 1 failures; 0 unverified |
| gam | gam / factor-by / poisson / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / factor-by / poisson / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ps / Gamma / GCV.Cp | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ps / Gamma / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ps / Gamma / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ps / binomial / GCV.Cp | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ps / binomial / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ps / binomial / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ps / gaussian / GCV.Cp | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ps / gaussian / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ps / gaussian / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ps / poisson / GCV.Cp | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ps / poisson / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ps / poisson / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / re / Gamma / GCV.Cp | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / re / Gamma / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / re / Gamma / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / re / binomial / GCV.Cp | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / re / binomial / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / re / binomial / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / re / gaussian / GCV.Cp | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / re / gaussian / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / re / gaussian / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / re / poisson / GCV.Cp | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / re / poisson / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / re / poisson / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / te / Gamma / GCV.Cp | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / te / Gamma / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / te / Gamma / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / te / binomial / GCV.Cp | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / te / binomial / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / te / binomial / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / te / gaussian / GCV.Cp | implemented and validated | 10 | 80.00% | 2 failures; 0 unverified |
| gam | gam / te / gaussian / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / te / gaussian / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / te / poisson / GCV.Cp | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / te / poisson / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / te / poisson / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ti / Gamma / GCV.Cp | implemented and validated | 10 | 90.00% | 1 failures; 0 unverified |
| gam | gam / ti / Gamma / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ti / Gamma / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ti / binomial / GCV.Cp | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ti / binomial / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ti / binomial / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ti / gaussian / GCV.Cp | implemented and validated | 10 | 90.00% | 1 failures; 0 unverified |
| gam | gam / ti / gaussian / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ti / gaussian / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ti / poisson / GCV.Cp | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ti / poisson / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / ti / poisson / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / tp / Gamma / GCV.Cp | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / tp / Gamma / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / tp / Gamma / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / tp / binomial / GCV.Cp | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / tp / binomial / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / tp / binomial / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / tp / gaussian / GCV.Cp | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / tp / gaussian / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / tp / gaussian / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / tp / poisson / GCV.Cp | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / tp / poisson / ML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| gam | gam / tp / poisson / REML | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| glmer | glmer / binomial-cloglog | implemented and validated | 150 | 95.33% | 7 failures; 0 unverified |
| glmer | glmer / binomial-logit | implemented and validated | 150 | 97.33% | 4 failures; 0 unverified |
| glmer | glmer / binomial-logit fractional binary | implemented and validated | 1 | 100.00% | 0 failures; 0 unverified |
| glmer | glmer / binomial-logit fractional cbind | implemented and validated | 1 | 100.00% | 0 failures; 0 unverified |
| glmer | glmer / binomial-logit fractional proportion | implemented and validated | 1 | 100.00% | 0 failures; 0 unverified |
| glmer | glmer / binomial-probit | implemented and validated | 150 | 96.67% | 5 failures; 0 unverified |
| glmer | glmer / poisson-log | implemented and validated | 150 | 98.67% | 2 failures; 0 unverified |
| gls | gls / corAR1 + varIdent / ML | implemented and validated | 30 | 100.00% | 0 failures; 0 unverified |
| gls | gls / corAR1 + varIdent / REML | implemented and validated | 30 | 100.00% | 0 failures; 0 unverified |
| gls | gls / corAR1 / ML | implemented and validated | 30 | 100.00% | 0 failures; 0 unverified |
| gls | gls / corAR1 / REML | implemented and validated | 30 | 100.00% | 0 failures; 0 unverified |
| gls | gls / corARMA / ML | implemented and validated | 30 | 96.67% | 1 failures; 0 unverified |
| gls | gls / corARMA / REML | implemented and validated | 30 | 100.00% | 0 failures; 0 unverified |
| gls | gls / corCompSymm + varExp / ML | implemented and validated | 30 | 93.33% | 2 failures; 0 unverified |
| gls | gls / corCompSymm + varExp / REML | implemented and validated | 30 | 93.33% | 2 failures; 0 unverified |
| gls | gls / corCompSymm / ML | implemented and validated | 30 | 100.00% | 0 failures; 0 unverified |
| gls | gls / corCompSymm / REML | implemented and validated | 30 | 100.00% | 0 failures; 0 unverified |
| gls | gls / corSymm / ML | implemented and validated | 30 | 86.67% | 4 failures; 0 unverified |
| gls | gls / corSymm / REML | implemented and validated | 30 | 96.67% | 1 failures; 0 unverified |
| gls | gls / independent / ML | implemented and validated | 30 | 100.00% | 0 failures; 0 unverified |
| gls | gls / independent / REML | implemented and validated | 30 | 100.00% | 0 failures; 0 unverified |
| gls | gls / independentvarExp / ML | implemented and validated | 30 | 90.00% | 3 failures; 0 unverified |
| gls | gls / independentvarExp / REML | implemented and validated | 30 | 100.00% | 0 failures; 0 unverified |
| gls | gls / independentvarIdent / ML | implemented and validated | 30 | 100.00% | 0 failures; 0 unverified |
| gls | gls / independentvarIdent / REML | implemented and validated | 30 | 100.00% | 0 failures; 0 unverified |
| gls | gls / varPower / ML | implemented and validated | 30 | 100.00% | 0 failures; 0 unverified |
| gls | gls / varPower / REML | implemented and validated | 30 | 100.00% | 0 failures; 0 unverified |
| inference | lmerTest / Kenward-Roger / anova / Type 1 | implemented and validated | 68 | 100.00% | 0 failures; 0 unverified |
| inference | lmerTest / Kenward-Roger / anova / Type 2 | implemented and validated | 66 | 96.97% | 2 failures; 0 unverified |
| inference | lmerTest / Kenward-Roger / anova / Type 3 | implemented and validated | 66 | 100.00% | 0 failures; 0 unverified |
| inference | lmerTest / Satterthwaite / anova / Type 1 | implemented and validated | 68 | 100.00% | 0 failures; 0 unverified |
| inference | lmerTest / Satterthwaite / anova / Type 2 | implemented and validated | 66 | 96.97% | 2 failures; 0 unverified |
| inference | lmerTest / Satterthwaite / anova / Type 3 | implemented and validated | 66 | 100.00% | 0 failures; 0 unverified |
| inference | lmerTest / Satterthwaite / fit / Type 1 | implemented and validated | 68 | 100.00% | 0 failures; 0 unverified |
| inference | lmerTest / Satterthwaite / fit / Type 2 | implemented and validated | 66 | 96.97% | 2 failures; 0 unverified |
| inference | lmerTest / Satterthwaite / fit / Type 3 | implemented and validated | 66 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (0+x\|g) | implemented and validated | 65 | 95.38% | 3 failures; 0 unverified |
| lmer | lmer / ML / (0+x\|g) / missingness | implemented and validated | 4 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (0+x\|g) / missingness / weights | implemented and validated | 1 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (0+x\|g) / treatment | implemented and validated | 16 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (0+x\|g) / treatment / missingness | implemented and validated | 1 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (0+x\|g) / treatment / weights | implemented and validated | 3 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (0+x\|g) / weights | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) | implemented and validated | 45 | 97.78% | 1 failures; 0 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / missingness | implemented and validated | 1 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / missingness / offset | implemented and validated | 2 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / missingness / weights | implemented and validated | 1 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / offset | implemented and validated | 21 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / treatment | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / treatment / missingness | implemented and validated | 2 | 50.00% | 1 failures; 0 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / treatment / offset | implemented and validated | 5 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / treatment / weights | implemented and validated | 2 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / treatment / weights / offset | implemented and validated | 1 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / weights | implemented and validated | 6 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (1\|g)+(1\|h) / weights / offset | implemented and validated | 4 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (x\|g) | implemented and validated | 64 | 96.88% | 2 failures; 0 unverified |
| lmer | lmer / ML / (x\|g) / missingness | implemented and validated | 4 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (x\|g) / missingness / weights | implemented and validated | 1 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (x\|g) / treatment | implemented and validated | 16 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (x\|g) / treatment / missingness | implemented and validated | 1 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (x\|g) / treatment / weights | implemented and validated | 3 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / ML / (x\|g) / weights | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (1\|g) | implemented and validated | 38 | 97.37% | 1 failures; 0 unverified |
| lmer | lmer / REML / (1\|g) / boundary | implemented and validated | 15 | 93.33% | 1 failures; 0 unverified |
| lmer | lmer / REML / (1\|g) / missingness | implemented and validated | 4 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (1\|g) / offset | implemented and validated | 22 | 90.91% | 2 failures; 0 unverified |
| lmer | lmer / REML / (1\|g) / sum | implemented and validated | 11 | 90.91% | 1 failures; 0 unverified |
| lmer | lmer / REML / (1\|g) / sum / missingness / offset | implemented and validated | 1 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (1\|g) / sum / offset | implemented and validated | 5 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (1\|g) / sum / weights | implemented and validated | 2 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (1\|g) / sum / weights / offset | implemented and validated | 1 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (1\|g) / weights | implemented and validated | 8 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (1\|g) / weights / offset | implemented and validated | 3 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (1\|g/h) | implemented and validated | 60 | 96.67% | 2 failures; 0 unverified |
| lmer | lmer / REML / (1\|g/h) / missingness | implemented and validated | 4 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (1\|g/h) / missingness / weights | implemented and validated | 1 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (1\|g/h) / sum | implemented and validated | 17 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (1\|g/h) / sum / missingness | implemented and validated | 1 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (1\|g/h) / sum / weights | implemented and validated | 2 | 50.00% | 1 failures; 0 unverified |
| lmer | lmer / REML / (1\|g/h) / weights | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (x\|\|g) | implemented and validated | 60 | 98.33% | 1 failures; 0 unverified |
| lmer | lmer / REML / (x\|\|g) / missingness | implemented and validated | 5 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (x\|\|g) / sum | implemented and validated | 17 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (x\|\|g) / sum / missingness / weights | implemented and validated | 1 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (x\|\|g) / sum / weights | implemented and validated | 2 | 100.00% | 0 failures; 0 unverified |
| lmer | lmer / REML / (x\|\|g) / weights | implemented and validated | 10 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~0 / disp ~1 / (1\|g) | implemented and validated | 12 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~0 / disp ~1 / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~0 / disp ~x + f / (1\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~0 / disp ~x + f / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~0 / disp ~x / (1\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~0 / disp ~x / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~1 / disp ~1 / (1\|g) | implemented and validated | 12 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~1 / disp ~1 / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~1 / disp ~x + f / (1\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~1 / disp ~x + f / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~1 / disp ~x / (1\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~1 / disp ~x / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~z + x / disp ~1 / (1\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~z + x / disp ~1 / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~z + x / disp ~x + f / (1\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~z + x / disp ~x + f / (x\|g) | implemented and validated | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~z + x / disp ~x / (1\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / beta / zi ~z + x / disp ~x / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~0 / disp ~1 / (1\|g) | implemented and validated | 12 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~0 / disp ~1 / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~0 / disp ~x + f / (1\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~0 / disp ~x + f / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~0 / disp ~x / (1\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~0 / disp ~x / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~1 / disp ~1 / (1\|g) | implemented and validated | 12 | 91.67% | 1 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~1 / disp ~1 / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~1 / disp ~x + f / (1\|g) | implemented and validated | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~1 / disp ~x + f / (x\|g) | implemented and validated | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~1 / disp ~x / (1\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~1 / disp ~x / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~z + x / disp ~1 / (1\|g) | implemented and validated | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~z + x / disp ~1 / (x\|g) | implemented and validated | 11 | 72.73% | 3 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~z + x / disp ~x + f / (1\|g) | implemented and validated | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~z + x / disp ~x + f / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~z + x / disp ~x / (1\|g) | implemented and validated | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / binomial / zi ~z + x / disp ~x / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~0 / disp ~1 / (1\|g) | implemented and validated | 12 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~0 / disp ~1 / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~0 / disp ~x + f / (1\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~0 / disp ~x + f / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~0 / disp ~x / (1\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~0 / disp ~x / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~1 / disp ~1 / (1\|g) | implemented and validated | 12 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~1 / disp ~1 / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~1 / disp ~x + f / (1\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~1 / disp ~x + f / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~1 / disp ~x / (1\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~1 / disp ~x / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~z + x / disp ~1 / (1\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~z + x / disp ~1 / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~z + x / disp ~x + f / (1\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~z + x / disp ~x + f / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~z + x / disp ~x / (1\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / gaussian / zi ~z + x / disp ~x / (x\|g) | implemented and validated | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~0 / disp ~1 / (1\|g) | implemented and validated | 12 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~0 / disp ~1 / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~0 / disp ~x + f / (1\|g) | implemented and validated | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~0 / disp ~x + f / (x\|g) | implemented and validated | 11 | 81.82% | 2 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~0 / disp ~x / (1\|g) | implemented and validated | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~0 / disp ~x / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~1 / disp ~1 / (1\|g) | implemented and validated | 12 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~1 / disp ~1 / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~1 / disp ~x + f / (1\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~1 / disp ~x + f / (x\|g) | implemented and validated | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~1 / disp ~x / (1\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~1 / disp ~x / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~z + x / disp ~1 / (1\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~z + x / disp ~1 / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~z + x / disp ~x + f / (1\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~z + x / disp ~x + f / (x\|g) | implemented and validated | 11 | 81.82% | 2 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~z + x / disp ~x / (1\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom1 / zi ~z + x / disp ~x / (x\|g) | implemented and validated | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~0 / disp ~1 / (1\|g) | implemented and validated | 12 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~0 / disp ~1 / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~0 / disp ~x + f / (1\|g) | implemented and validated | 11 | 81.82% | 2 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~0 / disp ~x + f / (x\|g) | implemented and validated | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~0 / disp ~x / (1\|g) | implemented and validated | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~0 / disp ~x / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~1 / disp ~1 / (1\|g) | implemented and validated | 12 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~1 / disp ~1 / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~1 / disp ~x + f / (1\|g) | implemented and validated | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~1 / disp ~x + f / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~1 / disp ~x / (1\|g) | implemented and validated | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~1 / disp ~x / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~z + x / disp ~1 / (1\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~z + x / disp ~1 / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~z + x / disp ~x + f / (1\|g) | implemented and validated | 11 | 81.82% | 2 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~z + x / disp ~x + f / (x\|g) | implemented and validated | 11 | 90.91% | 1 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~z + x / disp ~x / (1\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / nbinom2 / zi ~z + x / disp ~x / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~0 / disp ~1 / (1\|g) | implemented and validated | 12 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~0 / disp ~1 / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~0 / disp ~x + f / (1\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~0 / disp ~x + f / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~0 / disp ~x / (1\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~0 / disp ~x / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~1 / disp ~1 / (1\|g) | implemented and validated | 12 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~1 / disp ~1 / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~1 / disp ~x + f / (1\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~1 / disp ~x + f / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~1 / disp ~x / (1\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~1 / disp ~x / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~z + x / disp ~1 / (1\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~z + x / disp ~1 / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~z + x / disp ~x + f / (1\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~z + x / disp ~x + f / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~z + x / disp ~x / (1\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |
| tmb | glmmTMB / poisson / zi ~z + x / disp ~x / (x\|g) | implemented and validated | 11 | 100.00% | 0 failures; 0 unverified |

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

Total: 6256 synthetic cases, 6155 recorded passes.

Stage 1 requires at least 300 cases per module, 3,000 total and 98% passing.
Stage 2 separately requires at least 2,000 cases and 98% passing, with no new Stage 1 failures.

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
