# Black-box oracle

All numerical R observations pass through `run_case.R`. This script accepts a
single JSON case or `{ "cases": [...] }`. Each case specifies a public model
function, formula, data columns (or CSV), arguments, and an extraction operation.
It writes a JSON array of public results and captured warnings.

```sh
Rscript oracle/run_case.R oracle/cache/input.json oracle/cache/output.json
```

Supported fit calls are `lmer`, `glmer`, `gls`, `lm`, and `glm`. Operations
include `fit`, `anova`, `Anova`, `emmeans`, `emtrends`, `joint_tests`, `predict`,
`data`, and `versions`. Covariance constructors are explicitly allowlisted.
R's contrasts, mixed-model optimizer, GLS optimizer, and marginal-means degrees
of freedom are explicitly configured.

Synthetic golden files include their full inputs and observations. Built-in R
example data may only be saved beneath ignored `oracle/cache/`. Regeneration
requires development packages listed in `versions.json`, when available.
