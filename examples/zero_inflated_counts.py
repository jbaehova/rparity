"""Model overdispersed counts with extra zeros and site-level clustering.

Run with ``python examples/zero_inflated_counts.py``.
"""

import numpy as np
import pandas as pd
from scipy.special import expit

from rparity import Anova, glmmTMB


def main() -> None:
    rng = np.random.default_rng(713)
    site = np.repeat(np.arange(12), 24)
    treated = np.tile(np.repeat([0, 1], 12), 12)
    temperature = rng.normal(0, 1, len(site))
    site_effect = rng.normal(0, 0.35, 12)
    mean = np.exp(1.4 + 0.45 * treated + 0.3 * temperature + site_effect[site])
    dispersion = np.exp(1.0 + 0.45 * treated)
    counts = rng.negative_binomial(dispersion, dispersion / (dispersion + mean))
    structural_zero = rng.random(len(site)) < expit(-1.7 + 0.9 * treated)
    data = pd.DataFrame({
        "count": np.where(structural_zero, 0, counts),
        "treated": treated,
        "temperature": temperature,
        "site": site.astype(str),
    })

    # NB2 has variance mu + mu**2 / dispersion. The three formulas describe
    # the count mean, the extra-zero probability, and dispersion separately.
    model = glmmTMB(
        "count ~ treated + temperature + (1 | site)",
        data=data,
        family="nbinom2",
        ziformula="~ treated",
        dispformula="~ treated",
    )
    print(model.summary())
    print("\nTests for the conditional count component:")
    print(Anova(model, type=2, component="cond"))
    print("\nTests for the extra-zero component:")
    print(Anova(model, type=2, component="zi"))

    scenarios = pd.DataFrame({
        "treated": [0, 1],
        "temperature": [0.0, 0.0],
        "site": ["0", "0"],
    })
    scenarios["conditional_mean"] = model.predict(
        scenarios, type="conditional", re_form="NA"
    )
    scenarios["extra_zero_probability"] = model.predict(scenarios, type="zprob")
    scenarios["expected_count"] = model.predict(
        scenarios, type="response", re_form="NA"
    )
    scenarios["dispersion"] = model.predict(scenarios, type="disp")
    # Response means combine both components: (1 - zero_probability) * mu.
    # re_form="NA" sets site random effects to zero.
    print("\nPredictions at the reference temperature:")
    print(scenarios.drop(columns="site").to_string(index=False))


if __name__ == "__main__":
    main()
