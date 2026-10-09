"""Estimate a longitudinal trend while allowing each subject their own trajectory.

Run with ``python examples/quickstart.py`` after installing rparity.
"""

import numpy as np
import pandas as pd

from rparity import Anova, emmeans, lmer, pairs


def main() -> None:
    # Replace this synthetic DataFrame with your own repeated measurements.
    rng = np.random.default_rng(42)
    subject = np.repeat(np.arange(12), 6)
    day = np.tile(np.arange(6), 12)
    intercepts = rng.normal(0, 12, 12)
    slopes = rng.normal(0, 2, 12)
    data = pd.DataFrame({
        "reaction": (
            250 + 8 * day + intercepts[subject]
            + slopes[subject] * day + rng.normal(0, 5, len(day))
        ),
        "day": day,
        "subject": subject.astype(str),
    })

    # The fixed effect estimates the overall trend. (day | subject) adds
    # correlated subject-specific intercepts and slopes.
    model = lmer("reaction ~ day + (day | subject)", data=data)
    print(model.summary())
    print("\nOverall day effect:")
    print(Anova(model, type=2))

    means = emmeans(model, "day", at={"day": [0, 5]})
    print("\nChange from day 0 to day 5:")
    print(pairs(means).summary(infer=True).to_string(index=False))

    # Set random effects to zero for the population trajectory. Omitting
    # re_form includes fitted random effects for subjects in the training data.
    future = pd.DataFrame({"day": [0, 3, 5], "subject": ["0", "0", "0"]})
    future["population_prediction"] = model.predict(future, re_form="NA")
    future["subject_prediction"] = model.predict(future)
    print("\nPredicted reaction times:")
    print(future.to_string(index=False))


if __name__ == "__main__":
    main()
