"""Compare adjusted treatment means in a repeated-measures study.

Run with ``python examples/marginal_means.py``.
"""

import numpy as np
import pandas as pd

from rparity import contrast, emmeans, lmer, pairs


def main() -> None:
    rng = np.random.default_rng(81)
    subject = np.repeat(np.arange(18), 9)
    treatment_index = np.tile(np.repeat(np.arange(3), 3), 18)
    baseline = rng.normal(0, 6, len(subject))
    subject_effect = rng.normal(0, 4, 18)
    treatment_effect = np.asarray([0.0, -3.0, -6.0])
    labels = np.asarray(["control", "low", "high"])
    data = pd.DataFrame({
        "blood_pressure": (
            125 + 0.7 * baseline + treatment_effect[treatment_index]
            + subject_effect[subject] + rng.normal(0, 2, len(subject))
        ),
        "treatment": pd.Categorical(
            labels[treatment_index], categories=labels, ordered=True
        ),
        "baseline": baseline,
        "subject": subject.astype(str),
    })

    model = lmer("blood_pressure ~ treatment + baseline + (1 | subject)", data=data)
    # Evaluate all treatment levels at the same centered baseline, with subject
    # effects set to zero. The default lmer_df uses Kenward-Roger inference.
    means = emmeans(model, "treatment", at={"baseline": [0.0]})
    print("Adjusted blood pressure at centered baseline = 0:")
    print(means.summary().to_string(index=False))

    print("\nAll treatment comparisons with a Tukey adjustment:")
    print(pairs(means, adjust="tukey").summary(infer=True).to_string(index=False))

    # Coefficients follow the explicit factor order: control, low, high.
    high_vs_control = contrast(means, {"high - control": [-1, 0, 1]})
    print("\nPrespecified high-dose versus control comparison:")
    print(high_vs_control.summary(infer=True).to_string(index=False))


if __name__ == "__main__":
    main()
