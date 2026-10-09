"""Estimate a nonlinear temperature effect and predict revenue with uncertainty.

Run with ``python examples/generalized_additive_models.py``.
"""

import numpy as np
import pandas as pd

from rparity import gam


def main() -> None:
    rng = np.random.default_rng(120)
    temperature = rng.uniform(0, 30, 180)
    promotion = rng.integers(0, 2, len(temperature))
    data = pd.DataFrame({
        "revenue": (
            100 + 3 * temperature + 20 * np.sin(temperature / 5)
            + 15 * promotion + rng.normal(0, 6, len(temperature))
        ),
        "temperature": temperature,
        "promotion": promotion,
    })

    # A cubic regression spline captures the nonlinear effect. REML selects
    # its smoothing parameter while adjusting for the promotion indicator.
    model = gam(
        "revenue ~ promotion + s(temperature, bs='cr', k=10)",
        data=data,
        method="REML",
    )
    print(model.summary())

    forecast = pd.DataFrame({
        "temperature": [5.0, 10.0, 15.0, 20.0, 25.0],
        "promotion": 0,
    })
    prediction = model.predict(forecast, type="response", se_fit=True)
    forecast["expected_revenue"] = prediction["fit"]
    forecast["mean_standard_error"] = prediction["se.fit"]
    # These are approximate pointwise intervals for the expected response,
    # conditional on the estimated smoothing parameter, not prediction intervals.
    forecast["lower_95"] = prediction["fit"] - 1.96 * prediction["se.fit"]
    forecast["upper_95"] = prediction["fit"] + 1.96 * prediction["se.fit"]
    print("\nExpected revenue without a promotion:")
    print(forecast.to_string(index=False))

    # Plot these columns in your preferred plotting library, or export to CSV.
    effect = model.partial_effects("s(temperature)", n=5)
    print("\nCentered temperature effect and standard error:")
    print(effect[["temperature", "fit", "se"]].to_string(index=False))


if __name__ == "__main__":
    main()
