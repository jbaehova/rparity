"""README quick start, runnable with only runtime dependencies installed."""
import numpy as np
import pandas as pd

from rparity import Anova, emmeans, lmer, pairs

rng = np.random.default_rng(42)
subject = np.repeat(np.arange(12), 6)
days = np.tile(np.arange(6), 12)
intercepts = rng.normal(0, 12, 12)
slopes = rng.normal(0, 2, 12)
data = pd.DataFrame({
    "Reaction": 250 + 8 * days + intercepts[subject]
                + slopes[subject] * days + rng.normal(0, 5, len(days)),
    "Days": days,
    "Subject": subject.astype(str),
})
model = lmer("Reaction ~ Days + (Days | Subject)", data=data)
print(model.summary())
print(Anova(model, type=3))
means = emmeans(model, "Days", at={"Days": [0, 5]})
print(pairs(means, adjust="tukey").summary())
