# Implementation decisions

- 2026-10-09: Initialize a single local Git repository because the workspace did not contain one. Create a public remote when the scaffold is reviewable, as authorized by TASK.md.
- 2026-10-09: Use dense SciPy linear algebra first. Optional sparse or autodiff dependencies are not required for Stage 1.
- 2026-10-09: Keep version 0.1.0.dev0 and the in-progress badge until every Stage 1 acceptance criterion is verified.
