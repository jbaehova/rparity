"""Structural comparison helpers for independently written R-style summaries."""
from __future__ import annotations

import re


def section_signature(text: str) -> list[str]:
    """Extract headings in order, retaining R's named table column structure.

    Data and optimizer expressions belong to caller provenance. Numeric values
    are verified separately under the statistical tolerances in the project.
    """
    labels = {
        'Scaled residuals:', 'Standardized residuals:', 'Random effects:',
        'Fixed effects:', 'Correlation of Fixed Effects:', 'Coefficients:',
        'Correlation:', 'Parameter estimate(s):', 'Variance function:',
    }
    result: list[str] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped in labels:
            result.append(stripped)
            if index + 1 < len(lines):
                result.append(re.sub(r'\s+', ' ', lines[index + 1].strip()))
        elif stripped.startswith(('AIC', 'Number of obs:', 'Number of Observations:')):
            result.append(re.sub(r'\d+(?:\.\d+)?', '#', re.sub(r'\s+', ' ', stripped)))
    return result
