"""Fixed-effect prediction adapters for marginal mean calculations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import expit, ndtr

from rparity.formula import as_dataframe


@dataclass
class ModelAdapter:
    model: Any
    data: pd.DataFrame
    beta: NDArray[np.float64]
    covariance: NDArray[np.float64]
    design: NDArray[np.float64]
    names: list[str]
    variables: list[str]
    df: float
    link: str
    formula: str
    mixed: bool

    def matrix(self, data: pd.DataFrame) -> NDArray[np.float64]:
        """Build prediction rows with the fitting-time contrast coding."""
        if hasattr(self.model, "fixed_spec"):
            matrix = self.model.fixed_spec.get_model_matrix(data)
            if hasattr(matrix, "rhs"):
                matrix = matrix.rhs
            if hasattr(self.model, "_column_indices"):
                return np.asarray(matrix, dtype=float)[:, self.model._column_indices]
            return np.asarray(matrix, dtype=float)
        info = getattr(self.model.model.data, "design_info", None)
        if info is None:
            info = getattr(self.model.model.data, "model_spec", None)
        if info is not None:
            if hasattr(info, "get_model_matrix"):
                return np.asarray(info.get_model_matrix(data), dtype=float)
            from patsy import build_design_matrices

            return np.asarray(build_design_matrices([info], data)[0], dtype=float)
        if hasattr(self.model, "model"):
            output = np.empty((len(data), len(self.names)))
            for j, name in enumerate(self.names):
                output[:, j] = 1.0 if name in {"Intercept", "const"} else data[name]
            return output
        raise TypeError("The fitted model does not expose a reusable fixed-effect design.")

    def offsets(self, data: pd.DataFrame) -> NDArray[np.float64]:
        """Evaluate model formula offsets at the reference-grid values."""
        expressions = re.findall(r"offset\(([^()]*(?:\([^()]*\)[^()]*)*)\)", self.formula)
        output = np.zeros(len(data))
        for expression in expressions:
            namespace: dict[str, Any] = {str(name): data[name] for name in data}
            namespace.update({"np": np, "log": np.log, "exp": np.exp, "sqrt": np.sqrt})
            # Formula text is supplied by the caller, not downloaded content.
            output += np.asarray(eval(expression, {"__builtins__": {}}, namespace), dtype=float)
        if not expressions:
            offset = getattr(self.model, "offset", None)
            if offset is None and hasattr(self.model, "model"):
                offset = getattr(self.model.model, "offset", None)
            if offset is not None:
                output += float(np.mean(offset))
        return output

    def inverse(self, value: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.link in {"logit", "logistic"}:
            return np.asarray(expit(value), dtype=float)
        if self.link == "probit":
            return np.asarray(ndtr(value), dtype=float)
        if self.link in {"cloglog", "c-log-log"}:
            return -np.expm1(-np.exp(value))
        if self.link == "log":
            return np.exp(value)
        if self.link == "log10":
            return np.power(10.0, value)
        if self.link == "log2":
            return np.power(2.0, value)
        if self.link in {"inverse", "inverse_power"}:
            return 1 / value
        if self.link == "sqrt":
            return value**2
        return value.copy()

    def derivative(self, value: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.link in {"logit", "logistic"}:
            probability = np.asarray(expit(value), dtype=float)
            return probability * (1 - probability)
        if self.link == "probit":
            return np.exp(-value**2 / 2) / np.sqrt(2 * np.pi)
        if self.link in {"cloglog", "c-log-log"}:
            return np.exp(value - np.exp(value))
        if self.link == "log":
            return np.exp(value)
        if self.link == "log10":
            return np.log(10.0) * np.power(10.0, value)
        if self.link == "log2":
            return np.log(2.0) * np.power(2.0, value)
        if self.link in {"inverse", "inverse_power"}:
            return -1 / value**2
        if self.link == "sqrt":
            return 2 * value
        return np.ones_like(value)


def adapt(model: Any, data: Any = None) -> ModelAdapter:
    """Extract a common fixed-effect representation from a fitted model."""
    native = hasattr(model, "fixed_spec")
    if native:
        frame = model.data if data is None else data
        beta = np.asarray(model.beta, dtype=float)
        covariance = np.asarray(model.cov_beta, dtype=float)
        names = list(model.coef_names)
        design = np.asarray(model.X, dtype=float)
        formula = str(getattr(model, "fixed_formula", model.formula))
        original_formula = str(model.formula)
        mixed = "|" in original_formula
        df = float(getattr(model, "df_resid", len(frame) - len(beta)))
        family = getattr(model, "family", None)
        link = getattr(model, "link", "identity")
    elif hasattr(model, "model") and hasattr(model, "params"):
        frame = data if data is not None else getattr(model.model.data, "frame", None)
        if frame is None:
            frame = pd.DataFrame(model.model.exog, columns=model.model.exog_names)
        beta = np.asarray(model.params, dtype=float)
        covariance = np.asarray(model.cov_params(), dtype=float)
        names = list(model.model.exog_names)
        design = np.asarray(model.model.exog, dtype=float)
        formula = str(getattr(model.model, "formula", "~ " + " + ".join(names)))
        original_formula = formula
        mixed = False
        df = float(model.df_resid)
        family = getattr(model.model, "family", None)
        link = getattr(family, "link", "identity")
    else:
        raise TypeError("Expected a rparity fitted model or a statsmodels OLS/GLM result.")
    frame = as_dataframe(frame)
    if not native and data is None:
        missing = getattr(model.model.data, "missing_row_idx", [])
        if missing:
            frame = frame.drop(frame.index[missing])
    if not isinstance(link, str):
        link = type(link).__name__.lower()
    if link.lower() == "identity":
        response = formula.split("~", 1)[0].strip()
        transformation = re.match(r"(?:np\.)?(log|log10|log2|sqrt)\s*\(", response)
        if transformation:
            link = transformation.group(1)
    if family is not None and str(family).lower() not in {"gaussian", "normal", "none"}:
        family_name = family if isinstance(family, str) else type(family).__name__
        if str(family_name).lower() not in {"gaussian", "normal"}:
            df = float("inf")
    rhs = formula.split("~", 1)[-1]
    rhs += " " + " ".join(re.findall(
        r"offset\(([^()]*(?:\([^()]*\)[^()]*)*)\)", original_formula,
    ))
    variables = [
        str(name) for name in frame
        if re.search(r"(?<![\w])" + re.escape(str(name)) + r"(?![\w])", rhs)
    ]
    return ModelAdapter(
        model, frame, beta, covariance, design, names, variables, df,
        link.lower(), original_formula, mixed,
    )
