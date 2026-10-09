"""R-style random-effects parsing and formulaic fixed-effect designs.

Random blocks follow Bates et al. (2015), Section 2. Formula evaluation uses
formulaic, a permissively licensed implementation of model matrices.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from formulaic import model_matrix


@dataclass(frozen=True)
class RandomTerm:
    """One random-effects block and its grouping expression."""

    effects: str
    group: str
    correlated: bool = True


@dataclass(frozen=True)
class ParsedFormula:
    """Fixed formula and expanded random grouping terms."""

    response: str
    fixed: str
    random_terms: list[RandomTerm]
    offsets: list[str] = field(default_factory=list)

    @property
    def fixed_formula(self) -> str:
        """Return the fixed-effects formula after random blocks are removed."""
        return self.fixed


@dataclass
class FixedDesign:
    """Complete-case response, design matrix, and reusable formula specification."""

    X: np.ndarray
    y: np.ndarray
    names: list[str]
    spec: Any
    data: pd.DataFrame
    terms: dict[str, list[int]]


@dataclass
class RandomDesign:
    """Dense grouped model matrix with level-major coefficient ordering."""

    term: RandomTerm
    Z: np.ndarray
    values: np.ndarray
    levels: list[Any]
    names: list[str]
    groups: np.ndarray
    spec: Any = None
    column_indices: list[int] = field(default_factory=list)


def as_dataframe(data: Any) -> pd.DataFrame:
    """Copy pandas or Polars tabular input without requiring Polars at runtime."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if hasattr(data, "to_pandas"):
        try:
            return data.to_pandas()
        except ModuleNotFoundError:
            return pd.DataFrame(data.to_dict(as_series=False))
    return pd.DataFrame(data)


def parse_formula(formula: str) -> ParsedFormula:
    """Parse |, ||, crossed and nested random terms without evaluating code."""
    if not isinstance(formula, str) or "~" not in formula:
        raise ValueError("A model formula must contain '~'.")
    response, rhs = formula.split("~", 1)
    terms: list[RandomTerm] = []
    stack: list[int] = []
    spans: list[tuple[int, int]] = []
    for pos, char in enumerate(rhs):
        if char == "(":
            stack.append(pos)
        elif char == ")":
            if not stack:
                raise ValueError("Unbalanced parentheses in formula.")
            start = stack.pop()
            content = rhs[start + 1:pos]
            if "|" in content:
                parts = re.split(r"\|\|?", content)
                if len(parts) != 2:
                    raise ValueError("Each random term must contain one grouping operator.")
                effects, grouping = (s.strip() for s in parts)
                if not effects or not grouping:
                    raise ValueError("Random terms require effects and grouping variables.")
                groups = [s.strip() for s in grouping.split("/")]
                for depth in range(1, len(groups) + 1):
                    terms.append(RandomTerm(effects, ":".join(reversed(groups[:depth])), "||" not in content))
                spans.append((start, pos + 1))
    if stack:
        raise ValueError("Unbalanced parentheses in formula.")
    for start, end in reversed(spans):
        rhs = rhs[:start] + " " + rhs[end:]
    offsets: list[str] = []
    offset_spans: list[tuple[int, int]] = []
    for match in re.finditer(r"\boffset\(", rhs):
        start = match.end()
        depth = 1
        end = start
        while end < len(rhs) and depth:
            depth += (rhs[end] == "(") - (rhs[end] == ")")
            end += 1
        if depth:
            raise ValueError("Unbalanced formula offset expression.")
        offsets.append(rhs[start:end - 1])
        offset_spans.append((match.start(), end))
    for start, end in reversed(offset_spans):
        rhs = rhs[:start] + " " + rhs[end:]
    rhs = re.sub(r"\+\s*(?=\+|$)", "", rhs).strip()
    rhs = re.sub(r"^\s*\+", "", rhs).strip() or "1"
    if "|" in rhs:
        raise ValueError("Random effects must be enclosed in parentheses.")
    return ParsedFormula(response.strip(), f"{response.strip()} ~ {rhs}", terms, offsets)


def _normalize_fixed(formula: str, contrasts: str) -> str:
    formula = re.sub(r"\bfactor\(", "C(", formula)
    if contrasts not in {"treatment", "sum"}:
        raise ValueError("contrasts must be 'treatment' or 'sum'.")
    return formula


def _coefficient_names(columns: Any, frame: pd.DataFrame) -> list[str]:
    """Translate model-matrix labels into R's public coefficient naming style."""
    names: list[str] = []
    for column in columns:
        name = str(column)
        if name == "Intercept":
            names.append("(Intercept)")
            continue
        def replace_sum(match: re.Match[str]) -> str:
            variable, level = match.group(1), match.group(2)
            series = frame[variable]
            levels = list(series.cat.categories) if isinstance(series.dtype, pd.CategoricalDtype) else sorted(series.dropna().unique(), key=str)
            return variable + str([str(v) for v in levels].index(level) + 1)
        name = re.sub(r"C\((\w+),\s*contr\.sum\)\[S\.([^\]]+)\]", replace_sum, name)
        name = re.sub(r"(?:C\((\w+)\)|(\w+))\[T\.([^\]]+)\]", lambda m: (m.group(1) or m.group(2)) + m.group(3), name)
        names.append(name)
    return names


def build_fixed_design(
    formula: str, data: Any, contrasts: str = "treatment"
) -> FixedDesign:
    """Evaluate fixed effects with treatment or sum categorical contrasts."""
    parsed = parse_formula(formula)
    frame = as_dataframe(data).reset_index(drop=True)
    fixed = _normalize_fixed(parsed.fixed, contrasts)
    if contrasts == "sum":
        categorical = [str(c) for c in frame if (
            isinstance(frame[c].dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(frame[c])
            or pd.api.types.is_string_dtype(frame[c])
        )]
        for col in categorical:
            fixed = re.sub(rf"(?<![\w(]){re.escape(col)}\b(?![\w)])", f"C({col}, contr.sum)", fixed)
    needed = set()
    for term in parsed.random_terms:
        needed.update(re.findall(r"\b[A-Za-z_]\w*\b", term.effects + ":" + term.group))
    needed &= set(frame.columns)
    if needed:
        frame = frame.dropna(subset=sorted(needed))
    matrices = model_matrix(fixed, frame, context={"offset": lambda x: x}, na_action="drop")
    response = np.asarray(matrices.lhs, dtype=float).reshape(-1)
    design = matrices.rhs
    frame = frame.loc[design.index].copy()
    terms = {str(term): list(range(sl.start, sl.stop)) for term, sl in design.model_spec.term_slices.items()}
    names = _coefficient_names(design.columns, frame)
    return FixedDesign(np.asarray(design, dtype=float), response, names, design.model_spec, frame, terms)


def group_values(group: str, data: pd.DataFrame) -> np.ndarray:
    """Evaluate nested group labels while retaining single-column level types."""
    columns = [s.strip() for s in group.split(":")]
    if any(c not in data for c in columns):
        raise ValueError(f"Unknown grouping variable in {group!r}.")
    if data[columns].isna().any().any():
        raise ValueError("Grouping variables cannot contain missing values.")
    if len(columns) == 1:
        return data[columns[0]].to_numpy()
    return data[columns].astype(str).agg(":".join, axis=1).to_numpy()


def build_random_design(formula: ParsedFormula | str, data: pd.DataFrame) -> list[RandomDesign]:
    """Build correlated blocks or independent columns for double-bar terms."""
    parsed = parse_formula(formula) if isinstance(formula, str) else formula
    blocks: list[RandomDesign] = []
    for term in parsed.random_terms:
        mat = model_matrix(_normalize_fixed(term.effects, "treatment"), data, na_action="raise")
        values = np.asarray(mat, dtype=float)
        names = _coefficient_names(mat.columns, data)
        groups = group_values(term.group, data)
        if term.group in data and isinstance(data[term.group].dtype, pd.CategoricalDtype):
            observed = set(groups)
            levels = [level for level in data[term.group].cat.categories if level in observed]
        else:
            try:
                levels = sorted(pd.unique(groups).tolist())
            except TypeError:
                levels = sorted(pd.unique(groups).tolist(), key=str)
        codes = pd.Categorical(groups, categories=levels).codes
        subsets = [list(range(values.shape[1]))] if term.correlated else [[i] for i in range(values.shape[1])]
        for subset in subsets:
            val = values[:, subset]
            width = len(subset)
            z = np.zeros((len(data), len(levels) * width))
            for j in range(width):
                z[np.arange(len(data)), codes * width + j] = val[:, j]
            blocks.append(RandomDesign(term, z, val, levels, [names[i] for i in subset], groups, mat.model_spec, subset))
    return blocks


__all__ = ["FixedDesign", "ParsedFormula", "RandomDesign", "RandomTerm", "as_dataframe", "build_fixed_design", "build_random_design", "group_values", "parse_formula"]
