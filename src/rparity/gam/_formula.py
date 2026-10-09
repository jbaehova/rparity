"""GAM smooth specifications and persistent prediction model matrices."""
from __future__ import annotations

import ast
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from rparity.formula import as_dataframe, build_fixed_design, parse_formula

from ._basis import (
    SplineBasis,
    center_basis,
    normal_parameterization,
    scale_basis,
    spline_basis,
    tensor_basis,
)


@dataclass(frozen=True)
class SmoothSpec:
    """Parsed smooth call; formula syntax is data, never arbitrary Python code."""

    kind: str
    variables: tuple[str, ...]
    bs: tuple[str, ...] = ("tp",)
    k: tuple[int, ...] = (10,)
    by: str | None = None
    m: tuple[int, ...] = ()
    fx: bool = False
    normal: bool = True
    mc: tuple[bool, ...] = ()
    id: str | int | None = None

    @property
    def label(self) -> str:
        return f"{self.kind}({','.join(self.variables)})"


@dataclass
class SmoothInfo:
    """Coefficient and penalty indices belonging to a fitted smooth term."""

    label: str
    indices: np.ndarray
    penalty_indices: np.ndarray
    null_space_dim: int
    kind: str
    k: int
    variables: tuple[str, ...]
    by: str | None = None
    level: Any = None
    basis: SplineBasis | None = field(default=None, repr=False)
    spec: SmoothSpec | None = field(default=None, repr=False)


@dataclass
class _SmoothBlock:
    spec: SmoothSpec
    basis: SplineBasis
    label: str
    by_level: Any = None
    by_levels: tuple[Any, ...] = ()
    change: np.ndarray | None = None

    def matrix(self, frame: pd.DataFrame) -> np.ndarray:
        if self.spec.bs == ("re",):
            # Random effects use the same complete dummy interaction at fit and
            # prediction. Unknown levels have zero contribution.
            inputs = np.column_stack([evaluate_expression(v, frame) for v in self.spec.variables])
        else:
            inputs = np.column_stack([evaluate_numeric(v, frame) for v in self.spec.variables])
        result = self.basis.matrix(inputs)
        if self.spec.by:
            values = evaluate_expression(self.spec.by, frame)
            if self.by_levels:
                unknown = ~pd.Series(values).isin(self.by_levels).to_numpy()
                if np.any(unknown):
                    raise ValueError("New factor-by levels are not supported.")
                multiplier = (values == self.by_level).astype(float)
            else:
                multiplier = np.asarray(values, dtype=float)
            result = result * multiplier[:, None]
        return result if self.change is None else result @ self.change


@dataclass
class GamDesign:
    """Full penalized GAM design with immutable training basis information."""

    X: np.ndarray
    y: np.ndarray
    data: pd.DataFrame
    fixed_spec: Any
    coef_names: list[str]
    parametric_indices: np.ndarray
    term_slices: dict[str, list[int]]
    S: list[np.ndarray]
    smooths: list[SmoothInfo]
    offset: np.ndarray
    formula: str
    _blocks: list[_SmoothBlock] = field(repr=False)
    _offset_expressions: list[str] = field(default_factory=list, repr=False)

    @property
    def names(self) -> list[str]:
        return self.coef_names

    def predict(self, newdata: Any) -> np.ndarray:
        """Evaluate the training design at new observations, preserving columns."""
        frame = as_dataframe(newdata).reset_index(drop=True)
        fixed = self.fixed_spec.get_model_matrix(frame)
        if len(fixed) != len(frame):
            raise ValueError("Prediction data must not contain missing covariates.")
        return np.column_stack([np.asarray(fixed, dtype=float)] +
                               [block.matrix(frame) for block in self._blocks])

    def get_model_matrix(self, newdata: Any) -> np.ndarray:
        """Return the complete parametric and smooth matrix for adapters."""
        return self.predict(newdata)

    def prediction_offset(self, newdata: Any) -> np.ndarray:
        frame = as_dataframe(newdata).reset_index(drop=True)
        return sum((evaluate_numeric(v, frame) for v in self._offset_expressions),
                   np.zeros(len(frame)))


def _split_top_level(text: str, separator: str = "+") -> list[str]:
    result = []
    start = 0
    depth = 0
    quote = ""
    for i, char in enumerate(text):
        if quote:
            if char == quote and (i == 0 or text[i - 1] != "\\"):
                quote = ""
        elif char in {"'", '"'}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced GAM formula parentheses.")
        elif char == separator and depth == 0:
            result.append(text[start:i].strip())
            start = i + 1
    if depth or quote:
        raise ValueError("Unbalanced GAM formula parentheses.")
    result.append(text[start:].strip())
    return [s for s in result if s]


def _literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in {"TRUE", "FALSE", "NA", "NULL"}:
        return {"TRUE": True, "FALSE": False, "NA": None, "NULL": None}[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(x) for x in node.elts]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "c":
        return [_literal(x) for x in node.args]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_literal(node.operand)
    raise ValueError("Smooth options must be literal numbers, strings or vectors.")


def _vector(value: Any) -> tuple[Any, ...]:
    return tuple(value) if isinstance(value, (list, tuple)) else (value,)


def parse_gam_formula(formula: str) -> tuple[str, list[SmoothSpec]]:
    """Separate parametric terms from supported top-level s, te and ti calls."""
    if not isinstance(formula, str) or "~" not in formula:
        raise ValueError("A GAM formula must contain '~'.")
    lhs, rhs = formula.split("~", 1)
    fixed, smooths = [], []
    for term in _split_top_level(rhs):
        if not re.match(r"^(s|te|ti)\s*\(", term):
            fixed.append(term)
            continue
        try:
            node = ast.parse(term, mode="eval").body
        except SyntaxError as exc:
            raise ValueError(f"Invalid smooth term {term!r}.") from exc
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise TypeError("Smooth calls must be top-level additive terms.")
        kind = node.func.id
        variables = tuple(ast.unparse(arg) for arg in node.args)
        if not variables:
            raise ValueError("Smooth calls require at least one covariate.")
        if any(kw.arg is None for kw in node.keywords):
            raise ValueError("Unpacking smooth options is not supported.")
        options = {str(kw.arg): kw.value for kw in node.keywords}
        allowed = {"bs", "k", "by", "m", "fx", "np", "mc", "id"}
        if set(options) - allowed:
            raise ValueError(f"Unsupported smooth option: {sorted(set(options) - allowed)}.")
        kwargs = {key: _literal(value) for key, value in options.items() if key != "by"}
        by = ast.unparse(options["by"]) if "by" in options else None
        bs = tuple(str(v) for v in _vector(kwargs.get("bs", "tp" if kind == "s" else "cr")))
        default_k = 10 if kind == "s" else 5
        if kind == "s" and bs == ("tp",) and len(variables) > 1:
            order = (len(variables) + 1) // 2 + 1
            from math import comb
            default_k = comb(order + len(variables) - 1, len(variables)) + (
                27 if len(variables) == 2 else 100)
        k = tuple(int(v) for v in _vector(kwargs.get("k", default_k)))
        m = tuple(int(v) for v in _vector(kwargs["m"])) if kwargs.get("m") is not None else ()
        mc = tuple(bool(v) for v in _vector(kwargs["mc"])) if "mc" in kwargs else ()
        if kind == "s" and len(bs) != 1:
            raise ValueError("An s smooth uses one basis type.")
        if kind != "s" and (len(bs) not in {1, len(variables)} or
                            len(k) not in {1, len(variables)}):
            raise ValueError("Tensor basis options must have one value or one per margin.")
        if kind == "s" and len(variables) > 1 and bs != ("tp",) and bs != ("re",):
            raise ValueError("Multivariate s terms require tp or re bases.")
        smooths.append(SmoothSpec(kind, variables, bs, k, by, m,
                                  bool(kwargs.get("fx", False)), bool(kwargs.get("np", True)),
                                  mc, kwargs.get("id")))
    return f"{lhs.strip()} ~ {' + '.join(fixed) if fixed else '1'}", smooths


def evaluate_expression(expression: str, data: pd.DataFrame) -> np.ndarray:
    """Evaluate a small, explicit arithmetic language over data columns."""
    node = ast.parse(expression, mode="eval").body

    def visit(n: ast.AST) -> Any:
        if isinstance(n, ast.Name):
            if n.id not in data:
                raise ValueError(f"Unknown GAM variable {n.id!r}.")
            return data[n.id].to_numpy()
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float, str)):
            return n.value
        if isinstance(n, ast.UnaryOp):
            value = visit(n.operand)
            if isinstance(n.op, ast.USub):
                return -value
            if isinstance(n.op, ast.UAdd):
                return value
        if isinstance(n, ast.BinOp):
            left, right = visit(n.left), visit(n.right)
            operations = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
                          ast.Div: np.divide, ast.Pow: np.power}
            if type(n.op) in operations:
                return operations[type(n.op)](left, right)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            funcs: dict[str, Callable[..., Any]] = {"log": np.log, "log10": np.log10, "sqrt": np.sqrt, "exp": np.exp,
                     "sin": np.sin, "cos": np.cos, "abs": np.abs, "I": lambda x: x,
                     "factor": lambda x: x, "C": lambda x: x}
            if n.func.id in funcs and len(n.args) == 1 and not n.keywords:
                return funcs[n.func.id](visit(n.args[0]))
        raise ValueError(f"Unsupported GAM covariate expression {expression!r}.")

    value = np.asarray(visit(node))
    return np.repeat(value, len(data)) if value.ndim == 0 else value


def evaluate_numeric(expression: str, data: pd.DataFrame) -> np.ndarray:
    values = np.asarray(evaluate_expression(expression, data), dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Smooth covariates must contain finite numeric values.")
    return values


def _is_factor(expression: str, frame: pd.DataFrame) -> bool:
    if re.match(r"^(factor|C)\(", expression):
        return True
    return expression in frame and (isinstance(frame[expression].dtype, pd.CategoricalDtype)
                                   or not pd.api.types.is_numeric_dtype(frame[expression]))


def _levels(expression: str, frame: pd.DataFrame) -> tuple[Any, ...]:
    if expression in frame and isinstance(frame[expression].dtype, pd.CategoricalDtype):
        return tuple(frame[expression].cat.categories)
    return tuple(sorted(pd.unique(evaluate_expression(expression, frame)), key=str))


def _random_basis(spec: SmoothSpec, frame: pd.DataFrame) -> SplineBasis:
    encodings = []
    sizes = []
    for variable in spec.variables:
        levels = _levels(variable, frame) if _is_factor(variable, frame) else ()
        encodings.append(levels)
        sizes.append(len(levels) if levels else 1)
    width = int(np.prod(sizes))

    def evaluate(z: np.ndarray) -> np.ndarray:
        result = np.ones((len(z), 1))
        for j, levels in enumerate(encodings):
            marginal = (z[:, j, None] == np.asarray(levels)[None, :]).astype(float) if levels else (
                np.asarray(z[:, j], dtype=float)[:, None])
            result = (result[:, :, None] * marginal[:, None, :]).reshape(len(z), -1)
        return result

    return SplineBasis(evaluate, [np.eye(width)], "re", width, 0, len(spec.variables))


def _make_basis(spec: SmoothSpec, frame: pd.DataFrame) -> SplineBasis:
    if spec.bs == ("re",):
        return _random_basis(spec, frame)
    x = np.column_stack([evaluate_numeric(v, frame) for v in spec.variables])
    if spec.kind == "s":
        return scale_basis(spline_basis(x, spec.bs[0], spec.k[0], spec.m), x)
    margins = []
    for j in range(len(spec.variables)):
        kind = spec.bs[j] if len(spec.bs) > 1 else spec.bs[0]
        k = spec.k[j] if len(spec.k) > 1 else spec.k[0]
        margin = spline_basis(x[:, j], kind, k, spec.m)
        if spec.normal and kind not in {"cr", "cs"}:
            margin = normal_parameterization(margin, x[:, j])
        if spec.kind == "ti" and (not spec.mc or spec.mc[j]):
            margin = center_basis(margin, x[:, j])
        margins.append(margin)
    return scale_basis(tensor_basis(margins, spec.kind), x)


def build_design(formula: str, data: Any) -> GamDesign:
    """Build a pure-Python GAM design with absorbed identifiability constraints."""
    fixed_formula, specs = parse_gam_formula(formula)
    frame = as_dataframe(data).reset_index(drop=True)
    expressions = [v for spec in specs for v in spec.variables]
    expressions += [spec.by for spec in specs if spec.by]
    required = set(re.findall(r"\b[A-Za-z_]\w*\b", " ".join(expressions))) & set(frame.columns)
    frame = frame.dropna(subset=sorted(required))
    response = fixed_formula.split("~", 1)[0].strip()
    response_match = re.fullmatch(r"cbind\((.*)\)", response)
    evaluation_formula = fixed_formula
    if response_match:
        arguments = _split_top_level(response_match.group(1), ",")
        frame["__rparity_response"] = evaluate_numeric(arguments[0], frame)
        evaluation_formula = "__rparity_response ~ " + fixed_formula.split("~", 1)[1]
    fixed = build_fixed_design(evaluation_formula, frame)
    frame = fixed.data.reset_index(drop=True)
    xparts = [fixed.X]
    blocks: list[_SmoothBlock] = []
    smooths = []
    local_penalties: list[tuple[list[int], np.ndarray]] = []
    term_slices = dict(fixed.terms)
    names = list(fixed.names)
    for spec in specs:
        if spec.id is not None:
            raise ValueError("Linked smooth id parameters are not implemented.")
        basis = _make_basis(spec, frame)
        by_factor = bool(spec.by and _is_factor(spec.by, frame))
        by_values = evaluate_expression(spec.by, frame) if spec.by else None
        need_center = spec.bs != ("re",) and spec.kind != "ti" and (
            not spec.by or by_factor or np.ptp(np.asarray(by_values, dtype=float)) == 0)
        if need_center:
            values = np.column_stack([evaluate_numeric(v, frame) for v in spec.variables])
            basis = center_basis(basis, values)
        levels = _levels(spec.by, frame) if by_factor and spec.by else ()
        selected_levels = levels
        if (by_factor and spec.by in frame and
                isinstance(frame[spec.by].dtype, pd.CategoricalDtype) and
                frame[spec.by].cat.ordered):
            selected_levels = levels[1:]
        for level in (selected_levels if by_factor else (None,)):
            label = spec.label + (f":{spec.by}{level}" if by_factor else
                                  (f":{spec.by}" if spec.by else ""))
            block = _SmoothBlock(spec, basis, label, level, levels)
            penalties, null_dim = basis.S, basis.null_space_dim
            matrix = block.matrix(frame)
            start = len(names)
            indices = list(range(start, start + matrix.shape[1]))
            names.extend(f"{label}.{i + 1}" for i in range(matrix.shape[1]))
            term_slices[label] = indices
            penalty_indices = []
            if not spec.fx:
                for penalty in penalties:
                    penalty_indices.append(len(local_penalties))
                    local_penalties.append((indices, penalty))
            smooths.append(SmoothInfo(label, np.asarray(indices, dtype=int), np.asarray(penalty_indices, dtype=int),
                                      null_dim, basis.kind, spec.k[0],
                                      spec.variables, spec.by, level, basis, spec))
            blocks.append(block)
            xparts.append(matrix)
    X = np.column_stack(xparts)
    S = []
    for indices, penalty in local_penalties:
        full = np.zeros((X.shape[1], X.shape[1]))
        full[np.ix_(indices, indices)] = penalty
        S.append(full)
    offsets = parse_formula(fixed_formula).offsets
    offset = sum((evaluate_numeric(v, frame) for v in offsets), np.zeros(len(frame)))
    y = fixed.y
    # formulaic's cbind evaluates the public binomial success/failure form.
    response = fixed_formula.split("~", 1)[0].strip()
    match = re.fullmatch(r"cbind\((.*)\)", response)
    if match:
        arguments = _split_top_level(match.group(1), ",")
        if len(arguments) != 2:
            raise ValueError("A binomial cbind response requires success and failure columns.")
        y = np.column_stack([evaluate_numeric(v, frame) for v in arguments])
    return GamDesign(X, y, frame, fixed.spec, names, np.arange(fixed.X.shape[1]),
                     term_slices, S, smooths, offset, formula, blocks, offsets)
