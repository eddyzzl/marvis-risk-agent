from __future__ import annotations

import ast
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
import math
import operator
from typing import cast

import pandas as pd

from marvis.validation.input_contracts import JsonScalar, TransformationSpec


_THRESHOLD_OPERATORS: dict[str, Callable[[pd.Series, JsonScalar], pd.Series]] = {
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
    "eq": operator.eq,
    "ne": operator.ne,
}
_THRESHOLD_AST_OPERATORS: dict[type[ast.cmpop], str] = {
    ast.Lt: "lt",
    ast.LtE: "le",
    ast.Gt: "gt",
    ast.GtE: "ge",
    ast.Eq: "eq",
    ast.NotEq: "ne",
}
_DATE_TO_MONTH_MODES = frozenset(
    {"direct_string_slice", "astype_string_slice", "datetime_period"}
)
MAX_TRANSFORMATIONS = 10_000


@dataclass(frozen=True)
class TransformationExtraction:
    dataframe_root: str | None
    transformations: tuple[TransformationSpec, ...]
    root_conflict: bool


def extract_safe_transformations(
    tree: ast.AST, *, cell_index: int
) -> tuple[TransformationSpec, ...]:
    """Extract only exact, declarative dataframe assignment shapes.

    ``cell_index`` is part of the public extraction boundary for callers that keep
    source evidence alongside these candidates. It is deliberately not serialized
    into transformation parameters because runtime parameters are strictly
    allowlisted.
    """
    extraction = inspect_safe_transformations(tree, cell_index=cell_index)
    return () if extraction.root_conflict else extraction.transformations


def inspect_safe_transformations(
    tree: ast.AST, *, cell_index: int
) -> TransformationExtraction:
    del cell_index
    if not isinstance(tree, ast.Module):
        return TransformationExtraction(None, (), False)
    specs: list[TransformationSpec] = []
    dataframe_root: str | None = None
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        rename_specs = _extract_assigned_rename(statement)
        if rename_specs is not None:
            if not rename_specs:
                continue
            target = statement.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if dataframe_root is not None and dataframe_root != target.id:
                return TransformationExtraction(None, (), True)
            dataframe_root = target.id
            specs.extend(rename_specs)
            continue
        target = _column_reference(statement.targets[0])
        if target is None:
            continue
        frame_name, output_field = target
        spec = _extract_column_assignment(
            statement.value,
            frame_name=frame_name,
            output_field=output_field,
        )
        if spec is not None:
            if dataframe_root is not None and dataframe_root != frame_name:
                return TransformationExtraction(None, (), True)
            dataframe_root = frame_name
            specs.append(spec)
    return TransformationExtraction(dataframe_root, tuple(specs), False)


def topologically_sorted_transformations(
    specs: Sequence[TransformationSpec],
) -> tuple[TransformationSpec, ...]:
    if len(specs) > MAX_TRANSFORMATIONS:
        raise ValueError("transformation count limit exceeded")
    by_output: dict[str, TransformationSpec] = {}
    for spec in specs:
        if spec.output_field in by_output:
            raise ValueError(f"duplicate transformation output: {spec.output_field}")
        by_output[spec.output_field] = spec
    dependencies = {
        output: tuple(
            input_field
            for input_field in spec.input_fields
            if input_field in by_output
        )
        for output, spec in by_output.items()
    }
    state: dict[str, int] = {}
    ordered: list[TransformationSpec] = []

    for spec in specs:
        root = spec.output_field
        if state.get(root) == 2:
            continue
        stack: list[tuple[str, int]] = [(root, 0)]
        state[root] = 1
        while stack:
            output, dependency_index = stack[-1]
            output_dependencies = dependencies[output]
            if dependency_index < len(output_dependencies):
                dependency = output_dependencies[dependency_index]
                stack[-1] = (output, dependency_index + 1)
                dependency_state = state.get(dependency, 0)
                if dependency_state == 1:
                    raise ValueError(f"transformation cycle includes: {dependency}")
                if dependency_state == 0:
                    state[dependency] = 1
                    stack.append((dependency, 0))
                continue
            stack.pop()
            state[output] = 2
            ordered.append(by_output[output])
    return tuple(ordered)


def required_transformation_inputs(
    output_fields: Collection[str], specs: Sequence[TransformationSpec]
) -> tuple[str, ...]:
    if isinstance(output_fields, str):
        raise TypeError("output_fields must be a collection of field names")
    ordered = topologically_sorted_transformations(specs)
    by_output = {spec.output_field: spec for spec in ordered}
    inputs: list[str] = []
    seen_inputs: set[str] = set()
    expanded_outputs: set[str] = set()
    for output_field in output_fields:
        stack = [output_field]
        while stack:
            field = stack.pop()
            spec = by_output.get(field)
            if spec is None:
                if field not in seen_inputs:
                    seen_inputs.add(field)
                    inputs.append(field)
                continue
            if field in expanded_outputs:
                continue
            expanded_outputs.add(field)
            stack.extend(reversed(spec.input_fields))
    return tuple(inputs)


def validate_transformation_plan(
    specs: Sequence[TransformationSpec], *, sample_columns: Collection[str]
) -> tuple[TransformationSpec, ...]:
    for spec in specs:
        if not isinstance(spec, TransformationSpec):
            raise ValueError("transformation plan entries must be TransformationSpec")
        if not isinstance(spec.operation, str):
            raise ValueError("transformation operation must be a string")
        if not isinstance(spec.output_field, str) or not spec.output_field:
            raise ValueError("transformation output field must be a non-empty string")
        if not isinstance(spec.input_fields, tuple) or any(
            not isinstance(value, str) or not value for value in spec.input_fields
        ):
            raise ValueError("transformation input fields must be non-empty strings")
        if not isinstance(spec.params, dict) or not all(
            isinstance(key, str) for key in spec.params
        ):
            raise ValueError("transformation params must be a JSON object")
    ordered = topologically_sorted_transformations(specs)
    raw_columns = set(sample_columns)
    for spec in ordered:
        if spec.output_field in raw_columns:
            raise ValueError(
                f"transformation overwrites raw sample field: {spec.output_field}"
            )
        _validate_operation_arity_and_params(spec)

    missing = [
        field
        for field in required_transformation_inputs(
            tuple(spec.output_field for spec in ordered), ordered
        )
        if field not in raw_columns
    ]
    if missing:
        raise ValueError("transformation plan missing raw inputs: " + ", ".join(missing))
    return ordered


def apply_confirmed_transformations(
    frame: pd.DataFrame, specs: Sequence[TransformationSpec]
) -> pd.DataFrame:
    ordered = validate_transformation_plan(specs, sample_columns=frame.columns)
    result = frame.copy()
    for spec in ordered:
        if spec.operation in {"copy", "rename"}:
            result[spec.output_field] = result[spec.input_fields[0]].copy()
        elif spec.operation == "date_to_month":
            source = result[spec.input_fields[0]]
            mode = spec.params["mode"]
            if mode == "direct_string_slice":
                result[spec.output_field] = source.str.slice(0, 7)
            elif mode == "astype_string_slice":
                result[spec.output_field] = source.astype(str).str.slice(0, 7)
            else:
                result[spec.output_field] = (
                    pd.to_datetime(source).dt.to_period("M").astype(str)
                )
        elif spec.operation == "constant_mapping":
            pairs = cast(list[dict[str, JsonScalar]], spec.params["mapping"])
            mapping = {pair["source"]: pair["target"] for pair in pairs}
            mapped: list[JsonScalar] = []
            missing_values: list[object] = []
            for value in result[spec.input_fields[0]].tolist():
                try:
                    mapped.append(mapping[value])
                except (KeyError, TypeError):
                    missing_values.append(value)
            if missing_values:
                raise ValueError(
                    f"transformation {spec.output_field} has unmapped values"
                )
            result[spec.output_field] = pd.Series(
                mapped, index=result.index, dtype="object"
            )
        elif spec.operation == "constant_threshold":
            compare = _THRESHOLD_OPERATORS[str(spec.params["operator"])]
            mask = compare(
                result[spec.input_fields[0]],
                cast(JsonScalar, spec.params["threshold"]),
            )
            true_value = cast(JsonScalar, spec.params["true_value"])
            false_value = cast(JsonScalar, spec.params["false_value"])
            result[spec.output_field] = pd.Series(
                [true_value if bool(value) else false_value for value in mask.tolist()],
                index=result.index,
                dtype="object",
            )
        elif spec.operation == "constant_source_label":
            result[spec.output_field] = spec.params["value"]
        else:  # Defensive: validation already rejects unsupported operations.
            raise ValueError(
                f"unsupported confirmed transformation: {spec.operation}"
            )
    return result


def _extract_column_assignment(
    value: ast.expr, *, frame_name: str, output_field: str
) -> TransformationSpec | None:
    source = _column_reference(value)
    if source is not None and source[0] == frame_name:
        return TransformationSpec("copy", output_field, (source[1],), {})

    date = _extract_date_to_month(value, frame_name=frame_name)
    if date is not None:
        input_field, mode = date
        return TransformationSpec(
            "date_to_month", output_field, (input_field,), {"mode": mode}
        )

    mapping = _extract_constant_mapping(value, frame_name=frame_name)
    if mapping is not None:
        input_field, pairs = mapping
        return TransformationSpec(
            "constant_mapping",
            output_field,
            (input_field,),
            {"mapping": pairs},
        )

    threshold = _extract_constant_threshold(value, frame_name=frame_name)
    if threshold is not None:
        input_field, params = threshold
        return TransformationSpec(
            "constant_threshold", output_field, (input_field,), params
        )

    literal = _literal_json_scalar(value)
    if literal is not _MISSING:
        return TransformationSpec(
            "constant_source_label",
            output_field,
            (),
            {"value": cast(JsonScalar, literal)},
        )
    return None


def _extract_assigned_rename(
    statement: ast.Assign,
) -> tuple[TransformationSpec, ...] | None:
    target = statement.targets[0]
    value = statement.value
    if not isinstance(target, ast.Name) or not isinstance(value, ast.Call):
        return None
    if value.args or len(value.keywords) != 1:
        return None
    function = value.func
    if (
        not isinstance(function, ast.Attribute)
        or function.attr != "rename"
        or not isinstance(function.value, ast.Name)
        or function.value.id != target.id
    ):
        return None
    keyword = value.keywords[0]
    if keyword.arg != "columns" or not isinstance(keyword.value, ast.Dict):
        return None
    pairs: list[tuple[str, str]] = []
    for key_node, value_node in zip(
        keyword.value.keys, keyword.value.values, strict=True
    ):
        if key_node is None:
            return None
        old = _literal_json_scalar(key_node)
        new = _literal_json_scalar(value_node)
        if not isinstance(old, str) or not isinstance(new, str) or not old or not new:
            return None
        pairs.append((old, new))
    outputs = [new for _, new in pairs]
    inputs = [old for old, _ in pairs]
    if (
        len(set(inputs)) != len(inputs)
        or len(set(outputs)) != len(outputs)
        or set(inputs) & set(outputs)
    ):
        return None
    return tuple(
        TransformationSpec("rename", new, (old,), {}) for old, new in pairs
    )


def _extract_date_to_month(
    value: ast.expr, *, frame_name: str
) -> tuple[str, str] | None:
    sliced = _string_slice_base(value)
    if sliced is not None:
        base, direct = sliced
        source = _column_reference(base)
        if source is not None and source[0] == frame_name:
            return source[1], (
                "direct_string_slice" if direct else "astype_string_slice"
            )

    if not isinstance(value, ast.Call) or value.keywords:
        return None
    if len(value.args) != 1 or not _is_name(value.args[0], "str"):
        return None
    astype = value.func
    if not isinstance(astype, ast.Attribute) or astype.attr != "astype":
        return None
    to_period_call = astype.value
    if not isinstance(to_period_call, ast.Call) or to_period_call.keywords:
        return None
    if len(to_period_call.args) != 1:
        return None
    period = _literal_json_scalar(to_period_call.args[0])
    if period != "M":
        return None
    to_period = to_period_call.func
    if not isinstance(to_period, ast.Attribute) or to_period.attr != "to_period":
        return None
    dt = to_period.value
    if not isinstance(dt, ast.Attribute) or dt.attr != "dt":
        return None
    to_datetime_call = dt.value
    if (
        not isinstance(to_datetime_call, ast.Call)
        or to_datetime_call.keywords
        or len(to_datetime_call.args) != 1
        or _dotted_name(to_datetime_call.func)
        not in {"pd.to_datetime", "pandas.to_datetime", "to_datetime"}
    ):
        return None
    source = _column_reference(to_datetime_call.args[0])
    if source is None or source[0] != frame_name:
        return None
    return source[1], "datetime_period"


def _string_slice_base(value: ast.expr) -> tuple[ast.expr, bool] | None:
    if not isinstance(value, ast.Subscript):
        return None
    slice_node = value.slice
    upper = _literal_json_scalar(slice_node.upper) if isinstance(slice_node, ast.Slice) else _MISSING
    if (
        not isinstance(slice_node, ast.Slice)
        or slice_node.lower is not None
        or slice_node.step is not None
        or type(upper) is not int
        or upper != 7
    ):
        return None
    string_accessor = value.value
    if (
        not isinstance(string_accessor, ast.Attribute)
        or string_accessor.attr != "str"
    ):
        return None
    base = string_accessor.value
    if _column_reference(base) is not None:
        return base, True
    if not isinstance(base, ast.Call) or base.keywords or len(base.args) != 1:
        return None
    if not _is_name(base.args[0], "str"):
        return None
    if not isinstance(base.func, ast.Attribute) or base.func.attr != "astype":
        return None
    return base.func.value, False


def _extract_constant_mapping(
    value: ast.expr, *, frame_name: str
) -> tuple[str, list[dict[str, JsonScalar]]] | None:
    if not isinstance(value, ast.Call) or value.keywords or len(value.args) != 1:
        return None
    function = value.func
    if not isinstance(function, ast.Attribute) or function.attr != "map":
        return None
    source = _column_reference(function.value)
    if source is None or source[0] != frame_name:
        return None
    mapping = value.args[0]
    if not isinstance(mapping, ast.Dict):
        return None
    pairs: list[dict[str, JsonScalar]] = []
    sources: dict[JsonScalar, None] = {}
    for key_node, value_node in zip(mapping.keys, mapping.values, strict=True):
        if key_node is None:
            return None
        key = _literal_json_scalar(key_node)
        target = _literal_json_scalar(value_node)
        if key is _MISSING or target is _MISSING:
            return None
        typed_key = cast(JsonScalar, key)
        if typed_key in sources:
            return None
        sources[typed_key] = None
        pairs.append(
            {"source": typed_key, "target": cast(JsonScalar, target)}
        )
    if not pairs:
        return None
    return source[1], pairs


def _extract_constant_threshold(
    value: ast.expr, *, frame_name: str
) -> tuple[str, dict[str, JsonScalar]] | None:
    if isinstance(value, ast.Call):
        function_name = _dotted_name(value.func)
        if function_name in {"np.where", "numpy.where"}:
            if value.keywords or len(value.args) != 3:
                return None
            comparison = _comparison(value.args[0], frame_name=frame_name)
            true_value = _literal_json_scalar(value.args[1])
            false_value = _literal_json_scalar(value.args[2])
            if (
                comparison is None
                or true_value is _MISSING
                or false_value is _MISSING
            ):
                return None
            input_field, operator_name, threshold = comparison
            return input_field, {
                "operator": operator_name,
                "threshold": threshold,
                "true_value": cast(JsonScalar, true_value),
                "false_value": cast(JsonScalar, false_value),
            }

        function = value.func
        if (
            isinstance(function, ast.Attribute)
            and function.attr == "apply"
            and not value.keywords
            and len(value.args) == 1
        ):
            source = _column_reference(function.value)
            lambda_node = value.args[0]
            if source is None or source[0] != frame_name:
                return None
            if not isinstance(lambda_node, ast.Lambda):
                return None
            arguments = lambda_node.args
            if (
                len(arguments.args) != 1
                or arguments.posonlyargs
                or arguments.kwonlyargs
                or arguments.vararg is not None
                or arguments.kwarg is not None
                or arguments.defaults
                or arguments.kw_defaults
            ):
                return None
            body = lambda_node.body
            if not isinstance(body, ast.IfExp):
                return None
            comparison = _lambda_comparison(
                body.test, parameter=arguments.args[0].arg
            )
            true_value = _literal_json_scalar(body.body)
            false_value = _literal_json_scalar(body.orelse)
            if (
                comparison is None
                or true_value is _MISSING
                or false_value is _MISSING
            ):
                return None
            operator_name, threshold = comparison
            return source[1], {
                "operator": operator_name,
                "threshold": threshold,
                "true_value": cast(JsonScalar, true_value),
                "false_value": cast(JsonScalar, false_value),
            }
    return None


def _comparison(
    node: ast.expr, *, frame_name: str
) -> tuple[str, str, JsonScalar] | None:
    if (
        not isinstance(node, ast.Compare)
        or len(node.ops) != 1
        or len(node.comparators) != 1
    ):
        return None
    source = _column_reference(node.left)
    threshold = _literal_json_scalar(node.comparators[0])
    operator_name = _THRESHOLD_AST_OPERATORS.get(type(node.ops[0]))
    if (
        source is None
        or source[0] != frame_name
        or threshold is _MISSING
        or operator_name is None
    ):
        return None
    return source[1], operator_name, cast(JsonScalar, threshold)


def _lambda_comparison(
    node: ast.expr, *, parameter: str
) -> tuple[str, JsonScalar] | None:
    if (
        not isinstance(node, ast.Compare)
        or len(node.ops) != 1
        or len(node.comparators) != 1
        or not _is_name(node.left, parameter)
    ):
        return None
    operator_name = _THRESHOLD_AST_OPERATORS.get(type(node.ops[0]))
    threshold = _literal_json_scalar(node.comparators[0])
    if operator_name is None or threshold is _MISSING:
        return None
    return operator_name, cast(JsonScalar, threshold)


def _column_reference(node: ast.AST) -> tuple[str, str] | None:
    if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
        return None
    column = _literal_json_scalar(node.slice)
    if not isinstance(column, str) or not column:
        return None
    return node.value.id, column


def _validate_operation_arity_and_params(spec: TransformationSpec) -> None:
    _require_json_value(spec.params)
    if spec.operation in {"copy", "rename"}:
        _require_input_count(spec, 1, "exactly one input")
        _require_param_keys(spec, set())
        return
    if spec.operation == "date_to_month":
        _require_input_count(spec, 1, "exactly one input")
        _require_param_keys(spec, {"mode"})
        mode = spec.params["mode"]
        if not isinstance(mode, str):
            raise ValueError("date_to_month mode must be a string")
        if mode not in _DATE_TO_MONTH_MODES:
            raise ValueError("unsupported date_to_month mode")
        return
    if spec.operation == "constant_threshold":
        _require_input_count(spec, 1, "exactly one input")
        _require_param_keys(
            spec, {"operator", "threshold", "true_value", "false_value"}
        )
        operator_name = spec.params["operator"]
        if not isinstance(operator_name, str):
            raise ValueError("constant_threshold operator must be a string")
        if operator_name not in _THRESHOLD_OPERATORS:
            raise ValueError("unsupported constant_threshold operator")
        for key in ("threshold", "true_value", "false_value"):
            _require_json_scalar(spec.params[key], f"constant_threshold {key}")
        return
    if spec.operation == "constant_mapping":
        _require_input_count(spec, 1, "exactly one input")
        _require_param_keys(spec, {"mapping"})
        mapping = spec.params["mapping"]
        if not isinstance(mapping, list) or not mapping:
            raise ValueError("constant_mapping mapping must be a non-empty list")
        sources: dict[JsonScalar, None] = {}
        for pair in mapping:
            if not isinstance(pair, dict) or set(pair) != {"source", "target"}:
                raise ValueError(
                    "constant_mapping entries require source and target keys"
                )
            source = _require_json_scalar(pair["source"], "mapping source")
            _require_json_scalar(pair["target"], "mapping target")
            if source in sources:
                raise ValueError(
                    "constant_mapping has duplicate typed source or Python-equal source"
                )
            sources[source] = None
        return
    if spec.operation == "constant_source_label":
        _require_input_count(spec, 0, "no inputs")
        _require_param_keys(spec, {"value"})
        _require_json_scalar(spec.params["value"], "constant source label")
        return
    raise ValueError(f"unsupported transformation operation: {spec.operation}")


def _require_input_count(
    spec: TransformationSpec, expected: int, description: str
) -> None:
    if len(spec.input_fields) != expected:
        raise ValueError(f"{spec.operation} requires {description}")


def _require_param_keys(spec: TransformationSpec, expected: set[str]) -> None:
    if set(spec.params) != expected:
        raise ValueError(
            f"{spec.operation} parameter keys must be exactly {sorted(expected)}"
        )


def _require_json_value(value: object, *, depth: int = 0) -> None:
    if depth > 64:
        raise ValueError("JSON value exceeds maximum depth")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return
    if isinstance(value, list):
        for item in value:
            _require_json_value(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        for item in value.values():
            _require_json_value(item, depth=depth + 1)
        return
    raise ValueError(f"value is not JSON-compatible: {type(value).__name__}")


def _require_json_scalar(value: object, label: str) -> JsonScalar:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, float):
        raise ValueError(f"{label} must be finite")
    raise ValueError(f"{label} must be a JSON scalar")


_MISSING = object()


def _literal_json_scalar(node: ast.AST | None) -> JsonScalar | object:
    if node is None:
        return _MISSING
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return _MISSING
    try:
        return _require_json_scalar(value, "literal")
    except ValueError:
        return _MISSING


def _is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))
