from __future__ import annotations

import ast
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import tokenize
from io import StringIO
from typing import Literal, TypeVar, cast

from IPython.core.inputtransformer2 import TransformerManager

from marvis.notebook_io import read_notebook_bytes
from marvis.validation.field_transformations import (
    TransformationExtraction,
    inspect_safe_transformations,
)
from marvis.validation.input_contracts import (
    FieldCandidate,
    FieldEvidence,
    FieldRecognitionResult,
    JsonValue,
    TransformationSpec,
)


# The observed customer corpus peaks at about 5.9 MiB, 206 cells, and 74 KiB
# per code cell. These limits keep substantial headroom while still bounding all
# untrusted static inspection work.
MAX_NOTEBOOK_BYTES = 64 * 1024 * 1024
MAX_NOTEBOOK_CELLS = 5_000
MAX_NOTEBOOK_CELL_CHARS = 1_000_000
MAX_SAVED_OUTPUT_BYTES = 1_000_000
MAX_SAVED_OUTPUT_ITEMS = 1_000
MAX_EVIDENCE_COUNT = 2_000
MAX_EVIDENCE_EXCERPT_CHARS = 240
MAX_DIAGNOSTIC_CHARS = 256
MAX_DIAGNOSTICS = 500
MAX_LITERAL_NODES = 20_000
MAX_LITERAL_DEPTH = 64
MAX_LITERAL_STRING_CHARS = 1_000_000
MAX_SYMBOLS = 5_000
MAX_BINDINGS_PER_SYMBOL = 64
MAX_ESTIMATOR_PARAM_VARIANTS = 64
MAX_ALIAS_KEYWORD_EVIDENCE = 64
MAX_SECONDARY_EVIDENCE_COUNT = 512
MAX_AST_NODES = 100_000
MAX_EXCERPT_SOURCE_LINES = 8


_RMC_NAME_TO_FIELD = {
    "RMC_TARGET_COL": "target_col",
    "RMC_SPLIT_COL": "split_col",
    "RMC_TIME_COL": "time_col",
    "RMC_PMML_OUTPUT_FIELD": "pmml_output_field",
    "RMC_MODEL_PARAMS": "model_params",
    "RMC_ALGORITHM": "algorithm",
    "RMC_POSITIVE_LABEL": "positive_label",
    "RMC_NEGATIVE_LABEL": "negative_label",
    "RMC_SPLIT_VALUE_MAPPING": "split_value_mapping",
    "RMC_TIME_GRANULARITY": "time_granularity",
}
_ALIAS_NAME_TO_FIELD = {
    "TARGET_COL": "target_col",
    "TARGET": "target_col",
    "LABEL": "target_col",
    "target": "target_col",
    "label": "target_col",
    "SPLIT_COL": "split_col",
    "TIME_COL": "time_col",
    "PMML_OUTPUT_FIELD": "pmml_output_field",
    "MODEL_HYPERPARAMETERS": "model_params",
    "MODEL_PARAMS": "model_params",
    "ALGORITHM": "algorithm",
    "MODEL_ALGORITHM": "algorithm",
    "POSITIVE_LABEL": "positive_label",
    "NEGATIVE_LABEL": "negative_label",
    "SPLIT_VALUE_MAPPING": "split_value_mapping",
    "TIME_GRANULARITY": "time_granularity",
}
_NAME_TO_FIELD = {**_ALIAS_NAME_TO_FIELD, **_RMC_NAME_TO_FIELD}
_ESTIMATOR_NAMES = frozenset(
    {"XGBClassifier", "XGBRegressor", "LGBMClassifier", "LGBMRegressor"}
)
_ANCHORED_LITERAL = re.compile(
    r"^\s*(?P<name>[A-Za-z_]\w*)\s*(?::|=)\s*(?P<value>.+?)\s*$"
)
_REJECTED_OUTPUT_MIME_PREFIXES = (
    "text/html",
    "application/javascript",
    "text/javascript",
    "image/",
)
_BindingT = TypeVar("_BindingT")


@dataclass(frozen=True)
class _PositionedCandidate:
    candidate: FieldCandidate
    confidence: float
    cell_index: int
    line: int
    column: int
    sequence: int


@dataclass(frozen=True)
class _SourceIndex:
    lines: tuple[str, ...]

    @classmethod
    def from_source(cls, source: str) -> _SourceIndex:
        return cls(tuple(source.splitlines(keepends=True)))

    def excerpt(self, node: ast.AST) -> str:
        if not self.lines:
            return type(node).__name__
        start = max(0, int(getattr(node, "lineno", 1)) - 1)
        end = max(start + 1, int(getattr(node, "end_lineno", start + 1)))
        start = min(start, len(self.lines) - 1)
        end = min(end, len(self.lines), start + MAX_EXCERPT_SOURCE_LINES)
        selected = list(self.lines[start:end])
        if not selected:
            return type(node).__name__
        start_column = max(0, int(getattr(node, "col_offset", 0)))
        selected[0] = selected[0][start_column:]
        if end == int(getattr(node, "end_lineno", end)):
            end_column = max(0, int(getattr(node, "end_col_offset", 0)))
            if len(selected) == 1:
                selected[0] = selected[0][: max(0, end_column - start_column)]
            elif end_column:
                selected[-1] = selected[-1][:end_column]
        return _bounded_text("".join(selected), MAX_EVIDENCE_EXCERPT_CHARS)


@dataclass
class _RecognitionState:
    candidates: dict[str, list[_PositionedCandidate]] = field(default_factory=dict)
    transformations: list[TransformationSpec] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    literal_symbols: dict[str, list[JsonValue]] = field(default_factory=dict)
    estimator_symbols: dict[str, list[dict[str, JsonValue]]] = field(
        default_factory=dict
    )
    evidence_count: int = 0
    evidence_limit_reported: bool = False
    saved_output_bytes: int = 0
    saved_output_items: int = 0
    oversized_output_reported: bool = False
    aggregate_output_reported: bool = False
    output_item_limit_reported: bool = False
    unsafe_output_cells: set[int] = field(default_factory=set)
    function_keyword_parameters: dict[str, set[str]] = field(default_factory=dict)
    alias_keyword_seen: set[tuple[str, str]] = field(default_factory=set)
    alias_keyword_count: int = 0
    alias_keyword_limit_reported: bool = False
    secondary_evidence_seen: set[tuple[str, str]] = field(default_factory=set)
    secondary_evidence_count: int = 0
    secondary_evidence_limit_reported: bool = False
    candidate_sequence: int = 0
    transformation_root: str | None = None
    transformation_root_ambiguous: bool = False


def recognize_notebook_fields(notebook_path: str | Path) -> FieldRecognitionResult:
    """Recognize validation candidates from one immutable Notebook snapshot.

    This function parses bytes and AST only. It never executes a cell, imports a
    name mentioned by a cell, evaluates an expression, compiles executable code,
    or invokes ``get_ipython``.
    """
    path = Path(notebook_path)
    raw = path.read_bytes()
    if len(raw) > MAX_NOTEBOOK_BYTES:
        raise ValueError("Notebook exceeds static inspection byte limit")
    try:
        notebook = read_notebook_bytes(
            raw, source="<validation Notebook snapshot>", as_version=4
        )
    except Exception as exc:  # noqa: BLE001 - normalize nbformat parse failures.
        raise ValueError(
            _bounded_text(f"Unable to parse validation Notebook: {exc}")
        ) from exc
    cells = notebook.get("cells", ())
    if not isinstance(cells, (list, tuple)):
        raise ValueError("Notebook cells must be a bounded array")
    if len(cells) > MAX_NOTEBOOK_CELLS:
        raise ValueError("Notebook exceeds static inspection cell limit")

    state = _RecognitionState()
    transformer = TransformerManager()
    for cell_index, cell in enumerate(cells):
        source = _cell_source(cell)
        if source is None:
            state.conflicts.append(f"cell {cell_index} has invalid source text")
            continue
        if len(source) > MAX_NOTEBOOK_CELL_CHARS:
            state.conflicts.append(
                f"cell {cell_index} exceeds static inspection character limit"
            )
            continue
        cell_type = cell.get("cell_type")
        if cell_type == "markdown":
            _collect_anchored_text_candidates(
                state,
                source,
                cell_index=cell_index,
                source_kind="markdown",
                confidence=0.55,
            )
            continue
        if cell_type != "code":
            continue

        neutralized = _neutralize_leading_cell_magic(source)
        tree, transformed = _safe_ast(
            neutralized,
            cell_index=cell_index,
            transformer=transformer,
            state=state,
        )
        if tree is not None:
            source_index = _SourceIndex.from_source(transformed)
            _collect_function_definitions(state, tree)
            _collect_code_candidates(
                state,
                tree,
                source_index=source_index,
                cell_index=cell_index,
            )
            _collect_nested_static_evidence(
                state,
                tree,
                source_index=source_index,
                cell_index=cell_index,
            )
            _merge_transformation_extraction(
                state,
                inspect_safe_transformations(tree, cell_index=cell_index),
                cell_index=cell_index,
            )
        _collect_comment_candidates(state, neutralized, cell_index=cell_index)
        _collect_saved_output_candidates(
            state, cell.get("outputs", ()), cell_index=cell_index
        )

    return FieldRecognitionResult.from_candidates(
        notebook_sha256=sha256(raw).hexdigest(),
        candidates=_ordered_candidates(state),
        transformations=state.transformations,
        conflicts=state.conflicts,
        diagnostics=state.diagnostics,
    )


def _ordered_candidates(
    state: _RecognitionState,
) -> dict[str, tuple[FieldCandidate, ...]]:
    return {
        field_name: tuple(
            item.candidate
            for item in sorted(
                values,
                key=lambda item: (
                    -item.confidence,
                    item.cell_index,
                    item.line,
                    item.column,
                    item.sequence,
                ),
            )
        )
        for field_name, values in state.candidates.items()
    }


def _merge_transformation_extraction(
    state: _RecognitionState,
    extraction: TransformationExtraction,
    *,
    cell_index: int,
) -> None:
    if state.transformation_root_ambiguous:
        return
    if extraction.root_conflict:
        _mark_transformation_root_ambiguous(state, cell_index=cell_index)
        return
    if not extraction.transformations or extraction.dataframe_root is None:
        return
    if state.transformation_root is None:
        state.transformation_root = extraction.dataframe_root
    elif state.transformation_root != extraction.dataframe_root:
        _mark_transformation_root_ambiguous(state, cell_index=cell_index)
        return
    state.transformations.extend(extraction.transformations)


def _mark_transformation_root_ambiguous(
    state: _RecognitionState, *, cell_index: int
) -> None:
    state.transformations.clear()
    state.transformation_root = None
    state.transformation_root_ambiguous = True
    _add_diagnostic(
        state,
        f"cell {cell_index} safe transformations use multiple dataframe roots; "
        "transformation candidates cleared",
    )


def _collect_function_definitions(
    state: _RecognitionState, tree: ast.Module
) -> None:
    for statement in tree.body:
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        state.function_keyword_parameters[statement.name] = (
            _function_keyword_parameter_names(statement.args)
        )


def _safe_ast(
    source: str,
    *,
    cell_index: int,
    transformer: TransformerManager,
    state: _RecognitionState,
) -> tuple[ast.Module | None, str]:
    try:
        transformed = transformer.transform_cell(source)
        tree = ast.parse(transformed, mode="exec")
        node_count = 0
        stack: list[ast.AST] = [tree]
        while stack:
            current = stack.pop()
            node_count += 1
            if node_count > MAX_AST_NODES:
                state.conflicts.append(
                    f"cell {cell_index} exceeds static AST node limit"
                )
                return None, transformed
            stack.extend(ast.iter_child_nodes(current))
        return tree, transformed
    except (SyntaxError, IndentationError, tokenize.TokenError) as exc:
        _add_diagnostic(
            state,
            f"cell {cell_index} static parse skipped: "
            f"{type(exc).__name__}: {exc}",
        )
    except Exception as exc:  # noqa: BLE001 - TransformerManager has plugin errors.
        _add_diagnostic(
            state,
            f"cell {cell_index} input transformation skipped: "
            f"{type(exc).__name__}: {exc}",
        )
    return None, source


def _collect_code_candidates(
    state: _RecognitionState,
    tree: ast.Module,
    *,
    source_index: _SourceIndex,
    cell_index: int,
) -> None:
    for statement in tree.body:
        assignment = _plain_name_assignment(statement)
        if assignment is None:
            continue
        name, value_node = assignment
        field_name = _NAME_TO_FIELD.get(name)

        estimator_class = _estimator_class(value_node)
        if estimator_class is not None:
            excerpt = source_index.excerpt(statement)
            _append_candidate(
                state,
                "algorithm",
                estimator_class,
                source_kind="estimator_constructor",
                cell_index=cell_index,
                excerpt=excerpt,
                confidence=0.8,
                line=statement.lineno,
                column=statement.col_offset,
            )
            params = _resolve_estimator_params(
                state,
                value_node,
                cell_index=cell_index,
                excerpt=excerpt,
            )
            if params is not None:
                state.literal_symbols.pop(name, None)
                _store_estimator_bindings(state, name, params)
                for item in params:
                    _append_candidate(
                        state,
                        "model_params",
                        item,
                        source_kind="estimator_constructor",
                        cell_index=cell_index,
                        excerpt=excerpt,
                        confidence=0.8,
                        line=statement.lineno,
                        column=statement.col_offset,
                    )
            else:
                had_binding = _clear_name_bindings(
                    state.literal_symbols, state.estimator_symbols, name
                )
                if had_binding:
                    _add_unresolved_rebinding_diagnostic(
                        state, name=name, cell_index=cell_index
                    )
            continue

        get_params_name = _known_get_params_name(value_node)
        if get_params_name is not None:
            known_params = [
                dict(item)
                for item in state.estimator_symbols.get(get_params_name, ())
            ]
            if known_params:
                state.estimator_symbols.pop(name, None)
                _store_literal_bindings(state, name, known_params)
            if field_name == "model_params" and known_params:
                excerpt = source_index.excerpt(statement)
                for item in known_params:
                    _append_candidate(
                        state,
                        "model_params",
                        item,
                        source_kind="estimator_get_params",
                        cell_index=cell_index,
                        excerpt=excerpt,
                        confidence=0.95 if name in _RMC_NAME_TO_FIELD else 0.8,
                        line=statement.lineno,
                        column=statement.col_offset,
                    )
            elif not known_params:
                had_binding = _clear_name_bindings(
                    state.literal_symbols, state.estimator_symbols, name
                )
                if field_name == "model_params":
                    _add_diagnostic(
                        state,
                        f"cell {cell_index} model parameter call requires confirmation; "
                        "estimator has no complete static literal binding",
                    )
                elif had_binding:
                    _add_unresolved_rebinding_diagnostic(
                        state, name=name, cell_index=cell_index
                    )
            continue

        values, error = _resolve_literal_bindings(state, value_node)
        if values:
            state.estimator_symbols.pop(name, None)
            _store_literal_bindings(state, name, values)
            if field_name is not None:
                excerpt = source_index.excerpt(statement)
                source_kind = (
                    "rmc_literal" if name in _RMC_NAME_TO_FIELD else "alias_literal"
                )
                confidence = 1.0 if name in _RMC_NAME_TO_FIELD else 0.85
                for value in values:
                    _append_candidate(
                        state,
                        field_name,
                        value,
                        source_kind=source_kind,
                        cell_index=cell_index,
                        excerpt=excerpt,
                        confidence=confidence,
                        line=statement.lineno,
                        column=statement.col_offset,
                    )
        else:
            had_binding = _clear_name_bindings(
                state.literal_symbols, state.estimator_symbols, name
            )
            if field_name is not None:
                detail = error or "dynamic expression"
                _add_diagnostic(
                    state,
                    f"cell {cell_index} {name} requires confirmation: {detail}",
                )
            elif had_binding:
                _add_unresolved_rebinding_diagnostic(
                    state, name=name, cell_index=cell_index
                )


def _resolve_estimator_params(
    state: _RecognitionState,
    node: ast.expr,
    *,
    cell_index: int,
    excerpt: str,
    literal_symbols: dict[str, list[JsonValue]] | None = None,
) -> list[dict[str, JsonValue]] | None:
    if not isinstance(node, ast.Call) or node.args:
        _add_diagnostic(
            state,
            f"cell {cell_index} estimator parameter extraction requires confirmation: "
            "positional or dynamic constructor arguments",
        )
        return None
    variants: list[dict[str, JsonValue]] = [{}]
    for keyword in node.keywords:
        values, error = _resolve_literal_bindings(
            state, keyword.value, literal_symbols=literal_symbols
        )
        if not values:
            _add_diagnostic(
                state,
                f"cell {cell_index} estimator parameter extraction requires confirmation: "
                f"{error or _bounded_text(excerpt)}",
            )
            return None
        if keyword.arg is None:
            if any(not isinstance(value, dict) for value in values):
                _add_diagnostic(
                    state,
                    f"cell {cell_index} estimator **kwargs require a complete "
                    "literal JSON object",
                )
                return None
            next_variants: list[dict[str, JsonValue]] = []
            for existing in variants:
                for value in values:
                    dictionary = cast(dict[str, JsonValue], value)
                    if set(existing) & set(dictionary):
                        _add_diagnostic(
                            state,
                            f"cell {cell_index} estimator parameters require "
                            "confirmation: duplicate keyword",
                        )
                        return None
                    next_variants.append({**existing, **dictionary})
            variants = next_variants
        else:
            next_variants = []
            for existing in variants:
                if keyword.arg in existing:
                    _add_diagnostic(
                        state,
                        f"cell {cell_index} estimator parameters require "
                        "confirmation: duplicate keyword",
                    )
                    return None
                for value in values:
                    next_variants.append({**existing, keyword.arg: value})
            variants = next_variants
        if len(variants) > MAX_ESTIMATOR_PARAM_VARIANTS:
            _add_diagnostic(
                state,
                f"cell {cell_index} estimator parameter ambiguity exceeds "
                "static variant limit and requires confirmation",
            )
            return None
    return variants


def _collect_nested_static_evidence(
    state: _RecognitionState,
    tree: ast.Module,
    *,
    source_index: _SourceIndex,
    cell_index: int,
) -> None:
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _collect_nested_statements(
                state,
                statement.body,
                source_index=source_index,
                cell_index=cell_index,
                literal_symbols={},
                estimator_symbols={},
                scope_kind="function",
                function_enclosing_literals={},
                function_enclosing_estimators={},
            )
        elif isinstance(statement, ast.ClassDef):
            _collect_nested_statements(
                state,
                statement.body,
                source_index=source_index,
                cell_index=cell_index,
                literal_symbols={},
                estimator_symbols={},
                scope_kind="class",
                function_enclosing_literals={},
                function_enclosing_estimators={},
            )
        else:
            for body, shadowed_names in _nested_statement_contexts(statement):
                child_literals = {
                    name: list(values)
                    for name, values in state.literal_symbols.items()
                }
                child_estimators = {
                    name: [dict(value) for value in values]
                    for name, values in state.estimator_symbols.items()
                }
                _remove_shadowed_bindings(
                    shadowed_names, child_literals, child_estimators
                )
                _collect_nested_statements(
                    state,
                    body,
                    source_index=source_index,
                    cell_index=cell_index,
                    literal_symbols=child_literals,
                    estimator_symbols=child_estimators,
                    scope_kind="function",
                    function_enclosing_literals={},
                    function_enclosing_estimators={},
                )

    calls = sorted(
        (node for node in ast.walk(tree) if isinstance(node, ast.Call)),
        key=lambda node: (node.lineno, node.col_offset),
    )
    for call in calls:
        if not isinstance(call.func, ast.Name):
            continue
        accepted_parameters = state.function_keyword_parameters.get(call.func.id)
        if not accepted_parameters:
            continue
        for keyword in call.keywords:
            if (
                keyword.arg not in {"label", "target"}
                or keyword.arg not in accepted_parameters
            ):
                continue
            ok, value, _error = _bounded_literal_value(keyword.value)
            if not ok:
                continue
            _append_candidate(
                state,
                "target_col",
                cast(JsonValue, value),
                source_kind="alias_keyword",
                cell_index=cell_index,
                excerpt=source_index.excerpt(keyword),
                confidence=0.65,
                line=keyword.lineno,
                column=keyword.col_offset,
            )


def _collect_nested_statements(
    state: _RecognitionState,
    statements: list[ast.stmt],
    *,
    source_index: _SourceIndex,
    cell_index: int,
    literal_symbols: dict[str, list[JsonValue]],
    estimator_symbols: dict[str, list[dict[str, JsonValue]]],
    scope_kind: Literal["function", "class"],
    function_enclosing_literals: dict[str, list[JsonValue]],
    function_enclosing_estimators: dict[str, list[dict[str, JsonValue]]],
) -> None:
    for statement in statements:
        assignment = _plain_name_assignment(statement)
        if assignment is not None:
            name, value_node = assignment
            field_name = _NAME_TO_FIELD.get(name)
            estimator_class = _estimator_class(value_node)
            if estimator_class is not None:
                excerpt = source_index.excerpt(statement)
                _append_candidate(
                    state,
                    "algorithm",
                    estimator_class,
                    source_kind="nested_estimator_constructor",
                    cell_index=cell_index,
                    excerpt=excerpt,
                    confidence=0.65,
                    line=statement.lineno,
                    column=statement.col_offset,
                )
                params = _resolve_estimator_params(
                    state,
                    value_node,
                    cell_index=cell_index,
                    excerpt=excerpt,
                    literal_symbols=literal_symbols,
                )
                if params is not None:
                    literal_symbols.pop(name, None)
                    _store_bindings(
                        state,
                        estimator_symbols,
                        name,
                        params,
                        binding_kind="estimator",
                    )
                    for item in params:
                        _append_candidate(
                            state,
                            "model_params",
                            item,
                            source_kind="nested_estimator_constructor",
                            cell_index=cell_index,
                            excerpt=excerpt,
                            confidence=0.65,
                            line=statement.lineno,
                            column=statement.col_offset,
                        )
                else:
                    had_binding = _clear_name_bindings(
                        literal_symbols, estimator_symbols, name
                    )
                    if had_binding:
                        _add_unresolved_rebinding_diagnostic(
                            state, name=name, cell_index=cell_index
                        )
                continue

            get_params_name = _known_get_params_name(value_node)
            if get_params_name is not None:
                known_params = [
                    dict(item) for item in estimator_symbols.get(get_params_name, ())
                ]
                if known_params:
                    estimator_symbols.pop(name, None)
                    _store_bindings(
                        state,
                        literal_symbols,
                        name,
                        known_params,
                        binding_kind="literal",
                    )
                if field_name == "model_params" and known_params:
                    excerpt = source_index.excerpt(statement)
                    for item in known_params:
                        _append_candidate(
                            state,
                            "model_params",
                            item,
                            source_kind="nested_estimator_get_params",
                            cell_index=cell_index,
                            excerpt=excerpt,
                            confidence=0.7,
                            line=statement.lineno,
                            column=statement.col_offset,
                        )
                elif not known_params:
                    had_binding = _clear_name_bindings(
                        literal_symbols, estimator_symbols, name
                    )
                    if field_name == "model_params":
                        _add_diagnostic(
                            state,
                            f"cell {cell_index} nested model parameter call requires "
                            "confirmation; estimator has no complete static literal binding",
                        )
                    elif had_binding:
                        _add_unresolved_rebinding_diagnostic(
                            state, name=name, cell_index=cell_index
                        )
                continue

            values, error = _resolve_literal_bindings(
                state, value_node, literal_symbols=literal_symbols
            )
            if values:
                estimator_symbols.pop(name, None)
                _store_bindings(
                    state,
                    literal_symbols,
                    name,
                    values,
                    binding_kind="literal",
                )
                if field_name is not None:
                    excerpt = source_index.excerpt(statement)
                    rmc = name in _RMC_NAME_TO_FIELD
                    for value in values:
                        _append_candidate(
                            state,
                            field_name,
                            value,
                            source_kind=(
                                "nested_rmc_literal"
                                if rmc
                                else "nested_alias_literal"
                            ),
                            cell_index=cell_index,
                            excerpt=excerpt,
                            confidence=0.95 if rmc else 0.7,
                            line=statement.lineno,
                            column=statement.col_offset,
                        )
            else:
                had_binding = _clear_name_bindings(
                    literal_symbols, estimator_symbols, name
                )
                if field_name is not None:
                    _add_diagnostic(
                        state,
                        f"cell {cell_index} nested {name} requires confirmation: "
                        f"{error or 'dynamic expression'}",
                    )
                elif had_binding:
                    _add_unresolved_rebinding_diagnostic(
                        state, name=name, cell_index=cell_index
                    )

        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            inherited_literals = (
                function_enclosing_literals
                if scope_kind == "class"
                else literal_symbols
            )
            inherited_estimators = (
                function_enclosing_estimators
                if scope_kind == "class"
                else estimator_symbols
            )
            child_literals = {
                name: list(values) for name, values in inherited_literals.items()
            }
            child_estimators = {
                name: [dict(value) for value in values]
                for name, values in inherited_estimators.items()
            }
            for argument in _function_argument_names(statement.args):
                child_literals.pop(argument, None)
                child_estimators.pop(argument, None)
            _remove_shadowed_bindings(
                _bound_names_for_statements(statement.body),
                child_literals,
                child_estimators,
            )
            _collect_nested_statements(
                state,
                statement.body,
                source_index=source_index,
                cell_index=cell_index,
                literal_symbols=child_literals,
                estimator_symbols=child_estimators,
                scope_kind="function",
                function_enclosing_literals=child_literals,
                function_enclosing_estimators=child_estimators,
            )
            continue
        if isinstance(statement, ast.ClassDef):
            inherited_literals = (
                function_enclosing_literals
                if scope_kind == "class"
                else literal_symbols
            )
            inherited_estimators = (
                function_enclosing_estimators
                if scope_kind == "class"
                else estimator_symbols
            )
            child_literals = {
                name: list(values) for name, values in inherited_literals.items()
            }
            child_estimators = {
                name: [dict(value) for value in values]
                for name, values in inherited_estimators.items()
            }
            _collect_nested_statements(
                state,
                statement.body,
                source_index=source_index,
                cell_index=cell_index,
                literal_symbols=child_literals,
                estimator_symbols=child_estimators,
                scope_kind="class",
                function_enclosing_literals={
                    name: list(values) for name, values in inherited_literals.items()
                },
                function_enclosing_estimators={
                    name: [dict(value) for value in values]
                    for name, values in inherited_estimators.items()
                },
            )
            continue
        contexts = _nested_statement_contexts(statement)
        for body, shadowed_names in contexts:
            child_literals = {
                name: list(values) for name, values in literal_symbols.items()
            }
            child_estimators = {
                name: [dict(value) for value in values]
                for name, values in estimator_symbols.items()
            }
            _remove_shadowed_bindings(
                shadowed_names, child_literals, child_estimators
            )
            _collect_nested_statements(
                state,
                body,
                source_index=source_index,
                cell_index=cell_index,
                literal_symbols=child_literals,
                estimator_symbols=child_estimators,
                scope_kind=scope_kind,
                function_enclosing_literals=function_enclosing_literals,
                function_enclosing_estimators=function_enclosing_estimators,
            )
        if contexts:
            _remove_shadowed_bindings(
                _bound_names(statement), literal_symbols, estimator_symbols
            )


def _nested_statement_contexts(
    statement: ast.stmt,
) -> tuple[tuple[list[ast.stmt], set[str]], ...]:
    if isinstance(statement, ast.If):
        return (statement.body, set()), (statement.orelse, set())
    if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
        shadowed = (
            _stored_target_names(statement.target)
            if isinstance(statement, (ast.For, ast.AsyncFor))
            else set()
        )
        return (statement.body, set(shadowed)), (
            statement.orelse,
            set(shadowed),
        )
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        shadowed: set[str] = set()
        for item in statement.items:
            if item.optional_vars is not None:
                shadowed.update(_stored_target_names(item.optional_vars))
        return ((statement.body, shadowed),)
    if isinstance(statement, (ast.Try, ast.TryStar)):
        return (
            (statement.body, set()),
            *(
                (handler.body, {handler.name} if handler.name else set())
                for handler in statement.handlers
            ),
            (statement.orelse, set()),
            (statement.finalbody, set()),
        )
    if isinstance(statement, ast.Match):
        return tuple(
            (case.body, _match_capture_names(case.pattern))
            for case in statement.cases
        )
    return ()


def _stored_target_names(node: ast.AST) -> set[str]:
    return {
        item.id
        for item in ast.walk(node)
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
    }


def _match_capture_names(pattern: ast.pattern) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(pattern):
        if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
    return names


class _BoundNameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802 - ast API name.
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)

    def visit_FunctionDef(  # noqa: N802 - ast API name.
        self, node: ast.FunctionDef
    ) -> None:
        self.names.add(node.name)

    def visit_AsyncFunctionDef(  # noqa: N802 - ast API name.
        self, node: ast.AsyncFunctionDef
    ) -> None:
        self.names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self.names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        del node

    def visit_ExceptHandler(  # noqa: N802 - ast API name.
        self, node: ast.ExceptHandler
    ) -> None:
        if node.name:
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:  # noqa: N802
        if node.name:
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:  # noqa: N802
        if node.name:
            self.names.add(node.name)

    def visit_MatchMapping(  # noqa: N802 - ast API name.
        self, node: ast.MatchMapping
    ) -> None:
        if node.rest:
            self.names.add(node.rest)
        self.generic_visit(node)


def _bound_names(statement: ast.stmt) -> set[str]:
    collector = _BoundNameCollector()
    collector.visit(statement)
    return collector.names


def _bound_names_for_statements(statements: list[ast.stmt]) -> set[str]:
    names: set[str] = set()
    for statement in statements:
        names.update(_bound_names(statement))
    return names


def _remove_shadowed_bindings(
    names: set[str],
    literal_symbols: dict[str, list[JsonValue]],
    estimator_symbols: dict[str, list[dict[str, JsonValue]]],
) -> None:
    for name in names:
        literal_symbols.pop(name, None)
        estimator_symbols.pop(name, None)


def _clear_name_bindings(
    literal_symbols: dict[str, list[JsonValue]],
    estimator_symbols: dict[str, list[dict[str, JsonValue]]],
    name: str,
) -> bool:
    had_binding = name in literal_symbols or name in estimator_symbols
    literal_symbols.pop(name, None)
    estimator_symbols.pop(name, None)
    return had_binding


def _add_unresolved_rebinding_diagnostic(
    state: _RecognitionState, *, name: str, cell_index: int
) -> None:
    _add_diagnostic(
        state,
        f"cell {cell_index} {name} static binding was invalidated by an "
        "unresolved assignment and requires confirmation",
    )


def _function_argument_names(arguments: ast.arguments) -> set[str]:
    names = {
        argument.arg
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        )
    }
    if arguments.vararg is not None:
        names.add(arguments.vararg.arg)
    if arguments.kwarg is not None:
        names.add(arguments.kwarg.arg)
    return names


def _function_keyword_parameter_names(arguments: ast.arguments) -> set[str]:
    return {
        argument.arg for argument in (*arguments.args, *arguments.kwonlyargs)
    }


def _resolve_literal_bindings(
    state: _RecognitionState,
    node: ast.expr,
    *,
    literal_symbols: dict[str, list[JsonValue]] | None = None,
) -> tuple[list[JsonValue], str | None]:
    symbols = state.literal_symbols if literal_symbols is None else literal_symbols
    if isinstance(node, ast.Name):
        values = symbols.get(node.id)
        if values:
            return list(values), None
        return [], f"unknown static symbol {node.id}"
    ok, value, error = _bounded_literal_value(node)
    if ok:
        return [cast(JsonValue, value)], None
    return [], error


def _bounded_literal_value(
    node: ast.AST,
) -> tuple[bool, JsonValue | None, str | None]:
    stack: list[tuple[ast.AST, int]] = [(node, 0)]
    seen = 0
    while stack:
        current, depth = stack.pop()
        seen += 1
        if seen > MAX_LITERAL_NODES:
            return False, None, "literal exceeds static node limit"
        if depth > MAX_LITERAL_DEPTH:
            return False, None, "literal exceeds static depth limit"
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(current))
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as exc:
        return False, None, f"non-literal expression ({type(exc).__name__})"
    try:
        normalized = _normalize_json_value(value)
    except ValueError as exc:
        return False, None, str(exc)
    return True, normalized, None


def _normalize_json_value(
    value: object,
    *,
    depth: int = 0,
    budget: list[int] | None = None,
) -> JsonValue:
    if budget is None:
        budget = [MAX_LITERAL_NODES]
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError("literal exceeds static item limit")
    if depth > MAX_LITERAL_DEPTH:
        raise ValueError("literal exceeds static depth limit")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        if len(value) > MAX_LITERAL_STRING_CHARS:
            raise ValueError("literal string exceeds static character limit")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("literal JSON numbers must be finite")
        return value
    if isinstance(value, list):
        return [
            _normalize_json_value(item, depth=depth + 1, budget=budget)
            for item in value
        ]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("literal JSON object keys must be strings")
        return {
            key: _normalize_json_value(item, depth=depth + 1, budget=budget)
            for key, item in value.items()
        }
    raise ValueError(f"literal is not JSON-compatible: {type(value).__name__}")


def _collect_anchored_text_candidates(
    state: _RecognitionState,
    text: str,
    *,
    cell_index: int,
    source_kind: str,
    confidence: float,
    line_offset: int = 0,
    column: int = 0,
) -> None:
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        match = _ANCHORED_LITERAL.fullmatch(raw_line)
        if match is None:
            continue
        name = match.group("name")
        field_name = _NAME_TO_FIELD.get(name)
        if field_name is None:
            continue
        try:
            expression = ast.parse(match.group("value"), mode="eval").body
        except (SyntaxError, IndentationError) as exc:
            _add_diagnostic(
                state,
                f"cell {cell_index} {source_kind} literal parse skipped: {exc}",
            )
            continue
        ok, value, error = _bounded_literal_value(expression)
        if not ok:
            _add_diagnostic(
                state,
                f"cell {cell_index} {source_kind} candidate skipped: {error}",
            )
            continue
        _append_candidate(
            state,
            field_name,
            cast(JsonValue, value),
            source_kind=source_kind,
            cell_index=cell_index,
            excerpt=raw_line,
            confidence=confidence,
            line=line_offset + line_number,
            column=column,
        )


def _collect_comment_candidates(
    state: _RecognitionState, source: str, *, cell_index: int
) -> None:
    try:
        tokens = tokenize.generate_tokens(StringIO(source).readline)
        for token in tokens:
            if token.type != tokenize.COMMENT:
                continue
            _collect_anchored_text_candidates(
                state,
                token.string[1:],
                cell_index=cell_index,
                source_kind="comment",
                confidence=0.45,
                line_offset=token.start[0] - 1,
                column=token.start[1],
            )
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        _add_diagnostic(
            state,
            f"cell {cell_index} comment tokenization skipped: {type(exc).__name__}: {exc}",
        )


def _collect_saved_output_candidates(
    state: _RecognitionState, outputs: object, *, cell_index: int
) -> None:
    if not isinstance(outputs, (list, tuple)):
        _add_diagnostic(state, f"cell {cell_index} saved outputs are not a list")
        return
    for output in outputs:
        state.saved_output_items += 1
        if state.saved_output_items > MAX_SAVED_OUTPUT_ITEMS:
            if not state.output_item_limit_reported:
                state.output_item_limit_reported = True
                _add_diagnostic(state, "saved output item limit reached")
            return
        if not isinstance(output, dict):
            continue
        output_type = output.get("output_type")
        if output_type == "error":
            continue
        if output_type == "stream":
            text = output.get("text")
            if isinstance(text, list) and all(isinstance(item, str) for item in text):
                text = "".join(text)
            if isinstance(text, str) and _reserve_output_bytes(
                state, text, cell_index=cell_index
            ):
                _collect_anchored_text_candidates(
                    state,
                    text,
                    cell_index=cell_index,
                    source_kind="saved_output",
                    confidence=0.35,
                )
            continue
        data = output.get("data")
        if not isinstance(data, dict):
            continue
        mime_types = set(data)
        if any(
            mime.startswith(_REJECTED_OUTPUT_MIME_PREFIXES)
            for mime in mime_types
        ):
            if cell_index not in state.unsafe_output_cells:
                state.unsafe_output_cells.add(cell_index)
                _add_diagnostic(
                    state, f"cell {cell_index} unsafe saved output MIME skipped"
                )
            continue
        if not mime_types <= {"application/json", "text/plain"}:
            continue
        if "application/json" in data:
            _collect_saved_json_candidates(
                state, data["application/json"], cell_index=cell_index
            )
        if "text/plain" in data:
            text = data["text/plain"]
            if isinstance(text, list) and all(isinstance(item, str) for item in text):
                text = "".join(text)
            if isinstance(text, str) and _reserve_output_bytes(
                state, text, cell_index=cell_index
            ):
                _collect_anchored_text_candidates(
                    state,
                    text,
                    cell_index=cell_index,
                    source_kind="saved_output",
                    confidence=0.35,
                )


def _collect_saved_json_candidates(
    state: _RecognitionState, payload: object, *, cell_index: int
) -> None:
    try:
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, allow_nan=False
        )
        value = _normalize_json_value(payload)
    except (TypeError, ValueError, OverflowError) as exc:
        _add_diagnostic(
            state,
            f"cell {cell_index} saved JSON output skipped: {type(exc).__name__}: {exc}",
        )
        return
    if not _reserve_output_bytes(state, encoded, cell_index=cell_index):
        return
    if not isinstance(value, dict) or any(key not in _NAME_TO_FIELD for key in value):
        _add_diagnostic(
            state,
            f"cell {cell_index} saved JSON output contains non-allowlisted keys",
        )
        return
    for name, item in value.items():
        _append_candidate(
            state,
            _NAME_TO_FIELD[name],
            item,
            source_kind="saved_output",
            cell_index=cell_index,
            excerpt=f"{name}: {_bounded_json(item)}",
            confidence=0.4,
        )


def _reserve_output_bytes(
    state: _RecognitionState, text: str, *, cell_index: int
) -> bool:
    size = len(text.encode("utf-8", errors="replace"))
    if size > MAX_SAVED_OUTPUT_BYTES:
        if not state.oversized_output_reported:
            state.oversized_output_reported = True
            _add_diagnostic(
                state, f"cell {cell_index} saved output exceeds static byte limit"
            )
        return False
    if state.saved_output_bytes + size > MAX_SAVED_OUTPUT_BYTES:
        if not state.aggregate_output_reported:
            state.aggregate_output_reported = True
            _add_diagnostic(state, "saved output aggregate byte limit reached")
        return False
    state.saved_output_bytes += size
    return True


def _append_candidate(
    state: _RecognitionState,
    field_name: str,
    value: JsonValue,
    *,
    source_kind: str,
    cell_index: int,
    excerpt: str,
    confidence: float,
    line: int = 0,
    column: int = 0,
) -> None:
    shape_error = _field_shape_error(field_name, value)
    if shape_error is not None:
        _add_diagnostic(
            state,
            f"cell {cell_index} invalid {field_name} candidate: {shape_error}; "
            f"{source_kind} evidence skipped",
        )
        return
    if source_kind == "alias_keyword":
        identity = (
            field_name,
            json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False),
        )
        if identity in state.alias_keyword_seen:
            return
        if state.alias_keyword_count >= MAX_ALIAS_KEYWORD_EVIDENCE:
            if not state.alias_keyword_limit_reported:
                state.alias_keyword_limit_reported = True
                _add_diagnostic(state, "static alias keyword evidence limit reached")
            return
        state.alias_keyword_seen.add(identity)
        state.alias_keyword_count += 1
    elif "rmc" in source_kind or confidence >= 0.95:
        if state.evidence_count >= MAX_EVIDENCE_COUNT:
            if not state.evidence_limit_reported:
                state.evidence_limit_reported = True
                _add_diagnostic(state, "static recognition evidence limit reached")
            return
        state.evidence_count += 1
    else:
        identity = (
            field_name,
            json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False),
        )
        if identity in state.secondary_evidence_seen:
            return
        if state.secondary_evidence_count >= MAX_SECONDARY_EVIDENCE_COUNT:
            if not state.secondary_evidence_limit_reported:
                state.secondary_evidence_limit_reported = True
                _add_diagnostic(state, "static secondary evidence limit reached")
            return
        state.secondary_evidence_seen.add(identity)
        state.secondary_evidence_count += 1
    evidence = FieldEvidence(
        source_kind=source_kind,
        notebook_cell=cell_index,
        source_excerpt=_bounded_text(excerpt, MAX_EVIDENCE_EXCERPT_CHARS),
        confidence=confidence,
    )
    state.candidate_sequence += 1
    state.candidates.setdefault(field_name, []).append(
        _PositionedCandidate(
            candidate=FieldCandidate(value=value, evidence=(evidence,)),
            confidence=confidence,
            cell_index=cell_index,
            line=max(0, line),
            column=max(0, column),
            sequence=state.candidate_sequence,
        )
    )


_STRING_CANDIDATE_FIELDS = frozenset(
    {
        "target_col",
        "split_col",
        "time_col",
        "pmml_output_field",
        "algorithm",
        "time_granularity",
    }
)


def _field_shape_error(field_name: str, value: JsonValue) -> str | None:
    if field_name in _STRING_CANDIDATE_FIELDS:
        if not isinstance(value, str) or not value.strip():
            return "candidate must be a non-empty string"
        return None
    if field_name in {"positive_label", "negative_label"}:
        if not _is_finite_json_scalar(value):
            return "label candidate must be a finite JSON scalar"
        return None
    if field_name == "split_value_mapping":
        if not isinstance(value, dict) or not value:
            return "split mapping candidate must be a non-empty JSON object"
        if any(not isinstance(key, str) or not key.strip() for key in value):
            return "split mapping keys must be non-empty strings"
        if any(not _is_finite_json_scalar(item) for item in value.values()):
            return "split mapping values must be finite JSON scalars"
        return None
    if field_name == "model_params" and not isinstance(value, dict):
        return "model_params candidate must be a JSON object"
    return None


def _is_finite_json_scalar(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _store_literal_bindings(
    state: _RecognitionState, name: str, values: list[JsonValue]
) -> None:
    _store_bindings(
        state,
        state.literal_symbols,
        name,
        values,
        binding_kind="literal",
    )


def _store_estimator_bindings(
    state: _RecognitionState, name: str, values: list[dict[str, JsonValue]]
) -> None:
    _store_bindings(
        state,
        state.estimator_symbols,
        name,
        values,
        binding_kind="estimator",
    )


def _store_bindings(
    state: _RecognitionState,
    symbols: dict[str, list[_BindingT]],
    name: str,
    values: list[_BindingT],
    *,
    binding_kind: str,
) -> None:
    if name not in symbols and len(symbols) >= MAX_SYMBOLS:
        _add_diagnostic(state, f"static {binding_kind} symbol limit reached")
        return
    bindings = symbols.setdefault(name, [])
    seen = {_binding_identity(value) for value in bindings}
    unique_values: list[_BindingT] = []
    for value in values:
        identity = _binding_identity(value)
        if identity in seen:
            continue
        seen.add(identity)
        unique_values.append(value)
    remaining = MAX_BINDINGS_PER_SYMBOL - len(bindings)
    if remaining <= 0:
        _add_diagnostic(
            state, f"static {binding_kind} binding limit reached for {name}"
        )
        return
    bindings.extend(unique_values[:remaining])
    if len(unique_values) > remaining:
        _add_diagnostic(
            state, f"static {binding_kind} binding limit reached for {name}"
        )


def _binding_identity(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _plain_name_assignment(statement: ast.stmt) -> tuple[str, ast.expr] | None:
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        target = statement.targets[0]
        if isinstance(target, ast.Name):
            return target.id, statement.value
    if isinstance(statement, ast.AnnAssign) and statement.value is not None:
        if isinstance(statement.target, ast.Name):
            return statement.target.id, statement.value
    return None


def _estimator_class(node: ast.expr) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    dotted = _dotted_name(node.func)
    if dotted is None:
        return None
    class_name = dotted.rsplit(".", 1)[-1]
    return class_name if class_name in _ESTIMATOR_NAMES else None


def _known_get_params_name(node: ast.expr) -> str | None:
    if not isinstance(node, ast.Call) or node.args or node.keywords:
        return None
    function = node.func
    if (
        isinstance(function, ast.Attribute)
        and function.attr == "get_params"
        and isinstance(function.value, ast.Name)
    ):
        return function.value.id
    return None


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


def _neutralize_leading_cell_magic(source: str) -> str:
    lines = source.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        if line.lstrip().startswith("%%"):
            newline = "\n" if line.endswith(("\n", "\r")) else ""
            lines[index] = newline
        break
    return "".join(lines)


def _cell_source(cell: object) -> str | None:
    if not isinstance(cell, dict):
        return None
    source = cell.get("source", "")
    if isinstance(source, str):
        return source
    if isinstance(source, list) and all(isinstance(item, str) for item in source):
        return "".join(source)
    return None


def _bounded_json(value: JsonValue) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        text = "<invalid JSON>"
    return _bounded_text(text, MAX_EVIDENCE_EXCERPT_CHARS // 2)


def _add_diagnostic(state: _RecognitionState, message: str) -> None:
    if len(state.diagnostics) >= MAX_DIAGNOSTICS:
        return
    state.diagnostics.append(_bounded_text(message, MAX_DIAGNOSTIC_CHARS))


def _bounded_text(value: object, limit: int = MAX_DIAGNOSTIC_CHARS) -> str:
    text = str(value).replace("\x00", "�")
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."
