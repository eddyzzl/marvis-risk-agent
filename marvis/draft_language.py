"""The deliberately small language accepted by executable Draft tools.

Drafts are not general Python plugins.  They are one pure calculation function
with a finite, versioned set of capability imports.  This module is a leaf on
purpose: it only parses and validates source, and never imports application
code or executes Draft code.  The worker compiles the returned import-free AST
into a fresh restricted globals dictionary.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
from typing import Final


DRAFT_EXECUTION_PROFILE: Final = "draft_restricted_v1"
DRAFT_LANGUAGE_VERSION: Final = "Draft Language v1"


class DraftLanguageError(ValueError):
    """Raised when source is outside the supported Draft Language."""


@dataclass(frozen=True)
class DraftCapabilityBinding:
    """A source import binding and the worker-side capability it receives."""

    name: str
    capability: str


@dataclass(frozen=True)
class ValidatedDraft:
    """Validated, import-free source ready for trusted worker compilation."""

    entrypoint: str
    bindings: tuple[DraftCapabilityBinding, ...]
    execution_tree: ast.Module


@dataclass(frozen=True)
class _Member:
    kind: str
    result: str | None = None


# A capability is intentionally narrower than a Python module.  The worker
# creates facades with exactly these members; validation uses the same table so
# a syntactically valid Draft can never ask the worker for an unexposed member.
_MODULE_MEMBERS: Final[dict[str, dict[str, _Member]]] = {
    "collections": {
        "Counter": _Member("call", "value:counter"),
    },
    "decimal": {
        "Decimal": _Member("call", "value:decimal"),
        "ROUND_DOWN": _Member("value"),
        "ROUND_HALF_EVEN": _Member("value"),
        "ROUND_HALF_UP": _Member("value"),
        "ROUND_UP": _Member("value"),
    },
    "fractions": {
        "Fraction": _Member("call", "value:fraction"),
    },
    "functools": {
        "reduce": _Member("call"),
    },
    "itertools": {
        "chain": _Member("call"),
        "repeat": _Member("call"),
    },
    "json": {
        "dumps": _Member("call"),
        "loads": _Member("call"),
    },
    "math": {
        "ceil": _Member("call"),
        "e": _Member("value"),
        "exp": _Member("call"),
        "fabs": _Member("call"),
        "floor": _Member("call"),
        "inf": _Member("value"),
        "isclose": _Member("call"),
        "isfinite": _Member("call"),
        "log": _Member("call"),
        "log10": _Member("call"),
        "nan": _Member("value"),
        "pi": _Member("value"),
        "pow": _Member("call"),
        "sqrt": _Member("call"),
        "trunc": _Member("call"),
    },
    "operator": {
        "add": _Member("call"),
        "floordiv": _Member("call"),
        "mul": _Member("call"),
        "sub": _Member("call"),
        "truediv": _Member("call"),
    },
    "random": {
        "Random": _Member("call", "value:random"),
        "choice": _Member("call"),
        "randint": _Member("call"),
        "random": _Member("call"),
        "uniform": _Member("call"),
    },
    "re": {
        "findall": _Member("call"),
        "fullmatch": _Member("call"),
        "match": _Member("call"),
        "search": _Member("call"),
        "split": _Member("call"),
        "sub": _Member("call"),
    },
    "statistics": {
        "fmean": _Member("call"),
        "mean": _Member("call"),
        "median": _Member("call"),
        "median_high": _Member("call"),
        "median_low": _Member("call"),
        "pstdev": _Member("call"),
        "pvariance": _Member("call"),
        "quantiles": _Member("call"),
        "stdev": _Member("call"),
        "variance": _Member("call"),
    },
    "string": {
        "ascii_letters": _Member("value"),
        "ascii_lowercase": _Member("value"),
        "ascii_uppercase": _Member("value"),
        "digits": _Member("value"),
        "hexdigits": _Member("value"),
        "punctuation": _Member("value"),
        "whitespace": _Member("value"),
    },
}

# The old scanner exported this name and tests/importers use it.  It is now an
# allowlist descriptor, not a blacklist input to an AST walk.
ALLOWED_IMPORT_ROOTS: Final = frozenset({"__future__", *_MODULE_MEMBERS})
ALLOWED_BUILTINS: Final = frozenset({
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "filter",
    "float",
    "frozenset",
    "int",
    "len",
    "list",
    "map",
    "max",
    "min",
    "range",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
})
_SPECIAL_BINDINGS: Final = frozenset({"inputs", "ctx"})
_CONTEXT_MEMBERS: Final = frozenset({"task_id", "seed"})
_INPUT_MEMBERS: Final = {"get": _Member("call")}
_VALUE_MEMBERS: Final = {
    "value:random": {
        "choice": _Member("call"),
        "randint": _Member("call"),
        "random": _Member("call"),
        "uniform": _Member("call"),
    },
}


def validate_draft_source(
    code: str,
    *,
    entrypoint: str | None = None,
) -> ValidatedDraft:
    """Parse ``code`` as Draft Language v1 and return an import-free AST.

    The error text is intentionally user-facing: historical Drafts that no
    longer fit v1 must be rewritten, never silently executed via the old
    blacklist path.
    """

    if not isinstance(code, str):
        raise DraftLanguageError(f"{DRAFT_LANGUAGE_VERSION}: source must be text")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise DraftLanguageError(
            f"{DRAFT_LANGUAGE_VERSION}: source is not valid Python: {exc.msg}"
        ) from exc

    bindings: list[DraftCapabilityBinding] = []
    function: ast.FunctionDef | None = None
    saw_non_import = False
    for index, statement in enumerate(tree.body):
        if (
            index == 0
            and isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            if saw_non_import:
                _reject(statement, "imports must be module-level before the entrypoint")
            bindings.extend(_parse_import(statement))
            continue
        saw_non_import = True
        if not isinstance(statement, ast.FunctionDef):
            _reject(statement, "module may contain only controlled imports and one entrypoint")
        if function is not None:
            _reject(statement, "module must define exactly one entrypoint function")
        function = statement

    if function is None:
        raise DraftLanguageError(
            f"{DRAFT_LANGUAGE_VERSION}: module must define exactly one entrypoint function"
        )
    if entrypoint is not None and function.name != entrypoint:
        raise DraftLanguageError(
            f"{DRAFT_LANGUAGE_VERSION}: expected entrypoint {entrypoint!r}, got {function.name!r}"
        )
    _assert_safe_identifier(function.name, function)
    if len({binding.name for binding in bindings}) != len(bindings):
        raise DraftLanguageError(
            f"{DRAFT_LANGUAGE_VERSION}: an import binding may be declared only once"
        )
    _FunctionValidator(bindings).validate(function)

    execution_function = _execution_function(function)
    execution_tree = ast.Module(body=[execution_function], type_ignores=[])
    ast.fix_missing_locations(execution_tree)
    return ValidatedDraft(
        entrypoint=function.name,
        bindings=tuple(bindings),
        execution_tree=execution_tree,
    )


def _parse_import(statement: ast.Import | ast.ImportFrom) -> list[DraftCapabilityBinding]:
    if isinstance(statement, ast.Import):
        bindings: list[DraftCapabilityBinding] = []
        for alias in statement.names:
            if "." in alias.name or alias.name not in _MODULE_MEMBERS:
                _reject(statement, f"import {alias.name!r} is not an allowed Draft capability")
            binding = alias.asname or alias.name
            _assert_safe_identifier(binding, statement)
            bindings.append(DraftCapabilityBinding(binding, f"module:{alias.name}"))
        return bindings

    if statement.level or not statement.module:
        _reject(statement, "relative imports are not allowed")
    module = statement.module
    if module == "__future__":
        if len(statement.names) != 1 or statement.names[0].name != "annotations":
            _reject(statement, "only 'from __future__ import annotations' is allowed")
        if statement.names[0].asname:
            _reject(statement, "future imports may not be aliased")
        return []
    members = _MODULE_MEMBERS.get(module)
    if members is None:
        _reject(statement, f"from {module!r} import is not an allowed Draft capability")
    bindings = []
    for alias in statement.names:
        if alias.name == "*" or alias.name not in members:
            _reject(statement, f"{module}.{alias.name} is not an allowed Draft capability")
        binding = alias.asname or alias.name
        _assert_safe_identifier(binding, statement)
        bindings.append(DraftCapabilityBinding(binding, f"symbol:{module}.{alias.name}"))
    return bindings


def _execution_function(function: ast.FunctionDef) -> ast.FunctionDef:
    """Return a copy whose annotations cannot execute in restricted globals."""

    copied = copy.deepcopy(function)
    copied.decorator_list = []
    # Python 3.12+ generic type parameters can carry expressions outside the
    # function body. Draft Language does not expose type-level capabilities,
    # so never retain them in the tree trusted by the worker.
    if hasattr(copied, "type_params"):
        copied.type_params = []
    copied.returns = None
    copied.type_comment = None
    for argument in (
        *copied.args.posonlyargs,
        *copied.args.args,
        *copied.args.kwonlyargs,
    ):
        argument.annotation = None
        argument.type_comment = None
    if copied.args.vararg is not None:
        copied.args.vararg.annotation = None
    if copied.args.kwarg is not None:
        copied.args.kwarg.annotation = None
    return copied


class _FunctionValidator(ast.NodeVisitor):
    def __init__(self, bindings: list[DraftCapabilityBinding]):
        self._binding_capabilities = {item.name: item.capability for item in bindings}
        self._local_names: set[str] = set()
        self._local_capabilities: dict[str, str | None] = {}
        self._loop_depth = 0

    def validate(self, function: ast.FunctionDef) -> None:
        if function.decorator_list:
            _reject(function, "decorators are not allowed")
        if getattr(function, "type_params", ()):
            _reject(function, "type parameters are not allowed")
        arguments = function.args
        if (
            arguments.posonlyargs
            or arguments.vararg is not None
            or arguments.kwonlyargs
            or arguments.kwarg is not None
            or arguments.defaults
            or arguments.kw_defaults
            or len(arguments.args) != 2
            or [item.arg for item in arguments.args] != ["inputs", "ctx"]
        ):
            _reject(function, "entrypoint signature must be exactly (inputs, ctx)")
        self._local_names = _collect_local_names(function)
        forbidden = self._local_names & (_SPECIAL_BINDINGS | set(self._binding_capabilities))
        if forbidden:
            _reject(function, f"cannot rebind reserved name {sorted(forbidden)[0]!r}")
        for name in self._local_names:
            _assert_safe_identifier(name, function)
        if not function.body:
            _reject(function, "entrypoint function body may not be empty")
        for index, statement in enumerate(function.body):
            if (
                index == 0
                and isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                continue
            self.visit(statement)

    def generic_visit(self, node: ast.AST) -> None:
        _reject(node, f"{type(node).__name__} is not supported")

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        capability = self._capability_for(node.value)
        for target in node.targets:
            self._assignment_target(target)
            self._record_assignment_capability(target, capability)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
            _reject(node, "assignment operator is not supported")
        self._assignment_target(node.target)
        self.visit(node.value)
        self._record_assignment_capability(node.target, None)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is None:
            _reject(node, "entrypoint must return a JSON object")
        self.visit(node.value)

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        for statement in (*node.body, *node.orelse):
            self.visit(statement)

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self._assignment_target(node.target)
        self._record_assignment_capability(node.target, None)
        self._loop_depth += 1
        try:
            for statement in (*node.body, *node.orelse):
                self.visit(statement)
        finally:
            self._loop_depth -= 1

    def visit_Break(self, node: ast.Break) -> None:
        if not self._loop_depth:
            _reject(node, "break is only allowed inside a for loop")

    def visit_Continue(self, node: ast.Continue) -> None:
        if not self._loop_depth:
            _reject(node, "continue is only allowed inside a for loop")

    def visit_Pass(self, node: ast.Pass) -> None:
        return

    def visit_Assert(self, node: ast.Assert) -> None:
        self.visit(node.test)
        if node.msg is not None:
            self.visit(node.msg)

    def visit_Expr(self, node: ast.Expr) -> None:
        _reject(node, "expression statements are not allowed")

    def visit_Import(self, node: ast.Import) -> None:
        _reject(node, "imports are allowed only at module scope")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        _reject(node, "imports are allowed only at module scope")

    def visit_Match(self, node: ast.Match) -> None:
        _reject(node, "match statements are not allowed")

    def visit_While(self, node: ast.While) -> None:
        _reject(node, "while loops are not allowed")

    def visit_Try(self, node: ast.Try) -> None:
        _reject(node, "try statements are not allowed")

    def visit_With(self, node: ast.With) -> None:
        _reject(node, "with statements are not allowed")

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        _reject(node, "async with statements are not allowed")

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        _reject(node, "async for loops are not allowed")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        _reject(node, "async functions are not allowed")

    def visit_Lambda(self, node: ast.Lambda) -> None:
        _reject(node, "lambda expressions are not allowed")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        _reject(node, "class definitions are not allowed")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        _reject(node, "nested functions are not allowed")

    def visit_Name(self, node: ast.Name) -> None:
        _assert_safe_identifier(node.id, node)
        if isinstance(node.ctx, ast.Store):
            if node.id not in self._local_names:
                _reject(node, f"assignment to {node.id!r} is not allowed")
            return
        if isinstance(node.ctx, ast.Load) and node.id not in (
            self._local_names
            | _SPECIAL_BINDINGS
            | set(self._binding_capabilities)
            | ALLOWED_BUILTINS
        ):
            _reject(node, f"name {node.id!r} is not an allowed Draft capability")

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, (str, int, float, bool, type(None))):
            _reject(node, f"literal type {type(node.value).__name__} is not supported")

    def visit_Dict(self, node: ast.Dict) -> None:
        for key, value in zip(node.keys, node.values, strict=True):
            if key is None:
                _reject(node, "dictionary unpacking is not allowed")
            self.visit(key)
            self.visit(value)

    def visit_List(self, node: ast.List) -> None:
        for item in node.elts:
            self.visit(item)

    def visit_Tuple(self, node: ast.Tuple) -> None:
        for item in node.elts:
            self.visit(item)

    def visit_Set(self, node: ast.Set) -> None:
        for item in node.elts:
            self.visit(item)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
            _reject(node, "binary operator is not supported")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
            _reject(node, "unary operator is not supported")
        self.visit(node.operand)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if not isinstance(node.op, (ast.And, ast.Or)):
            _reject(node, "boolean operator is not supported")
        for item in node.values:
            self.visit(item)

    def visit_Compare(self, node: ast.Compare) -> None:
        if not all(
            isinstance(
                operator,
                (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot),
            )
            for operator in node.ops
        ):
            _reject(node, "comparison operator is not supported")
        self.visit(node.left)
        for item in node.comparators:
            self.visit(item)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.visit(node.test)
        self.visit(node.body)
        self.visit(node.orelse)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        capability = self._capability_for(node.value)
        if capability and (capability.startswith("module:") or capability.startswith("symbol:")):
            _reject(node, "capability objects cannot be indexed")
        self.visit(node.value)
        self.visit(node.slice)

    def visit_Slice(self, node: ast.Slice) -> None:
        for item in (node.lower, node.upper, node.step):
            if item is not None:
                self.visit(item)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._attribute_member(node)
        self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        member = self._call_member(node.func)
        if member.kind != "call":
            _reject(node, "Draft capability is not callable")
        for item in node.args:
            if isinstance(item, ast.Starred):
                _reject(item, "starred call arguments are not allowed")
            self.visit(item)
        for keyword in node.keywords:
            if keyword.arg is None:
                _reject(keyword, "dictionary-unpacked call arguments are not allowed")
            self.visit(keyword.value)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._comprehension(node.generators)
        self.visit(node.elt)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._comprehension(node.generators)
        self.visit(node.elt)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._comprehension(node.generators)
        self.visit(node.elt)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._comprehension(node.generators)
        self.visit(node.key)
        self.visit(node.value)

    def _comprehension(self, generators: list[ast.comprehension]) -> None:
        for generator in generators:
            if generator.is_async:
                _reject(generator, "async comprehensions are not allowed")
            self.visit(generator.iter)
            self._assignment_target(generator.target)
            self._record_assignment_capability(generator.target, None)
            for condition in generator.ifs:
                self.visit(condition)

    def _assignment_target(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            self.visit_Name(target)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self._assignment_target(item)
            return
        _reject(target, "assignment is only allowed to local names")

    def _record_assignment_capability(self, target: ast.AST, capability: str | None) -> None:
        if isinstance(target, ast.Name):
            self._local_capabilities[target.id] = capability
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self._record_assignment_capability(item, None)

    def _attribute_member(self, node: ast.Attribute) -> _Member:
        if node.attr.startswith("_"):
            _reject(node, "private or reflection attributes are not allowed")
        capability = self._capability_for(node.value)
        if capability == "context":
            if node.attr in _CONTEXT_MEMBERS:
                return _Member("value")
        elif capability == "inputs":
            member = _INPUT_MEMBERS.get(node.attr)
            if member is not None:
                return member
        elif capability is not None:
            members = _members_for(capability)
            member = members.get(node.attr) if members is not None else None
            if member is not None:
                return member
        _reject(node, f"attribute {node.attr!r} is not an allowed Draft capability")

    def _call_member(self, function: ast.AST) -> _Member:
        if isinstance(function, ast.Name):
            self.visit_Name(function)
            if function.id in ALLOWED_BUILTINS:
                return _Member("call")
            capability = self._capability_for(function)
            if capability is not None and capability.startswith("symbol:"):
                return _member_for_symbol(capability)
            _reject(function, f"call to {function.id!r} is not an allowed Draft capability")
        if isinstance(function, ast.Attribute):
            member = self._attribute_member(function)
            self.visit(function.value)
            return member
        _reject(function, "dynamic call targets are not allowed")

    def _capability_for(self, expression: ast.AST) -> str | None:
        if isinstance(expression, ast.Name):
            if expression.id == "inputs":
                return "inputs"
            if expression.id == "ctx":
                return "context"
            return self._local_capabilities.get(
                expression.id,
                self._binding_capabilities.get(expression.id),
            )
        if isinstance(expression, ast.Attribute):
            member = self._attribute_member(expression)
            return member.result
        if isinstance(expression, ast.Call):
            return self._call_member(expression.func).result
        return None


def _members_for(capability: str) -> dict[str, _Member] | None:
    if capability.startswith("module:"):
        return _MODULE_MEMBERS.get(capability.removeprefix("module:"))
    return _VALUE_MEMBERS.get(capability)


def _member_for_symbol(capability: str) -> _Member:
    raw = capability.removeprefix("symbol:")
    module, _, name = raw.partition(".")
    member = _MODULE_MEMBERS.get(module, {}).get(name)
    if member is None:
        raise DraftLanguageError(
            f"{DRAFT_LANGUAGE_VERSION}: unknown capability {capability!r}"
        )
    return member


def _collect_local_names(function: ast.FunctionDef) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
    return names


def _assert_safe_identifier(name: str, node: ast.AST) -> None:
    if not name.isidentifier() or name.startswith("_"):
        _reject(node, f"identifier {name!r} is not allowed")


def _reject(node: ast.AST, detail: str) -> None:
    location = getattr(node, "lineno", None)
    suffix = f" at line {location}" if location is not None else ""
    raise DraftLanguageError(f"{DRAFT_LANGUAGE_VERSION}: {detail}{suffix}")


__all__ = [
    "ALLOWED_BUILTINS",
    "ALLOWED_IMPORT_ROOTS",
    "DRAFT_EXECUTION_PROFILE",
    "DRAFT_LANGUAGE_VERSION",
    "DraftCapabilityBinding",
    "DraftLanguageError",
    "ValidatedDraft",
    "validate_draft_source",
]
