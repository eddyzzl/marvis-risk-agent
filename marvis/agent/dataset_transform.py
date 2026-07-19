"""Deterministic natural-language compiler for governed dataset transforms.

The compiler emits only the closed operation grammar accepted by
``marvis.data.transforms``.  It deliberately does not accept or synthesize
SQL/Python source, and it fails closed when a field or a destructive choice is
ambiguous.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import re


_TRANSFORM_HINTS = (
    "重命名",
    "改名",
    "填充",
    "补全",
    "类型转换",
    "转换为",
    "转为",
    "筛选",
    "过滤",
    "去重",
    "派生字段",
    "新增字段",
    "rename",
    "fill missing",
    "impute",
    "cast ",
    "filter rows",
    "deduplicate",
    "derive ",
    "create column",
)
_DELETE_HINT = re.compile(r"(?:删除|删掉|移除|去掉|\bdrop\b|\bremove\b)", re.I)
_NON_DATA_DELETE_HINT = re.compile(r"(?:规则|策略|节点|任务|报告|文件|plan|rule|strategy)", re.I)
_DATA_DELETE_HINT = re.compile(
    r"(?:列|字段|变量|标签|手机号|金额|收入|\bcolumns?\b|\bfields?\b|[a-z_][a-z0-9_]*)",
    re.I,
)
_SOURCE_CODE_REQUEST = re.compile(
    r"(?:用|使用|执行|运行|write|run|execute|using)\s*(?:一段\s*)?"
    r"(?:sql|python)(?:\s*(?:代码|脚本|code|script))?",
    re.I,
)
_ACTION_START = re.compile(
    r"^(?:(?:然后|再|并且|并|and\s+then)\s*)?"
    r"(?:把|将|用|使用|rename\b|drop\b|remove\b|fill\b|impute\b|"
    r"cast\b|筛选|过滤|filter\b|新增|创建|派生|derive\b|create\b|add\b|"
    r"删除|删掉|移除|去掉|按.+?去重|deduplicate\b)",
    re.I,
)
_PROTECTED_ROLES = frozenset({"target", "id", "phone", "idcard"})
_EXPLICIT_DROP_CONFIRMATION = re.compile(
    r"(?:我\s*)?(?:明确\s*)?确认\s*(?:要|执行)?\s*(?:删除|删掉|移除|去掉)"
    r"|\b(?:i\s+)?confirm(?:ed)?\b.{0,30}\b(?:drop|remove)\b",
    re.I,
)
_DESTINATION_PATTERN = r"(?:`[^`]+`|'[^']+'|\"[^\"]+\"|[\w\u4e00-\u9fff.\-]+)"
_SAFE_INTEGER_MAX = 2**53 - 1
_TYPE_ALIASES = {
    "整数": "INTEGER",
    "整型": "INTEGER",
    "长整数": "BIGINT",
    "浮点数": "DOUBLE",
    "浮点型": "DOUBLE",
    "数值": "DOUBLE",
    "字符串": "VARCHAR",
    "文本": "VARCHAR",
    "日期": "DATE",
    "时间": "TIMESTAMP",
    "时间戳": "TIMESTAMP",
    "布尔": "BOOLEAN",
    "布尔值": "BOOLEAN",
}
_SAFE_TYPE = re.compile(
    r"^(?:BOOLEAN|TINYINT|SMALLINT|INTEGER|BIGINT|HUGEINT|"
    r"UTINYINT|USMALLINT|UINTEGER|UBIGINT|REAL|FLOAT|DOUBLE|VARCHAR|DATE|"
    r"TIMESTAMP|TIMESTAMP WITH TIME ZONE|DECIMAL\([1-9]\d*,\s*\d+\))$",
    re.I,
)
_STATISTIC_METHODS = {
    "均值": "mean",
    "平均值": "mean",
    "mean": "mean",
    "average": "mean",
    "中位数": "median",
    "median": "median",
    "最小值": "min",
    "minimum": "min",
    "min": "min",
    "最大值": "max",
    "maximum": "max",
    "max": "max",
}


@dataclass(frozen=True)
class DatasetTransformRequest:
    operations: tuple[dict[str, object], ...]
    confirm_protected_drop: bool = False


@dataclass(frozen=True)
class DatasetTransformRequestResult:
    request: DatasetTransformRequest | None = None
    clarification: str | None = None
    operations: tuple[dict[str, object], ...] = ()
    protected_fields: tuple[str, ...] = ()


class _NeedClarification(ValueError):
    pass


class _ColumnResolver:
    def __init__(
        self,
        columns: Sequence[str],
        business_names: Mapping[str, str],
    ) -> None:
        ordered = tuple(dict.fromkeys(str(column) for column in columns))
        if not ordered or any(not column.strip() for column in ordered):
            raise _NeedClarification("当前数据集没有可加工的有效字段。")
        self.columns = ordered
        aliases: dict[str, list[str]] = {}
        for column in ordered:
            aliases.setdefault(column.casefold(), []).append(column)
            business_name = str(business_names.get(column) or "").strip()
            if business_name:
                aliases.setdefault(business_name.casefold(), []).append(column)
        self._aliases = aliases
        alternatives = sorted(aliases, key=lambda value: (-len(value), value))
        self.pattern = (
            r"(?<![A-Za-z0-9_])(?:"
            + "|".join(re.escape(value) for value in alternatives)
            + r")(?![A-Za-z0-9_])"
        )

    def resolve(self, value: str) -> str:
        cleaned = _clean_field_token(value)
        matches = self._aliases.get(cleaned.casefold(), [])
        if not matches:
            raise _NeedClarification(
                f"当前数据集中没有字段「{cleaned or value.strip()}」，请使用现有字段名或业务名称。"
            )
        unique = tuple(dict.fromkeys(matches))
        if len(unique) != 1:
            raise _NeedClarification(
                f"字段名称「{cleaned}」对应多个字段，请改用原始字段名。"
            )
        return unique[0]

    def resolve_list(self, value: str) -> list[str]:
        tokens = _split_field_list(value)
        if not tokens:
            raise _NeedClarification("请明确提供要处理的字段。")
        resolved = [self.resolve(token) for token in tokens]
        if len(resolved) != len(set(resolved)):
            raise _NeedClarification("同一个字段被重复指定，请确认后重试。")
        return resolved


def detect_dataset_transform_intent(utterance: str | None) -> bool:
    """Return whether text explicitly requests a governed dataset change."""

    text = str(utterance or "").strip().casefold()
    if not text:
        return False
    if any(hint in text for hint in _TRANSFORM_HINTS):
        return True
    if re.search(r"(?:新增|创建|计算)\s*(?:字段|列|变量)?\s*[\w\u4e00-\u9fff]+\s*=", text):
        return True
    if _DELETE_HINT.search(text):
        return not _NON_DATA_DELETE_HINT.search(text) and bool(
            _DATA_DELETE_HINT.search(text)
        )
    return False


def build_dataset_transform_request(
    utterance: str,
    *,
    columns: Sequence[str],
    business_names: Mapping[str, str],
    semantic_mapping: object,
) -> DatasetTransformRequestResult:
    """Compile a natural-language request into the closed transform grammar."""

    text = str(utterance or "").strip()
    if not text:
        return _clarify("请说明要如何加工当前数据集。")
    try:
        semantic_names, roles, target_col = _semantic_parts(semantic_mapping)
        merged_names = dict(semantic_names)
        merged_names.update({str(key): str(value) for key, value in business_names.items()})
        virtual_columns = tuple(dict.fromkeys(str(column) for column in columns))
        virtual_names = {
            column: name
            for column, name in merged_names.items()
            if column in virtual_columns
        }
        virtual_roles = {
            column: role
            for column, role in roles.items()
            if column in virtual_columns
        }
        virtual_target = target_col if target_col in virtual_columns else None
        resolver = _ColumnResolver(virtual_columns, virtual_names)
        if _SOURCE_CODE_REQUEST.search(text):
            raise _NeedClarification(
                "数据加工不接受 SQL 或 Python 代码；请直接描述重命名、删列、填充、转换、筛选、派生或去重需求。"
            )

        operations: list[dict[str, object]] = []
        confirmed_protected: list[str] = []
        pending_protected: list[str] = []
        for clause in _split_clauses(text):
            operation = _parse_clause(clause, resolver)
            if operation["op"] == "drop_columns":
                protected_in_action = [
                    column
                    for column in operation["columns"]  # type: ignore[union-attr]
                    if column == virtual_target
                    or virtual_roles.get(column) in _PROTECTED_ROLES
                ]
                if protected_in_action:
                    destination = (
                        confirmed_protected
                        if _EXPLICIT_DROP_CONFIRMATION.search(clause)
                        else pending_protected
                    )
                    destination.extend(protected_in_action)
            operations = _append_or_merge(operations, operation)
            (
                virtual_columns,
                virtual_names,
                virtual_roles,
                virtual_target,
            ) = _advance_virtual_schema(
                virtual_columns,
                virtual_names,
                virtual_roles,
                virtual_target,
                operation,
            )
            if virtual_columns:
                resolver = _ColumnResolver(virtual_columns, virtual_names)
        if not operations:
            raise _NeedClarification(
                "请明确重命名、删列、缺失填充、类型转换、条件筛选、算术派生或去重参数。"
            )

        pending = tuple(dict.fromkeys(pending_protected))
        canonical_operations = tuple(operations)
        if pending:
            labels = "、".join(pending)
            return DatasetTransformRequestResult(
                request=None,
                clarification=(
                    f"字段 {labels} 是 target 或关键标识字段。"
                    f"若仍要删除，请明确回复“确认删除 {labels}”。"
                ),
                operations=canonical_operations,
                protected_fields=pending,
            )
        confirmed = bool(confirmed_protected)
        return DatasetTransformRequestResult(
            request=DatasetTransformRequest(
                operations=canonical_operations,
                confirm_protected_drop=confirmed,
            ),
            operations=canonical_operations,
        )
    except _NeedClarification as exc:
        return _clarify(str(exc))


def _parse_clause(clause: str, resolver: _ColumnResolver) -> dict[str, object]:
    text = clause.strip(" ，,;；。")
    lowered = text.casefold()
    if re.search(r"(?:去重|\bdeduplicat(?:e|ion)\b|remove\s+duplicates)", lowered):
        return _parse_deduplicate(text, resolver)
    if re.search(r"(?:重命名|改名|\brename\b)", lowered):
        return _parse_rename(text, resolver)
    if re.search(r"(?:填充|补全|\bfill\b|\bimpute\b)", lowered):
        return _parse_fill(text, resolver)
    if re.search(r"(?:严格转|尝试转|类型转换|转换为|转为|\bcast\b)", lowered):
        return _parse_cast(text, resolver)
    if re.search(r"(?:筛选|过滤|\bfilter\b|keep\s+rows\s+where)", lowered):
        return _parse_filter(text, resolver)
    if re.search(r"(?:新增|创建|派生|\bderive\b|create\s+column|add\s+column)", lowered):
        return _parse_derive(text, resolver)
    if _DELETE_HINT.search(lowered):
        return _parse_drop(text, resolver)
    raise _NeedClarification(f"无法确定这一步数据加工的参数：「{text}」。")


def _parse_rename(text: str, resolver: _ColumnResolver) -> dict[str, object]:
    alias = resolver.pattern
    patterns = (
        re.compile(
            rf"(?:把|将)?\s*(?P<src>{alias})\s*(?:字段|列)?\s*"
            rf"(?:重命名为|改名为|rename\s+(?:to|as))\s*(?P<dst>{_DESTINATION_PATTERN})",
            re.I,
        ),
        re.compile(
            rf"\brename\s+(?:column\s+)?(?P<src>{alias})\s+"
            rf"(?:to|as)\s+(?P<dst>{_DESTINATION_PATTERN})",
            re.I,
        ),
    )
    matches = _unique_matches(text, patterns)
    marker_count = len(re.findall(r"(?:重命名(?:为)?|改名(?:为)?|\brename\b)", text, re.I))
    if not matches or len(matches) != marker_count:
        raise _NeedClarification("重命名需要明确现有字段和新的字段名。")
    mapping: dict[str, str] = {}
    for match in matches:
        source = resolver.resolve(match.group("src"))
        destination = _unquote(match.group("dst")).strip()
        if not destination or "\x00" in destination:
            raise _NeedClarification("新的字段名不能为空。")
        if source in mapping:
            raise _NeedClarification(f"字段「{source}」被重复重命名，请只保留一个目标名称。")
        if destination in resolver.columns and destination != source:
            raise _NeedClarification(f"新字段名「{destination}」已存在，请换一个名称。")
        mapping[source] = destination
    return {"op": "rename_columns", "mapping": mapping}


def _parse_drop(text: str, resolver: _ColumnResolver) -> dict[str, object]:
    chinese = re.search(
        r"(?:确认\s*)?(?:删除|删掉|移除|去掉)\s*(?:字段|列|变量)?\s*(?P<items>.+)$",
        text,
        re.I,
    )
    english = re.search(
        r"(?:\bconfirm(?:ed)?\s+)?\b(?:drop|remove)\s+"
        r"(?:columns?|fields?)?\s*(?P<items>.+)$",
        text,
        re.I,
    )
    match = chinese or english
    if match is None:
        raise _NeedClarification("删列需要明确要删除的字段。")
    columns = resolver.resolve_list(match.group("items"))
    return {"op": "drop_columns", "columns": columns}


def _parse_fill(text: str, resolver: _ColumnResolver) -> dict[str, object]:
    alias = resolver.pattern
    patterns = (
        re.compile(
            rf"(?:并)?(?:使用|用)\s*(?P<method>.+?)\s*(?:填充|补全)\s*"
            rf"(?P<column>{alias})(?:字段)?(?:的)?(?:缺失值|空值)?$",
            re.I,
        ),
        re.compile(
            rf"(?:填充|补全)\s*(?P<column>{alias})(?:字段)?(?:的)?(?:缺失值|空值)?"
            rf"\s*(?:为|用|使用)\s*(?P<method>.+)$",
            re.I,
        ),
        re.compile(
            rf"\b(?:fill|impute)\s+(?:missing(?:\s+values?)?(?:\s+in)?\s+)?"
            rf"(?P<column>{alias})(?:\s+missing(?:\s+values?)?)?\s+"
            rf"(?:with|using)\s+(?P<method>.+)$",
            re.I,
        ),
    )
    match = next((pattern.search(text) for pattern in patterns if pattern.search(text)), None)
    if match is None:
        unknown = _likely_unknown_field(text, resolver)
        if unknown:
            resolver.resolve(unknown)
        raise _NeedClarification("缺失填充需要明确字段，以及常量或 mean/median/min/max 方法。")
    column = resolver.resolve(match.group("column"))
    raw_method = match.group("method").strip(" ，,。")
    normalized_method = raw_method.casefold().strip()
    statistic = _STATISTIC_METHODS.get(normalized_method)
    if statistic is not None:
        fill: dict[str, object] = {"column": column, "method": statistic}
    else:
        constant_text = re.sub(
            r"^(?:常量|constant)\s*[:：]?\s*", "", raw_method, flags=re.I
        )
        value = _parse_literal(constant_text)
        if value is None:
            raise _NeedClarification("缺失值不能再用 null 填充，请提供非空常量。")
        fill = {"column": column, "method": "constant", "value": value}
    return {"op": "fill_missing", "fills": [fill]}


def _parse_cast(text: str, resolver: _ColumnResolver) -> dict[str, object]:
    alias = resolver.pattern
    patterns = (
        re.compile(
            rf"(?:将|把)?\s*(?P<column>{alias})\s*"
            rf"(?P<mode>尝试|容错|安全|严格|try|strict)?\s*"
            rf"(?:类型)?(?:转换|转|cast)\s*(?:为|成|to)\s*(?P<type>.+)$",
            re.I,
        ),
        re.compile(
            rf"(?P<mode>try|strict)?\s*\bcast\s+(?P<column>{alias})\s+"
            rf"to\s+(?P<type>.+?)(?:\s+(?:using|in)\s+"
            rf"(?P<mode_after>try|strict)\s*(?:mode)?)?$",
            re.I,
        ),
    )
    match = next((pattern.search(text) for pattern in patterns if pattern.search(text)), None)
    if match is None:
        unknown = _likely_unknown_field(text, resolver)
        if unknown:
            resolver.resolve(unknown)
        raise _NeedClarification("类型转换需要明确字段、目标类型以及 strict 或 try 模式。")
    column = resolver.resolve(match.group("column"))
    groups = match.groupdict()
    raw_mode = groups.get("mode") or groups.get("mode_after")
    if raw_mode is None:
        raise _NeedClarification(
            f"字段「{column}」的类型转换需要选择 strict（失败即停止）或 try（失败转空值）模式。"
        )
    mode = "try" if raw_mode.casefold() in {"try", "尝试", "容错", "安全"} else "strict"
    target_type = _normalize_type(match.group("type"))
    return {
        "op": "cast_columns",
        "casts": [{"column": column, "to_type": target_type, "mode": mode}],
    }


def _parse_filter(text: str, resolver: _ColumnResolver) -> dict[str, object]:
    body = re.sub(
        r"^.*?(?:筛选|过滤|filter\s+rows(?:\s+where)?|keep\s+rows\s+where)\s*",
        "",
        text,
        count=1,
        flags=re.I,
    ).strip()
    if not body:
        raise _NeedClarification("条件筛选需要提供字段、比较符和比较值。")
    parts, connectors = _split_conditions(body)
    predicates = [_parse_condition(part, resolver) for part in parts]
    if len(predicates) == 1:
        predicate = predicates[0]
    else:
        logical = {connector for connector in connectors}
        if len(logical) != 1:
            raise _NeedClarification("同时使用 AND 和 OR 时请分步说明，避免条件优先级歧义。")
        predicate = {"op": logical.pop(), "args": predicates}
    return {"op": "filter_rows", "predicate": predicate}


def _parse_condition(text: str, resolver: _ColumnResolver) -> dict[str, object]:
    clause = text.strip(" ，,。()")
    match = re.match(rf"(?P<column>{resolver.pattern})\s*(?P<rest>.+)$", clause, re.I)
    if match is None:
        unknown = re.match(r"[`'\"]?([A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fff]+)", clause)
        if unknown:
            resolver.resolve(unknown.group(1))
        raise _NeedClarification(f"无法识别筛选条件「{clause}」，请明确字段、比较符和值。")
    column = resolver.resolve(match.group("column"))
    rest = match.group("rest").strip()
    null_operators = (
        (r"^(?:不为空|非空|is\s+not\s+null)$", "is_not_null"),
        (r"^(?:为空|是空值|is\s+null)$", "is_null"),
    )
    for pattern, operation in null_operators:
        if re.fullmatch(pattern, rest, re.I):
            return {"op": operation, "arg": {"column": column}}
    operators = (
        (r"^(?:大于等于|不小于|至少|>=|≥)\s*(.+)$", "gte"),
        (r"^(?:小于等于|不大于|至多|<=|≤)\s*(.+)$", "lte"),
        (r"^(?:不等于|!=|<>)\s*(.+)$", "ne"),
        (r"^(?:大于|超过|>)\s*(.+)$", "gt"),
        (r"^(?:小于|低于|<)\s*(.+)$", "lt"),
        (r"^(?:等于|为|==|=)\s*(.+)$", "eq"),
    )
    for pattern, operation in operators:
        compared = re.fullmatch(pattern, rest, re.I)
        if compared:
            value = _parse_literal(compared.group(1))
            if value is None and operation in {"eq", "ne"}:
                null_operation = "is_null" if operation == "eq" else "is_not_null"
                return {"op": null_operation, "arg": {"column": column}}
            return {
                "op": operation,
                "left": {"column": column},
                "right": {"literal": value},
            }
    raise _NeedClarification(f"筛选字段「{column}」缺少受支持的比较符或比较值。")


def _parse_derive(text: str, resolver: _ColumnResolver) -> dict[str, object]:
    patterns = (
        re.compile(
            rf"(?:新增|创建|派生)\s*(?:字段|列|变量)?\s*"
            rf"(?P<name>{_DESTINATION_PATTERN})\s*(?:=|为)\s*(?P<expression>.+)$",
            re.I,
        ),
        re.compile(
            rf"\b(?:derive|create|add)\s+(?:column\s+)?"
            rf"(?P<name>{_DESTINATION_PATTERN})\s*(?:=|as)\s*(?P<expression>.+)$",
            re.I,
        ),
    )
    match = next((pattern.search(text) for pattern in patterns if pattern.search(text)), None)
    if match is None:
        raise _NeedClarification("派生字段需要提供新字段名和一个受限的二元算术表达式。")
    name = _unquote(match.group("name")).strip()
    if not name or name in resolver.columns:
        raise _NeedClarification(f"派生字段名「{name}」为空或已存在，请换一个名称。")
    expression = _parse_arithmetic(match.group("expression"), resolver)
    return {
        "op": "derive_columns",
        "derivations": [{"name": name, "expression": expression}],
    }


def _parse_arithmetic(text: str, resolver: _ColumnResolver) -> dict[str, object]:
    match = re.fullmatch(
        r"\s*(?P<left>.+?)\s*(?P<operator>除以|乘以|取模|加上|减去|[+\-*/%])"
        r"\s*(?P<right>.+?)\s*",
        text,
        re.I,
    )
    if match is None:
        raise _NeedClarification("派生表达式仅支持两个字段或数值之间的 +、-、*、/、% 运算。")
    operation = {
        "+": "add",
        "加上": "add",
        "-": "subtract",
        "减去": "subtract",
        "*": "multiply",
        "乘以": "multiply",
        "/": "divide",
        "除以": "divide",
        "%": "modulo",
        "取模": "modulo",
    }[match.group("operator").casefold()]
    return {
        "op": operation,
        "left": _parse_arithmetic_operand(match.group("left"), resolver),
        "right": _parse_arithmetic_operand(match.group("right"), resolver),
    }


def _parse_arithmetic_operand(text: str, resolver: _ColumnResolver) -> dict[str, object]:
    token = text.strip()
    try:
        return {"column": resolver.resolve(token)}
    except _NeedClarification:
        value = _parse_literal(token)
        if isinstance(value, bool) or value is None or not isinstance(value, (int, float)):
            raise _NeedClarification(
                f"算术表达式中的「{token}」不是现有字段或有限数值。"
            ) from None
        return {"literal": value}


def _parse_deduplicate(text: str, resolver: _ColumnResolver) -> dict[str, object]:
    chinese = re.search(r"按\s*(?P<keys>.+?)\s*去重", text, re.I)
    english = re.search(
        r"\bdeduplicate(?:\s+rows)?\s+by\s+(?P<keys>.+?)"
        r"(?=\s+order\s+by|\s+keep(?:ing)?\b|$)",
        text,
        re.I,
    )
    key_match = chinese or english
    if key_match is None:
        raise _NeedClarification("去重需要明确一个或多个 key 字段。")
    keys = resolver.resolve_list(key_match.group("keys"))
    order_by: list[dict[str, str]] = []
    nulls = "first" if re.search(r"(?:空值最前|nulls\s+first)", text, re.I) else "last"

    for match in re.finditer(
        rf"(?:保留|取)\s*(?P<column>{resolver.pattern})\s*"
        rf"(?P<direction>最新|最晚|最大|最早|最小)\s*(?:一条|记录)?",
        text,
        re.I,
    ):
        direction = "desc" if match.group("direction") in {"最新", "最晚", "最大"} else "asc"
        order_by.append(
            {
                "column": resolver.resolve(match.group("column")),
                "direction": direction,
                "nulls": nulls,
            }
        )
    for match in re.finditer(
        rf"\border\s+by\s+(?P<column>{resolver.pattern})\s*"
        rf"(?P<direction>asc|desc)?(?:\s+nulls\s+(?P<nulls>first|last))?",
        text,
        re.I,
    ):
        raw_direction = match.group("direction")
        if raw_direction is None:
            raise _NeedClarification("去重的排序字段需要明确 asc 或 desc。")
        order_by.append(
            {
                "column": resolver.resolve(match.group("column")),
                "direction": raw_direction.casefold(),
                "nulls": (match.group("nulls") or nulls).casefold(),
            }
        )
    if not order_by and chinese is not None:
        tail = text[chinese.end() :]
        for match in re.finditer(
            rf"按\s*(?P<column>{resolver.pattern})\s*(?P<direction>升序|降序|asc|desc)",
            tail,
            re.I,
        ):
            direction = "desc" if match.group("direction").casefold() in {"降序", "desc"} else "asc"
            order_by.append(
                {
                    "column": resolver.resolve(match.group("column")),
                    "direction": direction,
                    "nulls": nulls,
                }
            )
    if not order_by:
        raise _NeedClarification("去重还需要明确排序字段和升序/降序，以确定保留哪一条记录。")
    ordered_columns = [item["column"] for item in order_by]
    if len(ordered_columns) != len(set(ordered_columns)):
        raise _NeedClarification("去重排序字段不能重复。")
    return {"op": "deduplicate", "keys": keys, "order_by": order_by}


def _split_clauses(text: str) -> tuple[str, ...]:
    coarse = re.split(r"[;；。\n]+", text)
    result: list[str] = []
    for item in coarse:
        for remaining in _split_sequenced_actions(item.strip()):
            if not remaining:
                continue
            start = 0
            for match in re.finditer(r"[，,]", remaining):
                tail = remaining[match.end() :].lstrip()
                if _ACTION_START.match(tail):
                    result.append(remaining[start : match.start()].strip())
                    start = match.end()
            result.append(remaining[start:].strip())
    return tuple(item for item in result if item)


def _split_sequenced_actions(text: str) -> tuple[str, ...]:
    if not text:
        return ()
    result: list[str] = []
    start = 0
    sequence = re.compile(r"(?:然后|随后|接着|and\s+then|并(?=\s*(?:删除|删掉|移除|去掉)))", re.I)
    for match in sequence.finditer(text):
        tail = text[match.end() :].lstrip()
        if not _ACTION_START.match(tail):
            continue
        result.append(text[start : match.start()].strip())
        start = match.end()
    result.append(text[start:].strip())
    return tuple(item for item in result if item)


def _split_conditions(text: str) -> tuple[list[str], list[str]]:
    connector_pattern = re.compile(
        r"\s*(并且|且|以及|和|或者|或|\b(?:and|or)\b)\s*", re.I
    )
    parts: list[str] = []
    connectors: list[str] = []
    start = 0
    quote: str | None = None
    index = 0
    while index < len(text):
        char = text[index]
        if char in {"'", '"', "`"}:
            quote = None if quote == char else char if quote is None else quote
            index += 1
            continue
        if quote is None:
            match = connector_pattern.match(text, index)
            if match:
                part = text[start:index].strip()
                if not part:
                    raise _NeedClarification("筛选条件连接符前后都需要完整条件。")
                parts.append(part)
                connectors.append(
                    "or" if match.group(1).casefold() in {"或者", "或", "or"} else "and"
                )
                index = match.end()
                start = index
                continue
        index += 1
    final = text[start:].strip()
    if not final:
        raise _NeedClarification("筛选条件连接符前后都需要完整条件。")
    parts.append(final)
    return parts, connectors


def _append_or_merge(
    operations: list[dict[str, object]],
    operation: dict[str, object],
) -> list[dict[str, object]]:
    if not operations or operations[-1]["op"] != operation["op"]:
        operations.append(operation)
        return operations
    current = operations[-1]
    op = str(operation["op"])
    if op == "rename_columns":
        mapping = dict(current["mapping"])  # type: ignore[arg-type]
        incoming = dict(operation["mapping"])  # type: ignore[arg-type]
        if set(mapping.values()) & set(incoming):
            # The incoming rename consumes a name produced by the preceding
            # rename, so it must remain a separate ordered kernel operation.
            operations.append(operation)
            return operations
        duplicate = set(mapping) & set(incoming)
        if duplicate:
            raise _NeedClarification(
                f"字段「{sorted(duplicate)[0]}」被重复重命名，请只保留一个目标名称。"
            )
        mapping.update(incoming)
        current["mapping"] = mapping
        return operations
    member = {
        "drop_columns": "columns",
        "cast_columns": "casts",
        "fill_missing": "fills",
        "derive_columns": "derivations",
    }.get(op)
    if member is not None:
        if op in {"cast_columns", "fill_missing"}:
            item_key = "casts" if op == "cast_columns" else "fills"
            prior_columns = {
                str(item["column"])
                for item in current[item_key]  # type: ignore[union-attr]
            }
            incoming_columns = {
                str(item["column"])
                for item in operation[item_key]  # type: ignore[union-attr]
            }
            if prior_columns & incoming_columns:
                # Repeating a cast/fill on one column is an ordered request,
                # not two simultaneous members of the same kernel step.
                operations.append(operation)
                return operations
        if op == "derive_columns":
            prior_names = {
                str(item["name"])
                for item in current["derivations"]  # type: ignore[union-attr]
            }
            incoming_references = _referenced_columns(
                operation["derivations"]  # type: ignore[arg-type]
            )
            if prior_names & incoming_references:
                operations.append(operation)
                return operations
        current[member] = [
            *list(current[member]),  # type: ignore[arg-type]
            *list(operation[member]),  # type: ignore[arg-type]
        ]
        return operations
    operations.append(operation)
    return operations


def _advance_virtual_schema(
    columns: Sequence[str],
    business_names: Mapping[str, str],
    roles: Mapping[str, str],
    target_col: str | None,
    operation: Mapping[str, object],
) -> tuple[tuple[str, ...], dict[str, str], dict[str, str], str | None]:
    """Project one parsed operation onto the schema seen by later clauses."""

    next_columns = list(columns)
    next_names = dict(business_names)
    next_roles = dict(roles)
    next_target = target_col
    op = str(operation.get("op") or "")
    if op == "rename_columns":
        mapping = dict(operation["mapping"])  # type: ignore[arg-type]
        next_columns = [str(mapping.get(column, column)) for column in next_columns]
        for source, destination in mapping.items():
            source_name = str(source)
            destination_name = str(destination)
            if source_name in next_names:
                next_names[destination_name] = next_names.pop(source_name)
            if source_name in next_roles:
                next_roles[destination_name] = next_roles.pop(source_name)
            if next_target == source_name:
                next_target = destination_name
    elif op == "drop_columns":
        dropped = {str(column) for column in operation["columns"]}  # type: ignore[union-attr]
        next_columns = [column for column in next_columns if column not in dropped]
        for column in dropped:
            next_names.pop(column, None)
            next_roles.pop(column, None)
        if next_target in dropped:
            next_target = None
    elif op == "derive_columns":
        for derivation in operation["derivations"]:  # type: ignore[union-attr]
            name = str(derivation["name"])
            next_columns.append(name)
    if not next_columns:
        raise _NeedClarification("删列不能移除当前数据集的全部字段。")
    if len(next_columns) != len(set(next_columns)):
        raise _NeedClarification("数据加工会产生重复字段名，请调整重命名或派生字段。")
    return tuple(next_columns), next_names, next_roles, next_target


def _referenced_columns(value: object) -> set[str]:
    if isinstance(value, Mapping):
        referenced = (
            {str(value["column"])} if "column" in value else set()
        )
        for item in value.values():
            referenced.update(_referenced_columns(item))
        return referenced
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        referenced: set[str] = set()
        for item in value:
            referenced.update(_referenced_columns(item))
        return referenced
    return set()


def _semantic_parts(
    semantic_mapping: object,
) -> tuple[Mapping[str, str], Mapping[str, str], str | None]:
    if isinstance(semantic_mapping, Mapping):
        names = semantic_mapping.get("business_names", {})
        roles = semantic_mapping.get("field_roles", {})
        target = semantic_mapping.get("target_col")
    else:
        names = getattr(semantic_mapping, "business_names", {})
        roles = getattr(semantic_mapping, "field_roles", {})
        target = getattr(semantic_mapping, "target_col", None)
    if not isinstance(names, Mapping) or not isinstance(roles, Mapping):
        raise _NeedClarification("当前数据语义映射无效，请先修正字段配置。")
    return (
        {str(key): str(value) for key, value in names.items()},
        {str(key): str(value) for key, value in roles.items()},
        str(target) if target is not None else None,
    )


def _normalize_type(value: str) -> str:
    stripped = value.strip(" ，,。").strip()
    alias = _TYPE_ALIASES.get(stripped.casefold())
    normalized = alias or " ".join(stripped.upper().split())
    if not _SAFE_TYPE.fullmatch(normalized):
        raise _NeedClarification(f"目标类型「{stripped}」不在安全类型白名单中。")
    decimal = re.fullmatch(r"DECIMAL\(([1-9]\d*),\s*(\d+)\)", normalized, re.I)
    if decimal:
        precision = int(decimal.group(1))
        scale = int(decimal.group(2))
        if precision > 38 or scale > precision:
            raise _NeedClarification("DECIMAL 精度必须不超过 38，且 scale 不能大于 precision。")
        return f"DECIMAL({precision},{scale})"
    return normalized


def _parse_literal(value: str) -> object:
    text = value.strip(" ，,。").strip()
    if not text:
        raise _NeedClarification("比较值或填充值不能为空。")
    if (
        len(text) >= 2
        and text[0] == text[-1]
        and text[0] in {"'", '"', "`"}
    ):
        return text[1:-1]
    lowered = text.casefold()
    if lowered in {"null", "none", "空值"}:
        return None
    if lowered in {"true", "是", "真"}:
        return True
    if lowered in {"false", "否", "假"}:
        return False
    if re.fullmatch(r"[+-]?\d+", text):
        integer = int(text)
        if abs(integer) > _SAFE_INTEGER_MAX:
            raise _NeedClarification("整数超出精确 JSON 范围，请缩小范围或改用受控 DECIMAL 转换。")
        return integer
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?", text):
        number = float(text)
        if not math.isfinite(number):
            raise _NeedClarification("数值必须是有限值。")
        return number
    if re.search(r"(?:;|--|/\*|\*/|\bselect\b|\bimport\b|__)", text, re.I):
        raise _NeedClarification("值中包含不受支持的代码或表达式，请提供普通常量。")
    return text


def _clean_field_token(value: str) -> str:
    text = _unquote(str(value).strip())
    text = re.sub(r"^(?:字段|列|变量)\s*", "", text, flags=re.I)
    text = re.sub(r"^(?:column|field)\s+", "", text, flags=re.I)
    text = re.sub(r"\s*(?:字段|列|变量)$", "", text, flags=re.I)
    text = re.sub(r"\s+(?:column|field)$", "", text, flags=re.I)
    return text.strip()


def _split_field_list(value: str) -> list[str]:
    cleaned = value.strip(" ，,。").strip()
    cleaned = re.sub(r"\s*(?:字段|列|变量)\s*$", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+(?:columns?|fields?)\s*$", "", cleaned, flags=re.I)
    return [
        item.strip()
        for item in re.split(r"\s*(?:、|,|，|和|及|与|\band\b)\s*", cleaned, flags=re.I)
        if item.strip()
    ]


def _likely_unknown_field(text: str, resolver: _ColumnResolver) -> str | None:
    known = {column.casefold() for column in resolver.columns}
    reserved = {
        "fill",
        "missing",
        "with",
        "using",
        "impute",
        "cast",
        "to",
        "try",
        "strict",
        "mean",
        "median",
        "min",
        "max",
        "double",
        "integer",
        "varchar",
        "date",
        "timestamp",
    }
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        lowered = token.casefold()
        if lowered not in known and lowered not in reserved:
            return token
    return None


def _unique_matches(text: str, patterns: Sequence[re.Pattern[str]]) -> list[re.Match[str]]:
    matches = [match for pattern in patterns for match in pattern.finditer(text)]
    unique: dict[tuple[int, int], re.Match[str]] = {}
    for match in matches:
        unique.setdefault(match.span(), match)
    return [unique[key] for key in sorted(unique)]


def _unquote(value: str) -> str:
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"', "`"}:
        return text[1:-1]
    return text


def _clarify(message: str) -> DatasetTransformRequestResult:
    return DatasetTransformRequestResult(request=None, clarification=message)


__all__ = [
    "DatasetTransformRequest",
    "DatasetTransformRequestResult",
    "build_dataset_transform_request",
    "detect_dataset_transform_intent",
]
