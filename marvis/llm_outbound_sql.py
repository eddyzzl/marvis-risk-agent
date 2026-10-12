"""SQL intent compilation receives authenticated schema metadata, never rows."""
from marvis.data.sql_sources import SqlSourceSnapshot
from marvis.llm_outbound import OutboundContextError, _mint_prompt
from marvis.llm_outbound_instructions import user_instruction


def sql_query_prompt(question, source):
    if not isinstance(source, SqlSourceSnapshot):
        raise OutboundContextError("SQL 来源尚未完成身份和 schema 准入，未发送请求。")
    return _mint_prompt({"stage": "sql_select_compile", "question": user_instruction(question),
        "backend": source.backend, "source_id": source.source_id, "schema_hash": source.schema_hash,
        "tables": [{"table": item["table"], "columns": item["columns"]} for item in source.schema()],
        "instruction": "完整保留用户否定、条件、筛选值和口径。只解析查询规格或明确 clarify，不能执行或计算结果。只支持一个已提供的基础表、字段选择或 count/sum/avg/min/max、AND 字面量筛选、group_by、order_by、limit。不支持 JOIN、OR、子查询、CTE、UNION、窗口或其他函数时必须 clarify，不能删掉其约束。未提供连接信息、源行、类别值和文件正文，不得猜测；确认与执行由本地平台控制。"})


def installed_sql_schema(bindings):
    source = bindings.get("source")
    if set(bindings) != {"source"} or not isinstance(source, SqlSourceSnapshot):
        raise OutboundContextError("SQL 输出契约需要已认证来源，未发送请求。")
    fields = sorted({column["name"] for row in source.schema() for column in row["columns"]})
    column = {"type": "string", "enum": fields}
    spec = {"type": "object", "properties": {
        "table": {"type": "string", "enum": [row["table"] for row in source.schema()]},
        "select": {"type": "array", "minItems": 1, "maxItems": 128, "items": {"type": "object", "properties": {
            "column": {"type": "string", "enum": ["*", *fields]}, "aggregate": {"type": "string", "enum": ["count", "sum", "avg", "min", "max"]},
            "alias": {"type": "string", "maxLength": 128}}, "required": ["column"], "additionalProperties": False}},
        "filters": {"type": "array", "maxItems": 32, "items": {"type": "object", "properties": {
            "column": column, "op": {"type": "string", "enum": ["=", "!=", "<>", ">", ">=", "<", "<=", "in", "between", "is_null", "not_null"]},
            "value": {"type": ["string", "number", "integer", "boolean", "null", "array"]}}, "required": ["column", "op"], "additionalProperties": False}},
        "group_by": {"type": "array", "maxItems": 8, "items": column},
        "order_by": {"type": "array", "maxItems": 8, "items": {"type": "object", "properties": {
            "column": {"type": "string", "maxLength": 128}, "direction": {"type": "string", "enum": ["asc", "desc"]}},
            "required": ["column"], "additionalProperties": False}},
        "limit": {"type": "integer", "minimum": 1, "maximum": 100000}}, "required": ["table", "select"], "additionalProperties": False}
    return {"name": "sql_select_compile", "strict": False, "schema": {"type": "object", "properties": {
        "spec": spec, "clarify": {"type": "string", "maxLength": 1000}}, "additionalProperties": False,
        "oneOf": [{"required": ["spec"]}, {"required": ["clarify"]}]}}
