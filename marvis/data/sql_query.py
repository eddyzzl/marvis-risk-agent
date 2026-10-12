"""Disclosed bounded SELECT grammar, shared by manual and Agent proposals.

One base table, named projections, five aggregates, AND filters, grouping,
ordering and LIMIT are supported. No SQL from a caller is executed verbatim.
"""
import math
import re


class SqlQueryError(ValueError):
    def __init__(self, code="sql_query_unsupported"):
        self.code = code
        super().__init__("当前只支持单表 SELECT、字段选择、COUNT/SUM/AVG/MIN/MAX、AND 筛选、分组、排序和 LIMIT；请简化查询，尚未执行。")


OPERATORS = {"=", "!=", "<>", ">", ">=", "<", "<=", "in", "between", "is_null", "not_null"}
AGGREGATES = {"count", "sum", "avg", "min", "max"}
_TOKEN = re.compile(r"\s*(?:(?P<string>'(?:[^']|'')*')|(?P<quoted>\"(?:[^\"]|\"\")*\")|(?P<number>-?\d+(?:\.\d+)?)|(?P<name>[^\W\d]\w*)|(?P<symbol><=|>=|!=|<>|[=<>(),.*;]))", re.UNICODE)


def _identifier(value):
    if not isinstance(value, str) or not value or len(value) > 128 or any(char in value for char in "\x00\n\r"):
        raise SqlQueryError("sql_identifier_invalid")
    return '"' + value.replace('"', '""') + '"'


def _literal(value):
    if value is None:
        return "NULL"
    if type(value) is bool:
        return "TRUE" if value else "FALSE"
    if type(value) in (int, float) and math.isfinite(value) and abs(value) <= 10**16:
        return str(value)
    if isinstance(value, str) and len(value) <= 512 and "\x00" not in value:
        return "'" + value.replace("'", "''") + "'"
    raise SqlQueryError("sql_literal_invalid")


def compile_select(spec, tables, *, max_rows=100000):
    if not isinstance(spec, dict) or set(spec) - {"table", "select", "filters", "group_by", "order_by", "limit"}:
        raise SqlQueryError()
    matched = [table for table in tables if table["table"] == spec.get("table")]
    if len(matched) != 1:
        raise SqlQueryError("sql_table_not_allowed")
    table = matched[0]
    columns = {item["name"] for item in table["columns"]}
    def field(value):
        if value not in columns:
            raise SqlQueryError("sql_column_not_allowed")
        return _identifier(value)

    selections = spec.get("select")
    if not isinstance(selections, list) or not 1 <= len(selections) <= 128:
        raise SqlQueryError()
    projection, names, bare, aggregated = [], set(), set(), False
    for row in selections:
        if not isinstance(row, dict) or set(row) - {"column", "aggregate", "alias"}:
            raise SqlQueryError()
        column, aggregate, alias = row.get("column"), row.get("aggregate"), row.get("alias")
        if aggregate is None:
            expression = "*" if column == "*" else field(column)
            if column == "*" and len(selections) != 1:
                raise SqlQueryError()
            bare.add(column)
        elif aggregate in AGGREGATES:
            expression = aggregate.upper() + "(" + ("*" if aggregate == "count" and column == "*" else field(column)) + ")"
            aggregated = True
            if alias is None:
                alias = aggregate + ("_all" if column == "*" else "_" + column)
        else:
            raise SqlQueryError("sql_function_not_allowed")
        name = alias or column
        if name in names:
            raise SqlQueryError("sql_duplicate_projection")
        names.add(name)
        projection.append(expression + (" AS " + _identifier(alias) if alias else ""))
    groups = spec.get("group_by", [])
    if not isinstance(groups, list) or len(groups) > 8 or len(set(groups)) != len(groups):
        raise SqlQueryError()
    group_sql = [field(column) for column in groups]
    if (aggregated or groups) and ("*" in bare or not bare <= set(groups)):
        raise SqlQueryError("sql_grouping_required")
    filters = spec.get("filters", [])
    if not isinstance(filters, list) or len(filters) > 32:
        raise SqlQueryError()
    conditions = []
    for row in filters:
        if not isinstance(row, dict) or set(row) - {"column", "op", "value"} or row.get("op") not in OPERATORS:
            raise SqlQueryError()
        column, op, value = field(row.get("column")), row["op"], row.get("value")
        if op in {"is_null", "not_null"}:
            expression = column + (" IS NULL" if op == "is_null" else " IS NOT NULL")
        elif op == "in":
            if not isinstance(value, list) or not 1 <= len(value) <= 64:
                raise SqlQueryError()
            expression = column + " IN (" + ",".join(_literal(item) for item in value) + ")"
        elif op == "between":
            if not isinstance(value, list) or len(value) != 2:
                raise SqlQueryError()
            expression = column + " BETWEEN " + _literal(value[0]) + " AND " + _literal(value[1])
        else:
            if value is None:
                raise SqlQueryError("sql_null_comparison_unsupported")
            expression = column + " " + op + " " + _literal(value)
        conditions.append(expression)
    orders = spec.get("order_by", [])
    if not isinstance(orders, list) or len(orders) > 8:
        raise SqlQueryError()
    order_sql = []
    for row in orders:
        if not isinstance(row, dict) or set(row) - {"column", "direction"} or row.get("direction", "asc") not in {"asc", "desc"}:
            raise SqlQueryError()
        column = row.get("column")
        if column not in names and column not in columns:
            raise SqlQueryError("sql_column_not_allowed")
        if (aggregated or groups) and column not in names and column not in groups:
            raise SqlQueryError("sql_grouping_required")
        order_sql.append(_identifier(column) + " " + row.get("direction", "asc").upper())
    limit = spec.get("limit")
    if limit is not None and (type(limit) is not int or not 1 <= limit <= max_rows):
        raise SqlQueryError("sql_row_limit_invalid")
    relation = _identifier(table["relation"])
    if table.get("namespace"):
        relation = _identifier(table["namespace"]) + "." + relation
    query = "SELECT " + ", ".join(projection) + " FROM " + relation
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    if group_sql:
        query += " GROUP BY " + ", ".join(group_sql)
    if order_sql:
        query += " ORDER BY " + ", ".join(order_sql)
    if limit is not None:
        query += " LIMIT " + str(limit)
    return query


class _Parser:
    def __init__(self, query):
        if not isinstance(query, str) or not query.strip() or len(query) > 16384:
            raise SqlQueryError()
        self.tokens = []
        offset = 0
        while offset < len(query.rstrip()):
            match = _TOKEN.match(query, offset)
            if not match:
                raise SqlQueryError()
            self.tokens.append((match.lastgroup, match.group(match.lastgroup)))
            offset = match.end()
        self.position = 0
        if len(self.tokens) > 2048:
            raise SqlQueryError()

    def peek(self, token):
        return self.position < len(self.tokens) and self.tokens[self.position][1].casefold() == token.casefold()

    def take(self, token=None):
        if self.position >= len(self.tokens) or (token is not None and not self.peek(token)):
            raise SqlQueryError()
        result = self.tokens[self.position]
        self.position += 1
        return result

    def name(self):
        kind, value = self.take()
        if kind == "quoted":
            return value[1:-1].replace('""', '"')
        if kind != "name":
            raise SqlQueryError()
        return value

    def literal(self):
        kind, value = self.take()
        if kind == "string":
            return value[1:-1].replace("''", "'")
        if kind == "number":
            return float(value) if "." in value else int(value)
        if kind == "name" and value.casefold() in {"null", "true", "false"}:
            return {"null": None, "true": True, "false": False}[value.casefold()]
        raise SqlQueryError()

    def names(self):
        result = [self.name()]
        while self.peek(","):
            self.take(",")
            result.append(self.name())
        return result


def parse_manual_select(query, tables, *, max_rows=100000):
    parser = _Parser(query)
    parser.take("SELECT")
    selections = []
    while True:
        column = parser.take("*")[1] if parser.peek("*") else parser.name()
        row = {"column": column}
        if parser.peek("("):
            if column.casefold() not in AGGREGATES:
                raise SqlQueryError("sql_function_not_allowed")
            row["aggregate"] = column.casefold()
            parser.take("(")
            row["column"] = parser.take("*")[1] if parser.peek("*") else parser.name()
            parser.take(")")
        if parser.peek("AS"):
            parser.take("AS")
            row["alias"] = parser.name()
        selections.append(row)
        if not parser.peek(","):
            break
        parser.take(",")
    parser.take("FROM")
    table = parser.name()
    if parser.peek("."):
        parser.take(".")
        table += "." + parser.name()
    spec = {"table": table, "select": selections}
    if parser.peek("WHERE"):
        parser.take("WHERE")
        filters = []
        while True:
            column = parser.name()
            op = parser.take()[1].casefold()
            if op == "is":
                op = "not_null" if parser.peek("NOT") else "is_null"
                if parser.peek("NOT"):
                    parser.take("NOT")
                parser.take("NULL")
                value = None
            elif op == "in":
                parser.take("(")
                value = [parser.literal()]
                while parser.peek(","):
                    parser.take(",")
                    value.append(parser.literal())
                parser.take(")")
            elif op == "between":
                value = [parser.literal()]
                parser.take("AND")
                value.append(parser.literal())
            else:
                value = parser.literal()
            filters.append({"column": column, "op": op, "value": value})
            if not parser.peek("AND"):
                break
            parser.take("AND")
        spec["filters"] = filters
    if parser.peek("GROUP"):
        parser.take("GROUP")
        parser.take("BY")
        spec["group_by"] = parser.names()
    if parser.peek("ORDER"):
        parser.take("ORDER")
        parser.take("BY")
        orders = []
        while True:
            row = {"column": parser.name(), "direction": "asc"}
            if parser.peek("ASC") or parser.peek("DESC"):
                row["direction"] = parser.take()[1].casefold()
            orders.append(row)
            if not parser.peek(","):
                break
            parser.take(",")
        spec["order_by"] = orders
    if parser.peek("LIMIT"):
        parser.take("LIMIT")
        spec["limit"] = parser.literal()
    if parser.peek(";"):
        parser.take(";")
    if parser.position != len(parser.tokens):
        raise SqlQueryError()
    compile_select(spec, tables, max_rows=max_rows)
    return spec
