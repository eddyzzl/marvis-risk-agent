"""Installed, annotation-free structure for the native split controls."""
from __future__ import annotations


def split_configuration_schema():
    column = {"type": "string", "minLength": 1, "maxLength": 128, "pattern": r"^[^\x00]*\S[^\x00]*$"}
    literal = {"type": ["string", "number", "boolean", "null"]}
    condition = {"type": "object", "properties": {
        "col": dict(column), "op": {"type": "string", "enum": ["eq", "ne", "in", "lt", "le", "gt", "ge"]},
        "val": {"anyOf": [literal, {"type": "array", "items": literal}]},
    }, "required": ["col", "op", "val"], "additionalProperties": False,
        "allOf": [{"if": {"properties": {"op": {"const": "in"}}, "required": ["op"]},
                   "then": {"properties": {"val": {"type": "array", "items": literal}}},
                   "else": {"properties": {"val": literal}}}]}
    generated = {"type": "object", "properties": {
        "test_size": {"type": "number", "minimum": 0, "maximum": 1},
        "oot_size": {"type": "number", "minimum": 0, "maximum": 1},
        "oot_by_time": {"anyOf": [dict(column), {"type": "null"}]},
        "random_oot": {"type": "boolean"},
        "group_cols": {"type": "array", "items": dict(column)},
        "rules": {"type": "array", "items": {"type": "object", "properties": {
            "when": {"type": "array", "minItems": 1, "items": condition},
            "assign": {"type": "string", "enum": ["train", "test", "oot"]},
        }, "required": ["when", "assign"], "additionalProperties": False}},
    }, "additionalProperties": False,
        "allOf": [{"if": {"properties": {"oot_by_time": {"type": "string"}}, "required": ["oot_by_time"]},
                   "then": {"properties": {"random_oot": {"const": False}}}}]}
    return {"anyOf": [
        {"type": "object", "properties": {"method": {"type": "string", "const": "existing"}},
         "required": ["method"], "additionalProperties": False},
        generated,
    ]}


def split_configuration_semantics():
    return {"test_size_basis": "non_oot_remainder", "oot_size_basis": "whole_population",
            "existing_membership": "preserve_source", "oot_by_time_value": "column_name"}
