import pytest
from jsonschema.exceptions import SchemaError

from marvis.plugins.errors import SchemaValidationError
from marvis.plugins.schema_validation import validate_against_schema


def test_validate_against_schema_accepts_valid_object():
    schema = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
        "additionalProperties": False,
    }

    validate_against_schema({"message": "hello"}, schema, label="inputs")


def test_validate_against_schema_reports_missing_field_with_label_and_path():
    schema = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }

    with pytest.raises(SchemaValidationError) as exc_info:
        validate_against_schema({}, schema, label="inputs")

    assert exc_info.value.label == "inputs"
    assert exc_info.value.detail.startswith("$")
    assert "message" in exc_info.value.detail


def test_validate_against_schema_reports_nested_type_path():
    schema = {
        "type": "object",
        "properties": {
            "payload": {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
            }
        },
        "required": ["payload"],
    }

    with pytest.raises(SchemaValidationError) as exc_info:
        validate_against_schema({"payload": {"count": "3"}}, schema, label="output:echo")

    assert exc_info.value.label == "output:echo"
    assert "$.payload.count" in exc_info.value.detail


def test_validate_against_schema_lets_invalid_schema_raise_schema_error():
    schema = {"type": "not-a-json-schema-type"}

    with pytest.raises(SchemaError):
        validate_against_schema({}, schema, label="inputs")
