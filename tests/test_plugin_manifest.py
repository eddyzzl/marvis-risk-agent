import pytest

from marvis.plugins.errors import ManifestError, SchemaValidationError, ToolExecutionError
from marvis.plugins.manifest import manifest_to_dict, parse_manifest


def _manifest(**overrides):
    data = {
        "name": "_sample",
        "version": "0.1.0",
        "display_name": "Sample Echo Pack",
        "description": "Runtime smoke-test pack",
        "module": "marvis.packs._sample.tools",
        "python_requires": ">=3.10,<3.14",
        "tools": [
            {
                "name": "echo",
                "summary": "Echo a message",
                "input_schema": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
                "output_schema": {
                    "type": "object",
                    "properties": {"echoed": {"type": "string"}},
                    "required": ["echoed"],
                },
                "determinism": "deterministic",
                "timeout_seconds": 10,
                "failure_policy": "fail",
                "entrypoint": "tool_echo",
                "side_effects": ["read:input"],
            }
        ],
        "hooks": [{"event": "task.created", "tool": "echo"}],
        "permissions": ["workspace:read"],
        "checksum": "uploaded-value-is-not-trusted",
    }
    data.update(overrides)
    return data


def test_parse_manifest_round_trips_plugin_and_tool_contract():
    manifest = parse_manifest(_manifest(), builtin=True)

    assert manifest.name == "_sample"
    assert manifest.version == "0.1.0"
    assert manifest.display_name == "Sample Echo Pack"
    assert manifest.module == "marvis.packs._sample.tools"
    assert manifest.python_requires == ">=3.10,<3.14"
    assert manifest.builtin is True
    assert manifest.checksum == ""
    assert len(manifest.tools) == 1
    assert manifest.tools[0].name == "echo"
    assert manifest.tools[0].side_effects == ("read:input",)
    assert manifest.hooks[0].event == "task.created"
    assert manifest.hooks[0].tool == "echo"

    reparsed = parse_manifest(manifest_to_dict(manifest), builtin=True)
    assert reparsed == manifest


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("name", "", "name"),
        ("version", "", "version"),
        ("module", "", "module"),
        ("tools", [], "tools"),
    ],
)
def test_parse_manifest_rejects_missing_required_manifest_fields(field, value, message):
    data = _manifest(**{field: value})

    with pytest.raises(ManifestError, match=message):
        parse_manifest(data)


def test_parse_manifest_rejects_duplicate_tool_names():
    tool = _manifest()["tools"][0]
    data = _manifest(tools=[tool, dict(tool)])

    with pytest.raises(ManifestError, match="duplicate tool"):
        parse_manifest(data)


@pytest.mark.parametrize("version", ["1", "1.0", "v1.0.0", "1.0.0.0"])
def test_parse_manifest_rejects_non_semver_versions(version):
    with pytest.raises(ManifestError, match="semantic version"):
        parse_manifest(_manifest(version=version))


@pytest.mark.parametrize("python_requires", ["3.10", ">=py310", "=>3.10"])
def test_parse_manifest_rejects_invalid_python_requires(python_requires):
    with pytest.raises(ManifestError, match="python_requires"):
        parse_manifest(_manifest(python_requires=python_requires))


def test_parse_manifest_rejects_invalid_tool_contract_values():
    tool = dict(_manifest()["tools"][0])
    tool["determinism"] = "random"

    with pytest.raises(ManifestError, match="determinism"):
        parse_manifest(_manifest(tools=[tool]))

    tool = dict(_manifest()["tools"][0])
    tool["failure_policy"] = "ignore"
    with pytest.raises(ManifestError, match="failure_policy"):
        parse_manifest(_manifest(tools=[tool]))

    tool = dict(_manifest()["tools"][0])
    tool["timeout_seconds"] = 0
    with pytest.raises(ManifestError, match="timeout_seconds"):
        parse_manifest(_manifest(tools=[tool]))


def test_parse_manifest_rejects_unknown_hook_event_and_missing_tool():
    with pytest.raises(ManifestError, match="unknown hook event"):
        parse_manifest(_manifest(hooks=[{"event": "unknown.event", "tool": "echo"}]))

    with pytest.raises(ManifestError, match="hook tool"):
        parse_manifest(_manifest(hooks=[{"event": "task.created", "tool": "missing"}]))


def test_plugin_error_types_carry_context():
    schema_error = SchemaValidationError("inputs", "$.message is required")
    assert schema_error.label == "inputs"
    assert schema_error.detail == "$.message is required"
    assert "inputs schema validation failed" in str(schema_error)

    execution_error = ToolExecutionError("boom", "Traceback text")
    assert execution_error.traceback_text == "Traceback text"
