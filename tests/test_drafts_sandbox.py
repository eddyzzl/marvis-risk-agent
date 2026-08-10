import sys

import pytest

import marvis.drafts.sandbox as draft_sandbox_module
import marvis.repositories.drafts as draft_repo_module
from marvis.db import DraftRepository, PluginRepository, init_db
from marvis.drafts import DraftStateError, DraftTool
from marvis.drafts.registry import DraftRegistry
from marvis.drafts.sandbox import DraftSandbox
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner


def _runtime(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    plugin_repo = PluginRepository(db_path)
    tool_registry = ToolRegistry(PluginRegistry(plugin_repo))
    tool_runner = ToolRunner(
        tool_registry,
        plugin_repo,
        python_executable=sys.executable,
        datasets_root=tmp_path / "datasets",
        workspace=tmp_path / "workspace",
    )
    draft_repo = DraftRepository(db_path)
    draft_registry = DraftRegistry(draft_repo)
    return DraftSandbox(tool_runner, draft_registry, draft_repo), draft_registry, draft_repo, plugin_repo, tool_registry


def _draft(code: str) -> DraftTool:
    return DraftTool(
        id="draft-1",
        task_id="task-1",
        name="calc_margin",
        summary="Calculate margin.",
        code=code,
        input_schema={
            "type": "object",
            "properties": {"revenue": {"type": "number"}, "cost": {"type": "number"}},
            "required": ["revenue", "cost"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"margin": {"type": "number"}},
            "required": ["margin"],
            "additionalProperties": False,
        },
        determinism="deterministic",
        source="hand_written",
        learning_note_id=None,
        status="draft",
        created_at="2026-06-19T00:00:00Z",
    )


def test_draft_sandbox_runs_draft_in_subprocess_and_marks_tested(tmp_path):
    sandbox, drafts, repo, audit_repo, _tool_registry = _runtime(tmp_path)
    draft = _draft(
        "def calc_margin(inputs, ctx):\n"
        "    return {'margin': inputs['revenue'] - inputs['cost']}\n"
    )
    drafts.add(draft)

    run = sandbox.run_draft(draft.id, {"revenue": 10, "cost": 3}, task_id="task-1")

    assert run.ok is True
    assert run.output == {"margin": 7}
    assert run.error is None
    assert repo.list_runs(draft.id) == [run]
    assert drafts.get(draft.id).status == "tested"
    audits = audit_repo.list_audit(kind="draft.invoke")
    assert len(audits) == 1
    assert audits[0]["target_ref"] == "draft.calc_margin"
    run_audits = audit_repo.list_audit(kind="draft.run.record")
    assert len(run_audits) == 1
    assert run_audits[0]["target_ref"] == draft.id
    assert run_audits[0]["detail"]["run_id"] == run.id
    assert run_audits[0]["detail"]["status"] == "tested"


def test_draft_sandbox_records_failure_without_raising(tmp_path):
    sandbox, drafts, repo, audit_repo, _tool_registry = _runtime(tmp_path)
    draft = _draft(
        "def calc_margin(inputs, ctx):\n"
        "    return {'margin': inputs['missing']}\n"
    )
    drafts.add(draft)

    run = sandbox.run_draft(draft.id, {"revenue": 10, "cost": 3}, task_id="task-1")

    assert run.ok is False
    assert run.output is None
    assert "missing" in run.error
    assert repo.list_runs(draft.id) == [run]
    assert drafts.get(draft.id).status == "draft"
    run_audit = audit_repo.list_audit(kind="draft.run.record")[0]
    assert run_audit["outcome"] == "failed"
    assert run_audit["detail"]["status"] is None


def test_draft_sandbox_rejects_unsafe_draft_code_before_execution(tmp_path, monkeypatch):
    sandbox, drafts, repo, audit_repo, _tool_registry = _runtime(tmp_path)
    draft = _draft(
        "def calc_margin(inputs, ctx):\n"
        "    open('/tmp/secret.txt').read()\n"
        "    return {'margin': 0}\n"
    )
    drafts.add(draft)
    monkeypatch.setattr(
        sandbox._runner,
        "invoke_adhoc",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unsafe draft should not execute")),
    )

    run = sandbox.run_draft(draft.id, {"revenue": 10, "cost": 3}, task_id="task-1")

    assert run.ok is False
    assert "Draft Language v1" in run.error
    assert "expression statements" in run.error
    assert repo.list_runs(draft.id) == [run]
    assert drafts.get(draft.id).status == "draft"
    run_audit = audit_repo.list_audit(kind="draft.run.record")[0]
    assert run_audit["outcome"] == "failed"
    assert run_audit["detail"]["run_id"] == run.id


def test_draft_sandbox_rejects_path_read_text_before_execution(tmp_path, monkeypatch):
    sandbox, drafts, repo, _audit_repo, _tool_registry = _runtime(tmp_path)
    draft = _draft(
        "def calc_margin(inputs, ctx):\n"
        "    secret = Path('/tmp/secret.txt').read_text()\n"
        "    return {'margin': len(secret)}\n"
    )
    drafts.add(draft)
    monkeypatch.setattr(
        sandbox._runner,
        "invoke_adhoc",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unsafe draft should not execute")),
    )

    run = sandbox.run_draft(draft.id, {"revenue": 10, "cost": 3}, task_id="task-1")

    assert run.ok is False
    assert "Draft Language v1" in run.error
    assert "Path" in run.error
    assert repo.list_runs(draft.id) == [run]


def test_draft_sandbox_rejects_constructed_import_and_native_escape_before_execution(
    tmp_path,
    monkeypatch,
):
    sandbox, drafts, repo, _audit_repo, _tool_registry = _runtime(tmp_path)
    draft = _draft(
        "def calc_margin(inputs, ctx):\n"
        "    import_name = '__im' + 'port__'\n"
        "    importer = getattr(__builtins__, import_name)\n"
        "    os_module = importer('os')\n"
        "    native_module = importer('ct' + 'ypes')\n"
        "    native_pid = native_module.CDLL(None).getpid()\n"
        "    return {'margin': int(native_pid == os_module.getpid())}\n"
    )
    drafts.add(draft)
    monkeypatch.setattr(
        sandbox._runner,
        "invoke_adhoc",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("unsafe draft should not reach the worker")
        ),
    )

    run = sandbox.run_draft(draft.id, {"revenue": 10, "cost": 3}, task_id="task-1")

    assert run.ok is False
    assert "Draft Language v1" in run.error
    assert "getattr" in run.error
    assert repo.list_runs(draft.id) == [run]
    assert drafts.get(draft.id).status == "draft"


def test_draft_worker_rejects_dynamic_native_import_if_ast_gate_is_bypassed(
    tmp_path,
    monkeypatch,
):
    sandbox, drafts, repo, _audit_repo, _tool_registry = _runtime(tmp_path)
    draft = _draft(
        "def calc_margin(inputs, ctx):\n"
        "    import_name = '__im' + 'port__'\n"
        "    importer = __builtins__[import_name]\n"
        "    native_module = importer('ct' + 'ypes')\n"
        "    return {'margin': native_module.CDLL(None).getpid()}\n"
    )
    drafts.add(draft)
    monkeypatch.setattr(
        draft_sandbox_module,
        "assert_draft_code_safe",
        lambda _code, **_kwargs: None,
    )

    run = sandbox.run_draft(draft.id, {"revenue": 10, "cost": 3}, task_id="task-1")

    assert run.ok is False
    assert "Draft Language v1 rewrite required" in run.error
    assert "__builtins__" in run.error
    assert repo.list_runs(draft.id) == [run]
    assert drafts.get(draft.id).status == "draft"


def test_draft_worker_revalidates_module_introspection_when_parent_gate_is_bypassed(
    tmp_path,
    monkeypatch,
):
    sandbox, drafts, repo, _audit_repo, _tool_registry = _runtime(tmp_path)
    draft = _draft(
        "import statistics\n"
        "def calc_margin(inputs, ctx):\n"
        "    builtins_module = statistics.sys.modules['builtins']\n"
        "    import_name = '__im' + 'port__'\n"
        "    importer = builtins_module.__dict__[import_name]\n"
        "    native_module = importer('ct' + 'ypes')\n"
        "    return {'margin': native_module.CDLL(None).getpid()}\n"
    )
    drafts.add(draft)
    monkeypatch.setattr(
        draft_sandbox_module,
        "assert_draft_code_safe",
        lambda _code, **_kwargs: None,
    )

    run = sandbox.run_draft(draft.id, {"revenue": 10, "cost": 3}, task_id="task-1")

    assert run.ok is False
    assert "Draft Language v1 rewrite required" in run.error
    assert "sys" in run.error
    assert repo.list_runs(draft.id) == [run]
    assert drafts.get(draft.id).status == "draft"


def test_draft_worker_revalidates_match_case_escape_when_parent_gate_is_bypassed(
    tmp_path,
    monkeypatch,
):
    sandbox, drafts, repo, _audit_repo, _tool_registry = _runtime(tmp_path)
    draft = _draft(
        "import json\n"
        "def calc_margin(inputs, ctx):\n"
        "    match json.loads:\n"
        "        case object(__globals__=globals_dict):\n"
        "            pass\n"
        "    key = '__' + 'builtins' + '__'\n"
        "    builtins_dict = globals_dict[key]\n"
        "    importer = builtins_dict['__' + 'import' + '__']\n"
        "    os_module = importer('os')\n"
        "    return {'margin': len(os_module.getcwd())}\n"
    )
    drafts.add(draft)
    monkeypatch.setattr(
        draft_sandbox_module,
        "assert_draft_code_safe",
        lambda _code, **_kwargs: None,
    )

    run = sandbox.run_draft(draft.id, {"revenue": 10, "cost": 3}, task_id="task-1")

    assert run.ok is False
    assert "Draft Language v1 rewrite required" in run.error
    assert "match statements" in run.error
    assert repo.list_runs(draft.id) == [run]
    assert drafts.get(draft.id).status == "draft"


def test_draft_sandbox_allows_deterministic_calculation_imports(tmp_path):
    sandbox, drafts, repo, _audit_repo, _tool_registry = _runtime(tmp_path)
    draft = _draft(
        "from __future__ import annotations\n"
        "import collections\n"
        "import functools\n"
        "import itertools\n"
        "import json\n"
        "import math\n"
        "import operator\n"
        "import random\n"
        "import re\n"
        "import statistics\n"
        "import string\n"
        "from decimal import Decimal\n"
        "from fractions import Fraction\n"
        "def calc_margin(inputs, ctx):\n"
        "    delta = Decimal(str(inputs['revenue'])) - Decimal(str(inputs['cost']))\n"
        "    scaled = float(delta) * float(Fraction(1, 1))\n"
        "    assert collections.Counter('aa')['a'] == 2\n"
        "    assert functools.reduce(operator.add, itertools.repeat(1, 2)) == 2\n"
        "    assert json.loads('{\"ok\": true}')['ok'] is True\n"
        "    assert random.Random(7).randint(1, 1) == 1\n"
        "    assert re.fullmatch('[a-z]+', string.ascii_lowercase)\n"
        "    return {'margin': math.floor(statistics.fmean([scaled, scaled]))}\n"
    )
    drafts.add(draft)

    run = sandbox.run_draft(draft.id, {"revenue": 10, "cost": 3}, task_id="task-1")

    assert run.ok is True, run.error
    assert run.output == {"margin": 7}
    assert repo.list_runs(draft.id) == [run]
    assert drafts.get(draft.id).status == "tested"


def test_draft_audit_hook_is_inert_after_restricted_execution_scope():
    """A one-time audit-hook registration must not poison ordinary imports.

    Python audit hooks cannot be unregistered, so the worker deliberately
    gates its hook with a scope flag.  This directly regresses the former
    process-wide ``ctypes`` denial which broke a later ordinary tool import.
    """
    from marvis.plugins import subprocess_worker

    assert subprocess_worker._draft_audit_hook("ctypes.dlopen", ()) is None
    with subprocess_worker._draft_audit_scope():
        with pytest.raises(PermissionError, match="native library access denied"):
            subprocess_worker._draft_audit_hook("ctypes.dlopen", ())
    assert subprocess_worker._draft_audit_hook("ctypes.dlopen", ()) is None


def test_draft_sandbox_rolls_back_run_and_tested_status_when_record_audit_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    sandbox, drafts, repo, audit_repo, _tool_registry = _runtime(tmp_path)
    draft = _draft(
        "def calc_margin(inputs, ctx):\n"
        "    return {'margin': inputs['revenue'] - inputs['cost']}\n"
    )
    drafts.add(draft)
    original_write_audit = draft_repo_module._write_audit_row

    def fail_run_record_audit(conn, *args, **kwargs):
        if kwargs.get("kind") == "draft.run.record":
            raise RuntimeError("audit down")
        return original_write_audit(conn, *args, **kwargs)

    monkeypatch.setattr(draft_repo_module, "_write_audit_row", fail_run_record_audit)

    with pytest.raises(RuntimeError, match="audit down"):
        sandbox.run_draft(draft.id, {"revenue": 10, "cost": 3}, task_id="task-1")

    assert repo.list_runs(draft.id) == []
    assert drafts.get(draft.id).status == "draft"
    assert len(audit_repo.list_audit(kind="draft.invoke")) == 1
    assert audit_repo.list_audit(kind="draft.run.record") == []


def test_draft_sandbox_rejects_task_id_mismatch_without_recording_run(tmp_path):
    sandbox, drafts, repo, _audit_repo, _tool_registry = _runtime(tmp_path)
    draft = _draft(
        "def calc_margin(inputs, ctx):\n"
        "    return {'margin': inputs['revenue'] - inputs['cost']}\n"
    )
    drafts.add(draft)

    with pytest.raises(DraftStateError, match="task mismatch"):
        sandbox.run_draft(draft.id, {"revenue": 10, "cost": 3}, task_id="task-2")

    assert repo.list_runs(draft.id) == []
    assert drafts.get(draft.id).status == "draft"


def test_draft_sandbox_does_not_register_draft_in_planner_catalog(tmp_path):
    sandbox, drafts, _repo, _audit_repo, tool_registry = _runtime(tmp_path)
    drafts.add(
        _draft(
            "def calc_margin(inputs, ctx):\n"
            "    return {'margin': inputs['revenue'] - inputs['cost']}\n"
        )
    )

    sandbox.run_draft("draft-1", {"revenue": 10, "cost": 3}, task_id="task-1")

    assert tool_registry.catalog_for_planner() == []
