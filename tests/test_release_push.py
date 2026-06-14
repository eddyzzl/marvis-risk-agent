import importlib.util
from pathlib import Path


def _load_release_push_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "release_push.py"
    spec = importlib.util.spec_from_file_location("release_push", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bump_version_tag_advances_patch_version():
    release_push = _load_release_push_module()

    assert release_push.bump_version_tag("V1.0.0", "patch") == "V1.0.1"


def test_normalize_version_rejects_invalid_tag():
    release_push = _load_release_push_module()

    try:
        release_push.normalize_version("1.0")
    except ValueError as exc:
        assert "V<MAJOR>.<MINOR>.<PATCH>" in str(exc)
    else:
        raise AssertionError("invalid version tag was accepted")


def test_update_release_text_replaces_only_current_release_markers():
    release_push = _load_release_push_module()
    text = (
        "The current V1.0.0 release ships the first workflow.\n"
        "当前 V1.0.0 版本已经稳定落地第一个内置工作流。\n"
        "Example tags: V1.0.0, V1.0.1, V1.1.0.\n"
        "version = \"1.0.0\"\n"
    )

    updated = release_push.update_release_text(text, "1.0.0", "1.0.1")

    assert "The current V1.0.1 release ships the first workflow." in updated
    assert "当前 V1.0.1 版本已经稳定落地第一个内置工作流。" in updated
    assert "Example tags: V1.0.0, V1.0.1, V1.1.0." in updated
    assert 'version = "1.0.1"' in updated


def test_release_files_include_bilingual_readmes():
    release_push = _load_release_push_module()

    release_files = {path.as_posix() for path in release_push.RELEASE_FILES}

    assert "README.md" in release_files
    assert "README.zh-CN.md" in release_files


def test_update_release_text_updates_runtime_version_marker():
    release_push = _load_release_push_module()

    updated = release_push.update_release_text('__version__ = "1.0.0"\n', "1.0.0", "1.0.1")

    assert updated == '__version__ = "1.0.1"\n'


def test_push_release_uses_atomic_push(monkeypatch):
    release_push = _load_release_push_module()
    calls = []

    monkeypatch.setattr(release_push, "run", lambda command: calls.append(command))

    release_push.push_release("origin", "main", "V1.0.1")

    assert calls == [["git", "push", "--atomic", "origin", "main", "V1.0.1"]]


def test_dry_run_release_verifies_clean_main_before_reporting(monkeypatch, capsys):
    release_push = _load_release_push_module()
    calls = []

    monkeypatch.setattr(release_push, "latest_version_tag", lambda: "V1.0.0")
    monkeypatch.setattr(release_push, "tag_exists", lambda tag: False)
    monkeypatch.setattr(release_push, "ensure_clean_worktree", lambda: calls.append("clean"))
    monkeypatch.setattr(release_push, "ensure_on_branch", lambda branch: calls.append(("branch", branch)))

    result = release_push.main(["--dry-run", "--branch", "main", "--no-push"])

    assert result == 0
    assert calls == ["clean", ("branch", "main")]
    assert "verified clean worktree on branch main" in capsys.readouterr().out
