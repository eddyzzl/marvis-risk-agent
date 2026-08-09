import pytest

from marvis.safe_paths import (
    assert_within,
    lexical_child_path,
    prepare_task_output_dir,
    resolve_fixed_task_output,
    safe_filename_component,
)


def test_safe_filename_component_preserves_allowed_chinese_model_text():
    assert safe_filename_component("贷前评分卡 MOB3-v202604") == "贷前评分卡_MOB3-v202604"


def test_safe_filename_component_removes_path_separators_and_reserved_names():
    assert safe_filename_component("../../foo") == "foo"
    assert safe_filename_component("CON") == "_CON"


def test_assert_within_rejects_path_escape(tmp_path):
    parent = tmp_path / "workspace"
    parent.mkdir()
    inside = parent / "tasks" / "report.docx"
    outside = tmp_path / "report.docx"

    assert assert_within(parent, inside) == inside.resolve()
    with pytest.raises(PermissionError):
        assert_within(parent, outside)


def test_lexical_child_path_rejects_intermediate_symlink_but_preserves_final_link(
    tmp_path,
):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file.parquet").write_bytes(b"outside")
    final_parent = root / "task"
    final_parent.mkdir()
    final_link = final_parent / "file.parquet"
    final_link.symlink_to(outside / "file.parquet")

    assert lexical_child_path(root, "task/file.parquet") == final_link
    with pytest.raises(PermissionError, match="escapes root"):
        lexical_child_path(root, "../outside/file.parquet")

    intermediate_link = root / "linked-task"
    intermediate_link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(PermissionError, match="traverses symlink"):
        lexical_child_path(root, "linked-task/file.parquet")


def test_resolve_fixed_task_output_rejects_file_and_output_directory_symlinks(
    tmp_path,
):
    tasks_dir = tmp_path / "tasks"
    output_dir = tasks_dir / "task-1" / "outputs"
    output_dir.mkdir(parents=True)
    artifact = output_dir / "validation.xlsx"
    artifact.write_bytes(b"safe")

    assert resolve_fixed_task_output(
        tasks_dir, "task-1", "validation.xlsx"
    ) == artifact.resolve()

    artifact.unlink()
    outside = tmp_path / "outside.xlsx"
    outside.write_bytes(b"secret")
    artifact.symlink_to(outside)
    assert resolve_fixed_task_output(tasks_dir, "task-1", "validation.xlsx") is None

    artifact.unlink()
    output_dir.rmdir()
    other_outputs = tasks_dir / "task-2" / "outputs"
    other_outputs.mkdir(parents=True)
    (other_outputs / "validation.xlsx").write_bytes(b"other-task")
    output_dir.symlink_to(other_outputs, target_is_directory=True)
    assert resolve_fixed_task_output(tasks_dir, "task-1", "validation.xlsx") is None


def test_prepare_task_output_dir_rejects_existing_directory_symlink(tmp_path):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    task_dir = tasks_dir / "task-1"
    task_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (task_dir / "outputs").symlink_to(outside, target_is_directory=True)

    with pytest.raises(PermissionError, match="must not contain symlinks"):
        prepare_task_output_dir(tasks_dir, "task-1")

    assert list(outside.iterdir()) == []


def test_fixed_task_output_helpers_reject_symlinked_tasks_root(tmp_path):
    real_tasks = tmp_path / "real-tasks"
    artifact = real_tasks / "task-1" / "outputs" / "validation.xlsx"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"secret")
    linked_tasks = tmp_path / "tasks"
    linked_tasks.symlink_to(real_tasks, target_is_directory=True)

    assert resolve_fixed_task_output(
        linked_tasks, "task-1", "validation.xlsx"
    ) is None
    with pytest.raises(PermissionError, match="root must not be a symlink"):
        prepare_task_output_dir(linked_tasks, "task-1")
