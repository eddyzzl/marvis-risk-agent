import pytest

from riskmodel_checker.safe_paths import assert_within, safe_filename_component


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
