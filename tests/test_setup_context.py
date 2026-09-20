from types import SimpleNamespace

import pytest

from marvis.agent.setup_context import dataset_name, resolve_named_col


@pytest.mark.parametrize(
    "columns,requested,hints,expected",
    [
        (["MOB", "loan_age"], " loan_age ", ("mob",), "loan_age"),
        (["loan_age", "MOB"], "absent", ("mob", "age"), "MOB"),
        (["loan_mob", "MOB"], None, ("mob",), "MOB"),
        (["loan_mob", "observe_mob"], None, ("mob",), "loan_mob"),
        (["income"], None, ("mob",), ""),
        ([], None, ("mob",), ""),
    ],
)
def test_setup_column_resolution_preserves_explicit_hint_and_source_order(
    columns, requested, hints, expected
):
    assert resolve_named_col(columns, requested, hints) == expected


def test_dataset_name_prefers_authenticated_source_identity():
    seen = []

    def identity(dataset_id):
        seen.append(dataset_id)
        return {"original_name": "  原始样本.xlsx  "}

    dataset = SimpleNamespace(id="dataset-1", source_path="/normalized/hash.parquet")
    assert dataset_name(SimpleNamespace(source_identity=identity), dataset) == "原始样本.xlsx"
    assert seen == ["dataset-1"]


@pytest.mark.parametrize("error", [KeyError, OSError, TypeError, ValueError])
def test_dataset_name_falls_back_on_known_source_lookup_errors(error):
    def identity(_dataset_id):
        raise error("not available")

    registry = SimpleNamespace(source_identity=identity)
    assert dataset_name(registry, SimpleNamespace(id="d", source_path="/data/sample.csv")) == "sample.csv"
    assert dataset_name(registry, SimpleNamespace(id="d", source_path=None)) == "d"


def test_dataset_name_does_not_hide_unexpected_repository_errors():
    def identity(_dataset_id):
        raise RuntimeError("repository failure")

    with pytest.raises(RuntimeError, match="repository failure"):
        dataset_name(SimpleNamespace(source_identity=identity), SimpleNamespace(id="d"))
