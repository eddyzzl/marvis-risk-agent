from dataclasses import dataclass

from tests.conftest import _requires_pmml_gateway


@dataclass(frozen=True)
class _CollectedItem:
    nodeid: str
    markers: frozenset[str] = frozenset()

    def get_closest_marker(self, name: str):
        return object() if name in self.markers else None


def _item(nodeid: str, *markers: str) -> _CollectedItem:
    return _CollectedItem(nodeid=nodeid, markers=frozenset(markers))


def test_pure_selection_does_not_start_pmml_gateway_even_for_pmml_named_file():
    assert not _requires_pmml_gateway(
        [
            _item("tests/test_strategy_dsl.py::test_spec_round_trip"),
            _item(
                "tests/validation/test_pmml_scoring.py::"
                "test_score_returns_none_when_pmml_output_field_is_null"
            ),
        ]
    )


def test_direct_pmml_runtime_marker_starts_gateway():
    assert _requires_pmml_gateway(
        [
            _item(
                "tests/validation/test_pmml_scoring.py::"
                "test_load_and_score_matches_manual_sigmoid",
                "pmml_runtime",
            ),
        ]
    )


def test_parametrized_pmml_runtime_item_starts_gateway():
    assert _requires_pmml_gateway(
        [
            _item(
                "tests/test_modeling_artifact.py::"
                "test_export_tree_sklearn_wrapper_pmml_can_be_loaded_by_pypmml[lgb]",
                "pmml_runtime",
            ),
        ]
    )


def test_mixed_selection_starts_gateway_when_any_item_has_marker():
    assert _requires_pmml_gateway(
        [
            _item("tests/test_strategy_dsl.py::test_spec_round_trip"),
            _item(
                "tests/test_pipeline_v2.py::"
                "test_v2_pmml_scoring_and_metrics_never_execute_notebook",
                "pmml_runtime",
            ),
        ]
    )
