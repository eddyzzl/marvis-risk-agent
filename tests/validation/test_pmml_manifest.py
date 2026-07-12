from __future__ import annotations

from pathlib import Path

import pytest

from marvis.validation.pmml_manifest import (
    choose_pmml_output_field,
    parse_pmml_input_manifest,
)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
PMML_NS = "http://www.dmg.org/PMML-4_4"


def _write_pmml(tmp_path: Path, body: str, *, name: str = "model.pmml") -> Path:
    path = tmp_path / name
    path.write_text(
        f"<?xml version='1.0' encoding='UTF-8'?>"
        f"<PMML xmlns='{PMML_NS}' version='4.4'>{body}</PMML>",
        encoding="utf-8",
    )
    return path


def _dictionary(*names: str, target: str | None = None) -> str:
    fields = []
    for name in names:
        if name == target:
            fields.append(
                f"<DataField name='{name}' optype='categorical' dataType='integer'>"
                "<Value value='0'/><Value value='1'/></DataField>"
            )
        else:
            fields.append(
                f"<DataField name='{name}' optype='continuous' dataType='double'/>"
            )
    return f"<DataDictionary>{''.join(fields)}</DataDictionary>"


def test_manifest_resolves_direct_and_derived_field_dependencies_in_order():
    manifest = parse_pmml_input_manifest(FIXTURES / "pmml" / "derived_fields.pmml")

    assert manifest.raw_required_fields == ("age", "income")
    assert manifest.derived_fields == ("age_bucket",)
    assert manifest.model_features == ("age_bucket", "income")
    assert manifest.stress_units[0].model_feature == "age_bucket"
    assert manifest.stress_units[0].raw_input_fields == ("age",)
    assert "Discretize" in " ".join(manifest.stress_units[0].derivation_evidence)
    assert manifest.output_candidates == ("probability_1",)
    assert manifest.algorithm == "xgb"


def test_manifest_excludes_target_supplementary_weight_and_group_fields():
    manifest = parse_pmml_input_manifest(FIXTURES / "min_lr.pmml")
    assert manifest.raw_required_fields == ("x1", "x2")
    assert manifest.model_features == ("x1", "x2")
    assert manifest.algorithm == "lr"


def test_local_scope_shadows_global_and_descendant_local_scope_is_not_merged(tmp_path):
    body = (
        _dictionary("global_raw", "local_raw", "nested_raw", "target", target="target")
        + "<TransformationDictionary>"
        "<DerivedField name='bucket' optype='continuous' dataType='double'>"
        "<FieldRef field='global_raw'/></DerivedField>"
        "</TransformationDictionary>"
        "<MiningModel functionName='classification' algorithmName='LightGBM'>"
        "<MiningSchema><MiningField name='bucket'/><MiningField name='target' usageType='target'/></MiningSchema>"
        "<LocalTransformations>"
        "<DerivedField name='bucket' optype='continuous' dataType='double'>"
        "<NormContinuous field='local_raw'/></DerivedField>"
        "</LocalTransformations>"
        "<Segmentation multipleModelMethod='modelChain'><Segment><True/>"
        "<RegressionModel functionName='classification'>"
        "<MiningSchema><MiningField name='bucket'/></MiningSchema>"
        "<LocalTransformations>"
        "<DerivedField name='bucket' optype='continuous' dataType='double'>"
        "<FieldRef field='nested_raw'/></DerivedField>"
        "</LocalTransformations>"
        "<Output><OutputField name='probability(1)' feature='probability' value='1'/></Output>"
        "</RegressionModel></Segment></Segmentation></MiningModel>"
    )

    manifest = parse_pmml_input_manifest(_write_pmml(tmp_path, body))

    assert manifest.raw_required_fields == ("local_raw",)
    assert manifest.model_features == ("bucket",)
    assert manifest.stress_units[0].raw_input_fields == ("local_raw",)
    assert "nested_raw" not in manifest.raw_required_fields


def test_global_derivation_keeps_global_scope_when_local_name_shadows_dependency(
    tmp_path,
):
    body = (
        _dictionary("global_raw", "local_raw")
        + "<TransformationDictionary>"
        "<DerivedField name='base' optype='continuous' dataType='double'>"
        "<FieldRef field='global_raw'/></DerivedField>"
        "<DerivedField name='top' optype='continuous' dataType='double'>"
        "<FieldRef field='base'/></DerivedField>"
        "</TransformationDictionary>"
        "<RegressionModel functionName='regression'>"
        "<MiningSchema><MiningField name='top'/></MiningSchema>"
        "<LocalTransformations>"
        "<DerivedField name='base' optype='continuous' dataType='double'>"
        "<FieldRef field='local_raw'/></DerivedField>"
        "</LocalTransformations>"
        "</RegressionModel>"
    )

    manifest = parse_pmml_input_manifest(_write_pmml(tmp_path, body))

    assert manifest.raw_required_fields == ("global_raw",)
    assert manifest.stress_units[0].raw_input_fields == ("global_raw",)


def test_only_active_mining_fields_become_scoring_roots(tmp_path):
    body = (
        _dictionary("active", "target", "supplementary", "weight", target="target")
        + "<RegressionModel functionName='classification' algorithmName='logisticRegression'>"
        "<MiningSchema>"
        "<MiningField name='active' usageType='active'/><MiningField name='target' usageType='target'/>"
        "<MiningField name='supplementary' usageType='supplementary'/><MiningField name='weight' usageType='frequencyWeight'/>"
        "</MiningSchema></RegressionModel>"
    )

    manifest = parse_pmml_input_manifest(_write_pmml(tmp_path, body))

    assert manifest.model_features == ("active",)
    assert manifest.raw_required_fields == ("active",)


def test_multi_level_allowlisted_derivations_resolve_to_raw_leaf(tmp_path):
    body = (
        _dictionary("age")
        + "<TransformationDictionary>"
        "<DerivedField name='normalized' optype='continuous' dataType='double'>"
        "<NormContinuous field='age'/></DerivedField>"
        "<DerivedField name='bucket' optype='categorical' dataType='integer'>"
        "<Discretize field='normalized'/></DerivedField>"
        "</TransformationDictionary>"
        "<RegressionModel functionName='regression' algorithmName='logisticRegression'>"
        "<MiningSchema><MiningField name='bucket'/></MiningSchema>"
        "<Output><OutputField name='probability_1' feature='probability' value='1'/></Output>"
        "</RegressionModel>"
    )
    manifest = parse_pmml_input_manifest(_write_pmml(tmp_path, body))

    assert manifest.derived_fields == ("normalized", "bucket")
    assert manifest.raw_required_fields == ("age",)
    assert manifest.stress_units[0].raw_input_fields == ("age",)
    assert manifest.unsupported_derivations == ()


@pytest.mark.parametrize(
    ("transformations", "expected", "raw_fields"),
    [
        (
            "<DerivedField name='a' optype='continuous' dataType='double'><FieldRef field='b'/></DerivedField>"
            "<DerivedField name='b' optype='continuous' dataType='double'><FieldRef field='a'/></DerivedField>",
            "cycle",
            (),
        ),
        (
            "<DerivedField name='a' optype='continuous' dataType='double'><Apply function='+'><FieldRef field='raw'/><Constant>1</Constant></Apply></DerivedField>",
            "Apply",
            ("raw",),
        ),
        (
            "<DerivedField name='a' optype='continuous' dataType='double'><MapValues outputColumn='v'><FieldColumnPair field='raw' column='k'/></MapValues></DerivedField>",
            "MapValues",
            ("raw",),
        ),
    ],
)
def test_cycle_and_unsupported_derivations_are_explicit_without_losing_raw_inputs(
    tmp_path, transformations, expected, raw_fields
):
    body = (
        _dictionary("raw")
        + f"<TransformationDictionary>{transformations}</TransformationDictionary>"
        "<RegressionModel functionName='regression'>"
        "<MiningSchema><MiningField name='a'/></MiningSchema>"
        "<Output><OutputField name='probability_1' feature='probability' value='1'/></Output>"
        "</RegressionModel>"
    )

    manifest = parse_pmml_input_manifest(_write_pmml(tmp_path, body))

    assert manifest.raw_required_fields == raw_fields
    assert manifest.stress_units == ()
    assert any(expected.casefold() in item.casefold() for item in manifest.unsupported_derivations)


def test_norm_discrete_keeps_raw_field_without_guessing_stress_support(tmp_path):
    body = (
        _dictionary("state", "evil")
        + "<TransformationDictionary xmlns:ext='urn:extension'>"
        "<DerivedField name='state_ca' optype='categorical' dataType='double'>"
        "<NormDiscrete field='state' ext:field='evil' value='CA'/></DerivedField>"
        "</TransformationDictionary>"
        "<RegressionModel functionName='regression'>"
        "<MiningSchema><MiningField name='state_ca'/></MiningSchema>"
        "</RegressionModel>"
    )

    manifest = parse_pmml_input_manifest(_write_pmml(tmp_path, body))

    assert manifest.raw_required_fields == ("state",)
    assert manifest.derived_fields == ("state_ca",)
    assert manifest.stress_units == ()
    assert any("NormDiscrete" in item for item in manifest.unsupported_derivations)


def test_unknown_derived_field_reference_is_a_parse_error(tmp_path):
    body = (
        _dictionary("raw")
        + "<TransformationDictionary>"
        "<DerivedField name='a' optype='continuous' dataType='double'><FieldRef field='missing'/></DerivedField>"
        "</TransformationDictionary>"
        "<RegressionModel functionName='regression'>"
        "<MiningSchema><MiningField name='a'/></MiningSchema>"
        "</RegressionModel>"
    )

    with pytest.raises(ValueError, match=r"unknown field reference: missing"):
        parse_pmml_input_manifest(_write_pmml(tmp_path, body))


def test_unknown_reference_error_message_is_bounded(tmp_path):
    missing = "missing_" + ("x" * 1000)
    body = (
        _dictionary("raw")
        + "<TransformationDictionary>"
        f"<DerivedField name='a' optype='continuous' dataType='double'><FieldRef field='{missing}'/></DerivedField>"
        "</TransformationDictionary>"
        "<RegressionModel functionName='regression'><MiningSchema><MiningField name='a'/></MiningSchema></RegressionModel>"
    )

    with pytest.raises(ValueError) as raised:
        parse_pmml_input_manifest(_write_pmml(tmp_path, body))

    assert str(raised.value).startswith("unknown field reference: missing_")
    assert len(str(raised.value)) <= 256


def test_manifest_lists_only_reachable_derived_fields_in_declaration_order(tmp_path):
    body = (
        _dictionary("raw", "other")
        + "<TransformationDictionary>"
        "<DerivedField name='unused_before' optype='continuous' dataType='double'><FieldRef field='other'/></DerivedField>"
        "<DerivedField name='first' optype='continuous' dataType='double'><FieldRef field='raw'/></DerivedField>"
        "<DerivedField name='unused_middle' optype='continuous' dataType='double'><FieldRef field='other'/></DerivedField>"
        "<DerivedField name='second' optype='continuous' dataType='double'><NormContinuous field='first'/></DerivedField>"
        "</TransformationDictionary>"
        "<RegressionModel functionName='regression'><MiningSchema><MiningField name='second'/></MiningSchema></RegressionModel>"
    )

    manifest = parse_pmml_input_manifest(_write_pmml(tmp_path, body))

    assert manifest.derived_fields == ("first", "second")
    assert manifest.raw_required_fields == ("raw",)


def test_duplicate_derived_name_in_same_scope_is_rejected(tmp_path):
    body = (
        _dictionary("x")
        + "<TransformationDictionary>"
        "<DerivedField name='d' optype='continuous' dataType='double'><FieldRef field='x'/></DerivedField>"
        "<DerivedField name='d' optype='continuous' dataType='double'><FieldRef field='x'/></DerivedField>"
        "</TransformationDictionary>"
        "<RegressionModel functionName='regression'><MiningSchema><MiningField name='d'/></MiningSchema></RegressionModel>"
    )
    with pytest.raises(ValueError, match="duplicate DerivedField.*d"):
        parse_pmml_input_manifest(_write_pmml(tmp_path, body))


def test_multiple_top_level_scoring_models_are_ambiguous(tmp_path):
    body = (
        _dictionary("x")
        + "<RegressionModel functionName='regression'><MiningSchema><MiningField name='x'/></MiningSchema></RegressionModel>"
        "<TreeModel functionName='regression'><MiningSchema><MiningField name='x'/></MiningSchema><Node score='0'/></TreeModel>"
    )
    with pytest.raises(ValueError, match="multiple top-level scoring models"):
        parse_pmml_input_manifest(_write_pmml(tmp_path, body))


@pytest.mark.parametrize(
    ("algorithm_name", "expected"),
    [("XGBoost (GBTree)", "xgb"), ("LightGBM", "lgb")],
)
def test_jpmml_model_chain_finds_nested_final_probability_outputs_and_algorithm(
    tmp_path, algorithm_name, expected
):
    body = (
        _dictionary("x1", "target", target="target")
        + f"<MiningModel functionName='classification' algorithmName='{algorithm_name}'>"
        "<MiningSchema><MiningField name='x1'/><MiningField name='target' usageType='target'/></MiningSchema>"
        "<Segmentation multipleModelMethod='modelChain'>"
        "<Segment><True/><TreeModel functionName='regression'>"
        "<MiningSchema><MiningField name='x1'/></MiningSchema>"
        "<Output><OutputField name='intermediate' feature='transformedValue' isFinalResult='false'/></Output>"
        "<Node score='0'/></TreeModel></Segment>"
        "<Segment><True/><RegressionModel functionName='classification'>"
        "<MiningSchema><MiningField name='intermediate'/></MiningSchema>"
        "<Output>"
        "<OutputField name='ignored_probability' feature='probability' value='1' isFinalResult='false'/>"
        "<OutputField name='probability(0)' feature='probability' value='0'/>"
        "<OutputField name='probability(1)' feature='probability' value='1'/>"
        "</Output></RegressionModel></Segment>"
        "</Segmentation></MiningModel>"
    )
    manifest = parse_pmml_input_manifest(_write_pmml(tmp_path, body))

    assert manifest.raw_required_fields == ("x1",)
    assert manifest.algorithm == expected
    assert manifest.output_candidates == ("probability(0)", "probability(1)")


def test_sum_model_does_not_leak_nested_probability_outputs(tmp_path):
    body = (
        _dictionary("x")
        + "<MiningModel functionName='regression' algorithmName='LightGBM'>"
        "<MiningSchema><MiningField name='x'/></MiningSchema>"
        "<Segmentation multipleModelMethod='sum'><Segment><True/>"
        "<TreeModel functionName='regression'><MiningSchema><MiningField name='x'/></MiningSchema>"
        "<Output><OutputField name='internal_probability' feature='probability' value='1'/></Output>"
        "<Node score='0'/></TreeModel></Segment></Segmentation></MiningModel>"
    )

    manifest = parse_pmml_input_manifest(_write_pmml(tmp_path, body))

    assert manifest.output_candidates == ()


def test_model_chain_uses_only_final_direct_segment_output(tmp_path):
    body = (
        _dictionary("x")
        + "<MiningModel functionName='classification' algorithmName='XGBoost (GBTree)'>"
        "<MiningSchema><MiningField name='x'/></MiningSchema>"
        "<Segmentation multipleModelMethod='modelChain'>"
        "<Segment><True/><RegressionModel functionName='regression'>"
        "<MiningSchema><MiningField name='x'/></MiningSchema>"
        "<Output><OutputField name='early_probability' feature='probability' value='1'/></Output>"
        "</RegressionModel></Segment>"
        "<Segment><True/><RegressionModel functionName='classification'>"
        "<MiningSchema><MiningField name='x'/></MiningSchema>"
        "<Output><OutputField name='final_probability' feature='probability' value='1'/></Output>"
        "</RegressionModel></Segment>"
        "</Segmentation></MiningModel>"
    )

    manifest = parse_pmml_input_manifest(_write_pmml(tmp_path, body))

    assert manifest.output_candidates == ("final_probability",)


def test_nested_model_chain_recurses_only_through_each_final_segment(tmp_path):
    body = (
        _dictionary("x")
        + "<MiningModel functionName='classification' algorithmName='LightGBM'>"
        "<MiningSchema><MiningField name='x'/></MiningSchema>"
        "<Segmentation multipleModelMethod='modelChain'>"
        "<Segment><True/><TreeModel functionName='regression'><MiningSchema><MiningField name='x'/></MiningSchema><Node score='0'/></TreeModel></Segment>"
        "<Segment><True/><MiningModel functionName='classification'>"
        "<MiningSchema><MiningField name='x'/></MiningSchema>"
        "<Segmentation multipleModelMethod='modelChain'>"
        "<Segment><True/><RegressionModel functionName='regression'><MiningSchema><MiningField name='x'/></MiningSchema>"
        "<Output><OutputField name='nested_early' feature='probability' value='1'/></Output></RegressionModel></Segment>"
        "<Segment><True/><RegressionModel functionName='classification'><MiningSchema><MiningField name='x'/></MiningSchema>"
        "<Output><OutputField name='nested_final' feature='probability' value='1'/></Output></RegressionModel></Segment>"
        "</Segmentation></MiningModel></Segment>"
        "</Segmentation></MiningModel>"
    )

    manifest = parse_pmml_input_manifest(_write_pmml(tmp_path, body))

    assert manifest.output_candidates == ("nested_final",)


def test_top_model_direct_probability_output_takes_precedence_over_chain(tmp_path):
    body = (
        _dictionary("x")
        + "<MiningModel functionName='classification' algorithmName='LightGBM'>"
        "<MiningSchema><MiningField name='x'/></MiningSchema>"
        "<Output><OutputField name='top_probability' feature='probability' value='1'/></Output>"
        "<Segmentation multipleModelMethod='modelChain'><Segment><True/>"
        "<RegressionModel functionName='classification'><MiningSchema><MiningField name='x'/></MiningSchema>"
        "<Output><OutputField name='nested_probability' feature='probability' value='1'/></Output>"
        "</RegressionModel></Segment></Segmentation></MiningModel>"
    )

    manifest = parse_pmml_input_manifest(_write_pmml(tmp_path, body))

    assert manifest.output_candidates == ("top_probability",)


def test_foreign_namespace_output_wrapper_cannot_spoof_probability_output(tmp_path):
    body = (
        _dictionary("x")
        + "<RegressionModel xmlns:ext='urn:extension' functionName='regression'>"
        "<MiningSchema><MiningField name='x'/></MiningSchema>"
        "<ext:Output><OutputField name='spoofed' feature='probability' value='1'/></ext:Output>"
        "</RegressionModel>"
    )

    manifest = parse_pmml_input_manifest(_write_pmml(tmp_path, body))

    assert manifest.output_candidates == ()


def test_foreign_namespace_mining_schema_cannot_spoof_active_fields(tmp_path):
    body = (
        _dictionary("x", "evil")
        + "<RegressionModel xmlns:ext='urn:extension' functionName='regression'>"
        "<ext:MiningSchema><MiningField name='evil'/></ext:MiningSchema>"
        "<MiningSchema><MiningField name='x'/></MiningSchema>"
        "</RegressionModel>"
    )

    manifest = parse_pmml_input_manifest(_write_pmml(tmp_path, body))

    assert manifest.model_features == ("x",)
    assert manifest.raw_required_fields == ("x",)


def test_qualified_attributes_cannot_override_unqualified_pmml_attributes(tmp_path):
    body = (
        "<DataDictionary xmlns:ext='urn:extension'>"
        "<DataField name='x' ext:name='evil' optype='continuous' dataType='double'/>"
        "</DataDictionary>"
        "<MiningModel xmlns:ext='urn:extension' functionName='classification' algorithmName='LightGBM'>"
        "<MiningSchema><MiningField name='x' usageType='active' ext:usageType='target'/></MiningSchema>"
        "<Segmentation multipleModelMethod='modelChain' ext:multipleModelMethod='sum'>"
        "<Segment><True/><RegressionModel functionName='classification'>"
        "<MiningSchema><MiningField name='x'/></MiningSchema>"
        "<Output><OutputField name='final_probability' feature='probability' ext:feature='predictedValue' value='1'/></Output>"
        "</RegressionModel></Segment></Segmentation></MiningModel>"
    )

    manifest = parse_pmml_input_manifest(_write_pmml(tmp_path, body))

    assert manifest.model_features == ("x",)
    assert manifest.raw_required_fields == ("x",)
    assert manifest.output_candidates == ("final_probability",)


def test_qualified_only_pmml_attributes_are_ignored(tmp_path):
    body = (
        _dictionary("x")
        + "<RegressionModel xmlns:ext='urn:extension' functionName='regression'>"
        "<MiningSchema><MiningField name='x' ext:usageType='target'/></MiningSchema>"
        "<Output><OutputField name='not_probability' feature='predictedValue' ext:feature='probability' value='1'/></Output>"
        "</RegressionModel>"
    )

    manifest = parse_pmml_input_manifest(_write_pmml(tmp_path, body))

    assert manifest.model_features == ("x",)
    assert manifest.output_candidates == ()


def test_derived_dependency_depth_is_bounded_before_python_recursion(tmp_path):
    count = 1200
    definitions = "".join(
        "<DerivedField name='d{index}' optype='continuous' dataType='double'>"
        "<FieldRef field='{reference}'/></DerivedField>".format(
            index=index,
            reference=f"d{index + 1}" if index + 1 < count else "raw",
        )
        for index in range(count)
    )
    body = (
        _dictionary("raw")
        + f"<TransformationDictionary>{definitions}</TransformationDictionary>"
        "<RegressionModel functionName='regression'><MiningSchema><MiningField name='d0'/></MiningSchema></RegressionModel>"
    )

    with pytest.raises(ValueError, match="derived dependency depth exceeds limit") as raised:
        parse_pmml_input_manifest(_write_pmml(tmp_path, body))

    assert not isinstance(raised.value, RecursionError)
    assert len(str(raised.value)) < 256


def test_nested_scoring_model_depth_is_bounded_before_python_recursion(tmp_path):
    nested = (
        "<RegressionModel functionName='regression'>"
        "<MiningSchema><MiningField name='x'/></MiningSchema>"
        "</RegressionModel>"
    )
    for _ in range(1200):
        nested = (
            "<MiningModel functionName='regression'>"
            "<MiningSchema><MiningField name='x'/></MiningSchema>"
            "<Segmentation multipleModelMethod='modelChain'><Segment><True/>"
            + nested
            + "</Segment></Segmentation></MiningModel>"
        )
    body = _dictionary("x") + nested

    with pytest.raises(ValueError, match="scoring model depth exceeds limit") as raised:
        parse_pmml_input_manifest(_write_pmml(tmp_path, body))

    assert not isinstance(raised.value, RecursionError)
    assert len(str(raised.value)) < 256


def test_arbitrary_xml_expression_depth_is_bounded_before_stack_growth(tmp_path):
    expression = "<FieldRef field='raw'/>"
    for _ in range(10_000):
        expression = f"<Apply function='identity'>{expression}</Apply>"
    body = (
        _dictionary("raw")
        + "<TransformationDictionary><DerivedField name='deep' optype='continuous' dataType='double'>"
        + expression
        + "</DerivedField></TransformationDictionary>"
        "<RegressionModel functionName='regression'><MiningSchema><MiningField name='deep'/></MiningSchema></RegressionModel>"
    )

    with pytest.raises(ValueError, match="XML depth exceeds limit") as raised:
        parse_pmml_input_manifest(_write_pmml(tmp_path, body))

    assert not isinstance(raised.value, RecursionError)
    assert len(str(raised.value)) < 256


def test_xml_node_count_is_bounded(tmp_path, monkeypatch):
    import marvis.validation.pmml_manifest as module

    monkeypatch.setattr(module, "MAX_XML_NODES", 50, raising=False)
    values = "".join(f"<Value value='{index}'/>" for index in range(100))
    body = (
        "<DataDictionary><DataField name='x' optype='categorical' dataType='integer'>"
        + values
        + "</DataField></DataDictionary>"
        "<RegressionModel functionName='regression'><MiningSchema><MiningField name='x'/></MiningSchema></RegressionModel>"
    )

    with pytest.raises(ValueError, match="XML node count exceeds limit") as raised:
        parse_pmml_input_manifest(_write_pmml(tmp_path, body))

    assert len(str(raised.value)) < 256


def test_pmml_model_inside_foreign_wrapper_is_not_counted_or_captured(
    tmp_path, monkeypatch
):
    import marvis.validation.pmml_manifest as module

    monkeypatch.setattr(module, "MAX_SCORING_MODEL_COUNT", 1)
    body = (
        _dictionary("x", "evil")
        + "<ext:Wrapper xmlns:ext='urn:extension'>"
        "<MiningModel functionName='regression'><MiningSchema><MiningField name='evil'/></MiningSchema></MiningModel>"
        "</ext:Wrapper>"
        "<RegressionModel functionName='regression'><MiningSchema><MiningField name='x'/></MiningSchema></RegressionModel>"
    )

    manifest = parse_pmml_input_manifest(_write_pmml(tmp_path, body))

    assert manifest.model_features == ("x",)
    assert manifest.raw_required_fields == ("x",)


def test_output_alias_resolution_returns_exact_pmml_name(tmp_path):
    body = (
        _dictionary("x", "target", target="target")
        + "<RegressionModel functionName='classification'>"
        "<MiningSchema><MiningField name='x'/><MiningField name='target' usageType='target'/></MiningSchema>"
        "<Output><OutputField name='probability(0)' feature='probability' value='0'/>"
        "<OutputField name='probability(1.0)' feature='probability' value='1'/></Output>"
        "</RegressionModel>"
    )
    manifest = parse_pmml_input_manifest(_write_pmml(tmp_path, body))

    notebook = choose_pmml_output_field(
        manifest, notebook_hint="probability_1", user_confirmation=None
    )
    user = choose_pmml_output_field(
        manifest, notebook_hint=None, user_confirmation="probability_1"
    )

    assert notebook.selected == "probability(1.0)"
    assert notebook.source == "notebook"
    assert user.selected == "probability(1.0)"
    assert user.source == "user"


def test_output_alias_does_not_fuzzy_match_unrelated_names(tmp_path):
    body = (
        _dictionary("x")
        + "<RegressionModel functionName='regression'><MiningSchema><MiningField name='x'/></MiningSchema>"
        "<Output><OutputField name='my_probability_1' feature='probability' value='1'/></Output>"
        "</RegressionModel>"
    )
    manifest = parse_pmml_input_manifest(_write_pmml(tmp_path, body))
    with pytest.raises(ValueError, match="not present"):
        choose_pmml_output_field(
            manifest, notebook_hint=None, user_confirmation="probability_1"
        )


def test_binary_target_synthesizes_standard_probability_candidates(tmp_path):
    body = (
        _dictionary("x", "target", target="target")
        + "<RegressionModel functionName='classification'>"
        "<MiningSchema><MiningField name='x'/><MiningField name='target' usageType='target'/></MiningSchema>"
        "</RegressionModel>"
    )
    manifest = parse_pmml_input_manifest(_write_pmml(tmp_path, body))
    assert manifest.output_candidates == ("probability(0)", "probability(1)")


@pytest.mark.parametrize("token", ["DOCTYPE", "ENTITY"])
def test_manifest_rejects_unsafe_xml_declarations(tmp_path, token):
    path = tmp_path / "unsafe.pmml"
    if token == "DOCTYPE":
        payload = b"<!DOCTYPE PMML><PMML/>"
    else:
        payload = b"<!ENTITY x 'value'><PMML/>"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="DOCTYPE and ENTITY are not allowed"):
        parse_pmml_input_manifest(path)


@pytest.mark.parametrize("encoding", ["utf-16", "utf-16-le", "utf-16-be"])
def test_manifest_rejects_utf16_and_nul_without_bypassing_guard(tmp_path, encoding):
    path = tmp_path / "unsafe.pmml"
    path.write_bytes("<?xml version='1.0'?><!DOCTYPE PMML><PMML/>".encode(encoding))
    with pytest.raises(ValueError, match="UTF-8 or ASCII"):
        parse_pmml_input_manifest(path)


def test_manifest_maps_malformed_xml_to_bounded_validation_error(tmp_path):
    path = tmp_path / "malformed.pmml"
    path.write_text("<PMML><broken></PMML>", encoding="utf-8")
    with pytest.raises(ValueError, match=r"^invalid PMML XML$"):
        parse_pmml_input_manifest(path)


def test_manifest_rejects_non_pmml_root(tmp_path):
    path = tmp_path / "not-pmml.xml"
    path.write_text("<root/>", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be PMML"):
        parse_pmml_input_manifest(path)


def test_manifest_source_path_is_opened_only_once(tmp_path, monkeypatch):
    path = _write_pmml(
        tmp_path,
        _dictionary("x")
        + "<RegressionModel functionName='regression'><MiningSchema><MiningField name='x'/></MiningSchema></RegressionModel>",
    )
    original_open = Path.open
    open_count = 0

    def counting_open(self, *args, **kwargs):
        nonlocal open_count
        if self.resolve() == path.resolve():
            open_count += 1
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)

    parse_pmml_input_manifest(path)

    assert open_count == 1
