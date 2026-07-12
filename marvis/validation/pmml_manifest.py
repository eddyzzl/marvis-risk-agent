from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import BinaryIO, Iterator
import xml.sax
from xml.sax import handler

from marvis.model_algorithms import normalize_algorithm
from marvis.validation.input_contracts import (
    PMML_INPUT_MANIFEST_SCHEMA,
    PmmlInputManifest,
    StressUnit,
)


MAX_PMML_MANIFEST_BYTES = 512 * 1024 * 1024
SNAPSHOT_MEMORY_BYTES = 8 * 1024 * 1024
COPY_CHUNK_BYTES = 1024 * 1024
MAX_DERIVED_DEPENDENCY_DEPTH = 256
MAX_DERIVED_FIELD_COUNT = 100_000
MAX_SCORING_MODEL_DEPTH = 128
MAX_SCORING_MODEL_COUNT = 100_000
# General XML limits apply to every element, including unsupported expressions
# and foreign extensions. They are deliberately far above realistic tree PMML
# structure while remaining well below Python/runtime resource limits.
MAX_XML_DEPTH = 512
MAX_XML_NODES = 2_000_000
_ALLOWED_XML_ENCODINGS = {"utf-8", "utf8", "us-ascii", "ascii"}
_MODEL_TAGS = frozenset(
    {
        "AnomalyDetectionModel",
        "AssociationModel",
        "BaselineModel",
        "ClusteringModel",
        "GeneralRegressionModel",
        "MiningModel",
        "NaiveBayesModel",
        "NearestNeighborModel",
        "NeuralNetwork",
        "RegressionModel",
        "RuleSetModel",
        "Scorecard",
        "SequenceModel",
        "SupportVectorMachineModel",
        "TextModel",
        "TimeSeriesModel",
        "TreeModel",
    }
)
_ALLOWED_DERIVATIONS = frozenset({"FieldRef", "NormContinuous", "Discretize"})
_XML_ENCODING = re.compile(br"encoding\s*=\s*['\"]([^'\"]+)['\"]", re.I)
_PROBABILITY_UNDERSCORE = re.compile(
    r"^probability_([+-]?(?:\d+(?:\.\d*)?|\.\d+))$", re.I
)
_PROBABILITY_PARENS = re.compile(
    r"^probability\(([+-]?(?:\d+(?:\.\d*)?|\.\d+))\)$", re.I
)


@dataclass(frozen=True)
class OutputFieldResolution:
    selected: str | None
    candidates: tuple[str, ...]
    source: str
    needs_confirmation: bool


@dataclass
class _DerivedDefinition:
    name: str
    expression: str | None
    references: list[str]
    scope: str


@dataclass
class _ActiveDerived:
    depth: int
    definition: _DerivedDefinition
    owner: _ModelCapture | None


@dataclass
class _DataField:
    name: str
    depth: int
    values: list[str] = field(default_factory=list)


@dataclass
class _OutputField:
    attributes: dict[str, str]
    order: int


@dataclass
class _SegmentCapture:
    models: list[_ModelCapture] = field(default_factory=list)


@dataclass
class _ActiveSegment:
    depth: int
    owner: _ModelCapture
    segment: _SegmentCapture


@dataclass
class _ModelCapture:
    tag: str
    attributes: dict[str, str]
    depth: int
    mining_fields: list[dict[str, str]] = field(default_factory=list)
    local_derived: dict[str, _DerivedDefinition] = field(default_factory=dict)
    local_derived_order: list[_DerivedDefinition] = field(default_factory=list)
    outputs: list[_OutputField] = field(default_factory=list)
    segmentation_method: str | None = None
    segmentation_depth: int | None = None
    segments: list[_SegmentCapture] = field(default_factory=list)


class _PmmlManifestHandler(xml.sax.handler.ContentHandler):
    def __init__(self) -> None:
        super().__init__()
        self.stack: list[tuple[str | None, str]] = []
        self.semantic_stack: list[bool] = []
        self.namespace: str | None = None
        self.global_derived: dict[str, _DerivedDefinition] = {}
        self.global_derived_order: list[_DerivedDefinition] = []
        self.derived_stack: list[_ActiveDerived] = []
        self.data_fields: dict[str, _DataField] = {}
        self.active_data_field: _DataField | None = None
        self.model_stack: list[_ModelCapture] = []
        self.top_models: list[_ModelCapture] = []
        self.segment_stack: list[_ActiveSegment] = []
        self.output_order = 0
        self.derived_count = 0
        self.model_count = 0
        self.xml_node_count = 0

    def startElementNS(self, name, qname, attrs) -> None:  # noqa: N802
        del qname
        uri, local = name
        depth = len(self.stack)
        if depth >= MAX_XML_DEPTH:
            raise ValueError("PMML XML depth exceeds limit")
        self.xml_node_count += 1
        if self.xml_node_count > MAX_XML_NODES:
            raise ValueError("PMML XML node count exceeds limit")
        attributes = _attribute_dict(attrs)
        parent = self.stack[-1] if self.stack else None

        if not self.stack:
            if local != "PMML":
                raise ValueError("PMML XML root must be PMML")
            self.namespace = uri

        semantic_pmml = uri == self.namespace and (
            not self.semantic_stack or self.semantic_stack[-1]
        )
        if semantic_pmml and local in _MODEL_TAGS:
            # The SAX parser is iterative, but later dependency/output resolution
            # walks this structure. Reject pathological depth before Python's
            # recursion limit can become the effective security boundary.
            if len(self.model_stack) >= MAX_SCORING_MODEL_DEPTH:
                raise ValueError("PMML scoring model depth exceeds limit")
            self.model_count += 1
            if self.model_count > MAX_SCORING_MODEL_COUNT:
                raise ValueError("PMML scoring model count exceeds limit")
            model = _ModelCapture(local, attributes, depth)
            if (
                self.segment_stack
                and self.model_stack
                and parent == (self.namespace, "Segment")
                and self.segment_stack[-1].depth == depth - 1
                and self.segment_stack[-1].owner is self.model_stack[-1]
            ):
                self.segment_stack[-1].segment.models.append(model)
            elif not self.model_stack and parent == (self.namespace, "PMML"):
                self.top_models.append(model)
            self.model_stack.append(model)

        if (
            semantic_pmml
            and local == "DataField"
            and parent == (self.namespace, "DataDictionary")
            and len(self.stack) >= 2
            and self.stack[-2] == (self.namespace, "PMML")
        ):
            name_value = _required_name(attributes, "DataField")
            self.active_data_field = _DataField(name_value, depth)
        elif (
            semantic_pmml
            and local == "Value"
            and self.active_data_field is not None
            and parent == (self.namespace, "DataField")
            and self.active_data_field.depth == depth - 1
            and "value" in attributes
        ):
            self.active_data_field.values.append(attributes["value"])

        global_derived = (
            semantic_pmml
            and local == "DerivedField"
            and parent == (self.namespace, "TransformationDictionary")
            and len(self.stack) >= 2
            and self.stack[-2] == (self.namespace, "PMML")
        )
        local_derived = (
            semantic_pmml
            and local == "DerivedField"
            and parent == (self.namespace, "LocalTransformations")
            and self.model_stack
            and self.model_stack[-1].depth == depth - 2
            and len(self.stack) >= 2
            and self.stack[-2]
            == (self.namespace, self.model_stack[-1].tag)
        )
        if global_derived or local_derived:
            owner = self.model_stack[-1] if local_derived else None
            definition = _DerivedDefinition(
                name=_required_name(attributes, "DerivedField"),
                expression=None,
                references=[],
                scope="global" if owner is None else "local",
            )
            self.derived_stack.append(_ActiveDerived(depth, definition, owner))

        if semantic_pmml and self.derived_stack:
            active = self.derived_stack[-1]
            if depth == active.depth + 1:
                active.definition.expression = local
            # Raw sample dependencies are broader than the deterministic stress
            # allowlist. Every standard PMML expression attribute named `field`
            # is a required input edge (eg NormDiscrete, TextIndex, MapValues).
            if depth > active.depth:
                reference = attributes.get("field")
                if reference:
                    active.definition.references.append(reference)

        if (
            semantic_pmml
            and local == "Segmentation"
            and self.model_stack
            and parent == (self.namespace, self.model_stack[-1].tag)
            and self.model_stack[-1].depth == depth - 1
        ):
            current_model = self.model_stack[-1]
            if current_model.segmentation_depth is not None:
                raise ValueError("scoring model contains duplicate Segmentation elements")
            current_model.segmentation_method = attributes.get(
                "multipleModelMethod", ""
            ).strip()
            current_model.segmentation_depth = depth
        elif (
            semantic_pmml
            and local == "Segment"
            and self.model_stack
            and parent == (self.namespace, "Segmentation")
            and self.model_stack[-1].segmentation_depth == depth - 1
        ):
            segment = _SegmentCapture()
            owner = self.model_stack[-1]
            owner.segments.append(segment)
            self.segment_stack.append(_ActiveSegment(depth, owner, segment))

        if (
            semantic_pmml
            and local == "MiningField"
            and parent == (self.namespace, "MiningSchema")
            and self.model_stack
            and self.model_stack[-1].depth == depth - 2
            and len(self.stack) >= 2
            and self.stack[-2]
            == (self.namespace, self.model_stack[-1].tag)
        ):
            self.model_stack[-1].mining_fields.append(attributes)
        elif (
            semantic_pmml
            and local == "OutputField"
            and parent == (self.namespace, "Output")
            and self.model_stack
            and self.model_stack[-1].depth == depth - 2
            and len(self.stack) >= 2
            and self.stack[-2]
            == (self.namespace, self.model_stack[-1].tag)
        ):
            self.model_stack[-1].outputs.append(
                _OutputField(attributes, self.output_order)
            )
            self.output_order += 1

        self.stack.append((uri, local))
        self.semantic_stack.append(semantic_pmml)

    def endElementNS(self, name, qname) -> None:  # noqa: N802
        del qname
        uri, local = name
        if (
            not self.stack
            or not self.semantic_stack
            or self.stack[-1] != (uri, local)
        ):
            raise ValueError("invalid PMML XML")
        semantic_pmml = self.semantic_stack[-1]

        if semantic_pmml and local == "DerivedField" and self.derived_stack:
            active = self.derived_stack[-1]
            if active.depth == len(self.stack) - 1:
                self.derived_stack.pop()
                self.derived_count += 1
                if self.derived_count > MAX_DERIVED_FIELD_COUNT:
                    raise ValueError("PMML DerivedField count exceeds limit")
                _store_derived_definition(
                    active.definition,
                    mapping=(
                        self.global_derived
                        if active.owner is None
                        else active.owner.local_derived
                    ),
                    ordered=(
                        self.global_derived_order
                        if active.owner is None
                        else active.owner.local_derived_order
                    ),
                )

        if (
            semantic_pmml
            and local == "DataField"
            and self.active_data_field is not None
            and self.active_data_field.depth == len(self.stack) - 1
        ):
            data_field = self.active_data_field
            if data_field.name in self.data_fields:
                raise ValueError(f"duplicate DataField name: {data_field.name}")
            self.data_fields[data_field.name] = data_field
            self.active_data_field = None

        if semantic_pmml and local == "Segment" and self.segment_stack:
            if self.segment_stack[-1].depth == len(self.stack) - 1:
                self.segment_stack.pop()

        if (
            semantic_pmml
            and local in _MODEL_TAGS
            and self.model_stack
            and self.model_stack[-1].depth == len(self.stack) - 1
        ):
            self.model_stack.pop()

        self.stack.pop()
        self.semantic_stack.pop()


def parse_pmml_input_manifest(pmml_path: Path) -> PmmlInputManifest:
    """Parse only PMML metadata needed for deterministic validation scoring."""

    with _immutable_pmml_snapshot(Path(pmml_path)) as snapshot:
        parsed = _parse_snapshot(snapshot)

    if not parsed.top_models:
        raise ValueError("PMML contains no top-level scoring model")
    if len(parsed.top_models) > 1:
        raise ValueError("PMML contains multiple top-level scoring models")
    model = parsed.top_models[0]
    model_features = _active_model_features(model)
    raw_required_fields, reachable_derived = _expand_raw_dependencies(
        model_features=model_features,
        data_fields=parsed.data_fields,
        global_derived=parsed.global_derived,
        local_derived=model.local_derived,
    )
    derived_fields = tuple(
        definition.name
        for definition in (*parsed.global_derived_order, *model.local_derived_order)
        if _derived_identity(definition) in reachable_derived
    )
    stress_units, unsupported = _resolve_stress_units(
        model_features=model_features,
        data_fields=parsed.data_fields,
        global_derived=parsed.global_derived,
        local_derived=model.local_derived,
    )
    outputs = _output_candidates(model, parsed.data_fields)
    return PmmlInputManifest(
        schema_version=PMML_INPUT_MANIFEST_SCHEMA,
        raw_required_fields=raw_required_fields,
        derived_fields=derived_fields,
        model_features=model_features,
        stress_units=stress_units,
        unsupported_derivations=unsupported,
        output_candidates=outputs,
        algorithm=_infer_algorithm(model),
    )


def choose_pmml_output_field(
    manifest: PmmlInputManifest,
    *,
    notebook_hint: str | None,
    user_confirmation: str | None,
) -> OutputFieldResolution:
    candidates = manifest.output_candidates
    if user_confirmation:
        selected, ambiguous = _match_output_candidate(user_confirmation, candidates)
        if ambiguous:
            raise ValueError("confirmed PMML output field alias is ambiguous")
        if selected is None:
            raise ValueError("confirmed PMML output field is not present in the model")
        return OutputFieldResolution(selected, candidates, "user", False)

    if notebook_hint:
        selected, ambiguous = _match_output_candidate(notebook_hint, candidates)
        if selected is not None and not ambiguous:
            return OutputFieldResolution(selected, candidates, "notebook", False)
    if len(candidates) == 1:
        return OutputFieldResolution(candidates[0], candidates, "pmml", False)
    return OutputFieldResolution(None, candidates, "ambiguous", True)


@contextmanager
def _immutable_pmml_snapshot(path: Path) -> Iterator[BinaryIO]:
    try:
        source = path.open("rb")
    except OSError as exc:
        raise ValueError("unable to read PMML file") from exc
    with source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("PMML path must be a regular file")
        if before.st_size > MAX_PMML_MANIFEST_BYTES:
            raise ValueError("PMML file exceeds manifest inspection limit")
        with tempfile.SpooledTemporaryFile(
            max_size=SNAPSHOT_MEMORY_BYTES, mode="w+b"
        ) as snapshot:
            prefix = bytearray()
            tail = b""
            copied = 0
            while True:
                chunk = source.read(COPY_CHUNK_BYTES)
                if not chunk:
                    break
                copied += len(chunk)
                if copied > MAX_PMML_MANIFEST_BYTES:
                    raise ValueError("PMML file exceeds manifest inspection limit")
                if len(prefix) < 1024:
                    prefix.extend(chunk[: 1024 - len(prefix)])
                if b"\x00" in chunk:
                    raise ValueError("PMML must use UTF-8 or ASCII XML encoding")
                inspected = (tail + chunk).upper()
                if b"<!DOCTYPE" in inspected or b"<!ENTITY" in inspected:
                    raise ValueError("PMML DOCTYPE and ENTITY are not allowed")
                tail = inspected[-16:]
                snapshot.write(chunk)
            after = os.fstat(source.fileno())
            if copied != before.st_size or _file_identity(before) != _file_identity(after):
                raise ValueError("PMML file changed during manifest inspection")
            _validate_xml_encoding(bytes(prefix))
            snapshot.seek(0)
            yield snapshot


def _parse_snapshot(snapshot: BinaryIO) -> _PmmlManifestHandler:
    parser = xml.sax.make_parser()
    parser.setFeature(handler.feature_namespaces, True)
    for feature in (handler.feature_external_ges, handler.feature_external_pes):
        try:
            parser.setFeature(feature, False)
        except (xml.sax.SAXNotRecognizedException, xml.sax.SAXNotSupportedException):
            pass
    content = _PmmlManifestHandler()
    parser.setContentHandler(content)
    try:
        parser.parse(snapshot)
    except xml.sax.SAXParseException as exc:
        raise ValueError("invalid PMML XML") from exc
    return content


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _validate_xml_encoding(prefix: bytes) -> None:
    if prefix.startswith((b"\xff\xfe", b"\xfe\xff", b"\x00\x00\xfe\xff", b"\xff\xfe\x00\x00")):
        raise ValueError("PMML must use UTF-8 or ASCII XML encoding")
    match = _XML_ENCODING.search(prefix)
    if not match:
        return
    try:
        encoding = match.group(1).decode("ascii").casefold()
    except UnicodeDecodeError as exc:
        raise ValueError("PMML must use UTF-8 or ASCII XML encoding") from exc
    if encoding not in _ALLOWED_XML_ENCODINGS:
        raise ValueError("PMML must use UTF-8 or ASCII XML encoding")


def _attribute_dict(attrs) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in attrs.getNames():
        if isinstance(key, tuple):
            uri, local = key
            # PMML attributes are unqualified. A foreign qualified attribute
            # must never override a standard attribute with the same local name.
            if uri not in {None, ""}:
                continue
        else:
            local = str(key)
        result[local] = attrs.getValue(key)
    return result


def _required_name(attributes: dict[str, str], element: str) -> str:
    value = attributes.get("name", "").strip()
    if not value:
        raise ValueError(f"{element} name must be non-empty")
    return value


def _store_derived_definition(
    definition: _DerivedDefinition,
    *,
    mapping: dict[str, _DerivedDefinition],
    ordered: list[_DerivedDefinition],
) -> None:
    if definition.name in mapping:
        raise ValueError(f"duplicate DerivedField name in the same scope: {definition.name}")
    mapping[definition.name] = definition
    ordered.append(definition)


def _active_model_features(model: _ModelCapture) -> tuple[str, ...]:
    features: list[str] = []
    for field_value in model.mining_fields:
        usage = field_value.get("usageType", "active").strip().casefold()
        if usage != "active":
            continue
        name = _required_name(field_value, "MiningField")
        if name in features:
            raise ValueError(f"duplicate active MiningField name: {name}")
        features.append(name)
    return tuple(features)


def _derived_identity(definition: _DerivedDefinition) -> str:
    return f"{definition.scope}:{definition.name}"


def _lookup_derived(
    name: str,
    *,
    lookup_scope: str,
    global_derived: dict[str, _DerivedDefinition],
    local_derived: dict[str, _DerivedDefinition],
) -> _DerivedDefinition | None:
    if lookup_scope == "global":
        return global_derived.get(name)
    return local_derived.get(name) or global_derived.get(name)


def _expand_raw_dependencies(
    *,
    model_features: tuple[str, ...],
    data_fields: dict[str, _DataField],
    global_derived: dict[str, _DerivedDefinition],
    local_derived: dict[str, _DerivedDefinition],
) -> tuple[tuple[str, ...], frozenset[str]]:
    """Expand all reachable references, independent of stress support."""

    raw_fields: list[str] = []
    reachable: set[str] = set()

    def visit(name: str, lookup_scope: str, stack: tuple[str, ...]) -> None:
        if len(stack) >= MAX_DERIVED_DEPENDENCY_DEPTH:
            raise ValueError("PMML derived dependency depth exceeds limit")
        definition = _lookup_derived(
            name,
            lookup_scope=lookup_scope,
            global_derived=global_derived,
            local_derived=local_derived,
        )
        if definition is None:
            if name not in data_fields:
                raise ValueError(
                    f"unknown field reference: {_bounded_identifier(name)}"
                )
            if name not in raw_fields:
                raw_fields.append(name)
            return
        identity = _derived_identity(definition)
        reachable.add(identity)
        if identity in stack:
            return
        for reference in dict.fromkeys(definition.references):
            visit(reference, definition.scope, (*stack, identity))

    for feature in model_features:
        visit(feature, "local", ())
    return tuple(raw_fields), frozenset(reachable)


def _resolve_stress_units(
    *,
    model_features: tuple[str, ...],
    data_fields: dict[str, _DataField],
    global_derived: dict[str, _DerivedDefinition],
    local_derived: dict[str, _DerivedDefinition],
) -> tuple[tuple[StressUnit, ...], tuple[str, ...]]:
    units: list[StressUnit] = []
    unsupported: list[str] = []

    def resolve(
        name: str, lookup_scope: str, stack: tuple[str, ...]
    ) -> tuple[tuple[str, ...], tuple[str, ...], str | None]:
        if len(stack) >= MAX_DERIVED_DEPENDENCY_DEPTH:
            raise ValueError("PMML derived dependency depth exceeds limit")
        definition = _lookup_derived(
            name,
            lookup_scope=lookup_scope,
            global_derived=global_derived,
            local_derived=local_derived,
        )
        if definition is None:
            if name in data_fields:
                return (name,), (), None
            raise ValueError(f"unknown field reference: {_bounded_identifier(name)}")
        identity = _derived_identity(definition)
        if identity in stack:
            chain = " -> ".join((*stack, identity))
            return (), (), f"derivation cycle: {chain}"
        expression = definition.expression or "missing expression"
        if expression not in _ALLOWED_DERIVATIONS:
            return (), (), f"unsupported derivation {expression} for {name}"
        references = tuple(dict.fromkeys(definition.references))
        if len(references) != 1:
            return (), (), f"invalid {expression} dependency count for {name}"
        leaves: list[str] = []
        evidence = [f"{name}:{expression}({references[0]})"]
        for reference in references:
            nested_leaves, nested_evidence, error = resolve(
                reference, definition.scope, (*stack, identity)
            )
            if error:
                return (), (), error
            for leaf in nested_leaves:
                if leaf not in leaves:
                    leaves.append(leaf)
            evidence.extend(nested_evidence)
        return tuple(leaves), tuple(evidence), None

    for feature in model_features:
        leaves, evidence, error = resolve(feature, "local", ())
        if error:
            unsupported.append(f"{feature}: {error}")
            continue
        units.append(StressUnit(feature, leaves, evidence))
    return tuple(units), tuple(unsupported)


def _bounded_identifier(value: str, *, limit: int = 200) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _walk_models(model: _ModelCapture) -> Iterator[_ModelCapture]:
    yield model
    for segment in model.segments:
        for child in segment.models:
            yield from _walk_models(child)


def _output_candidates(
    model: _ModelCapture, data_fields: dict[str, _DataField]
) -> tuple[str, ...]:
    result = _visible_probability_outputs(model)
    if result:
        return result

    target_names = [
        field_value.get("name", "").strip()
        for field_value in model.mining_fields
        if field_value.get("usageType", "active").casefold() in {"target", "predicted"}
    ]
    for target_name in target_names:
        data_field = data_fields.get(target_name)
        if data_field is not None and len(data_field.values) == 2:
            return tuple(f"probability({value})" for value in data_field.values)
    return ()


def _visible_probability_outputs(model: _ModelCapture) -> tuple[str, ...]:
    result: list[str] = []
    for output in sorted(model.outputs, key=lambda value: value.order):
        attributes = output.attributes
        if attributes.get("feature", "").casefold() != "probability":
            continue
        if attributes.get("isFinalResult", "true").casefold() in {"false", "0"}:
            continue
        name = attributes.get("name", "").strip()
        if not name and attributes.get("value") is not None:
            name = f"probability({attributes['value']})"
        if name and name not in result:
            result.append(name)
    if result:
        return tuple(result)
    if (
        model.tag == "MiningModel"
        and (model.segmentation_method or "").casefold() == "modelchain"
        and model.segments
    ):
        final_models = model.segments[-1].models
        if len(final_models) != 1:
            raise ValueError(
                "final modelChain segment must contain exactly one scoring model"
            )
        return _visible_probability_outputs(final_models[0])
    return ()


def _infer_algorithm(model: _ModelCapture) -> str:
    for item in _walk_models(model):
        raw = item.attributes.get("algorithmName", "").strip()
        folded = raw.casefold()
        if "xgboost" in folded:
            return "xgb"
        if "lightgbm" in folded:
            return "lgb"
        if raw:
            try:
                return normalize_algorithm(raw)
            except ValueError:
                pass
    fallback = {
        "RegressionModel": "lr",
        "NeuralNetwork": "dnn",
        "Scorecard": "scorecard",
    }
    return fallback.get(model.tag, "")


def _match_output_candidate(
    requested: str, candidates: tuple[str, ...]
) -> tuple[str | None, bool]:
    if requested in candidates:
        return requested, False
    requested_alias = _probability_alias_key(requested)
    if requested_alias is None:
        return None, False
    matches = [
        candidate
        for candidate in candidates
        if _probability_alias_key(candidate) == requested_alias
    ]
    if len(matches) == 1:
        return matches[0], False
    return None, len(matches) > 1


def _probability_alias_key(value: str) -> str | None:
    text = str(value).strip()
    match = _PROBABILITY_UNDERSCORE.fullmatch(text) or _PROBABILITY_PARENS.fullmatch(
        text
    )
    if not match:
        return None
    try:
        number = Decimal(match.group(1))
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    normalized = number.normalize()
    if normalized == 0:
        normalized = Decimal(0)
    return f"probability:{normalized}"
