"""Independent exact V2 membership, selectors and declared cutoff replay."""
from datetime import date, timedelta
import hashlib
import json
import math
import struct

import pandas as pd

from .runtime_archive_domain import equal
from .runtime_archive_reader import _Unsupported, _file, _original_path

ORDER = [f"{role}/{partition}" for role in ("approval", "risk") for partition in ("development", "validation", "oot")]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate native evidence key")
        result[key] = value
    return result


def selector(row, node, depth=0):
    if depth > 20:
        raise _Unsupported("sample selector depth requires bounded reference")
    if "column" in node:
        return row[node["column"]]
    if "literal" in node:
        return node["literal"]
    op = node["op"]
    if op in {"and", "or"}:
        values = [selector(row, child, depth + 1) for child in node["args"]]
        # SQL selector nulls are not selected. This admitted closed subset has
        # explicit null tests and finite scalar comparisons only.
        return all(values) if op == "and" else any(values)
    if op == "not":
        return not selector(row, node["arg"], depth + 1)
    if op in {"is_null", "is_not_null"}:
        missing = bool(pd.isna(selector(row, node["arg"], depth + 1)))
        return missing if op == "is_null" else not missing
    left, right = selector(row, node["left"], depth + 1), selector(row, node["right"], depth + 1)
    if bool(pd.isna(left)) or bool(pd.isna(right)):
        raise _Unsupported("nullable SQL boolean selector requires three-valued replay")
    operations = {"eq": lambda: left == right, "ne": lambda: left != right, "gt": lambda: left > right,
                  "gte": lambda: left >= right, "lt": lambda: left < right, "lte": lambda: left <= right}
    if op not in operations:
        raise _Unsupported("sample selector operation has no independent reference")
    return bool(operations[op]())


def membership_reference(frame, request):
    rows = frame.to_dict("records")
    populations = {}
    for role in ("approval", "risk"):
        controls = request[f"{role}_population"]
        populations[role] = [
            (True if controls["inclusion"] is None else selector(row, controls["inclusion"]))
            and not (False if controls["exclusion"] is None else selector(row, controls["exclusion"])) for row in rows]
    partitioning = request["partitioning"]
    parts = {}
    for partition in ("development", "validation", "oot"):
        if partitioning["method"] == "time_ranges":
            bounds = partitioning["ranges"][partition]
            dates = [date.fromisoformat(str(v)[:10]) for v in frame[partitioning["column"]]]
            start, end = bounds["start"], bounds["end"]
            parts[partition] = [(start is None or d >= date.fromisoformat(start)) and (end is None or d <= date.fromisoformat(end)) for d in dates]
        elif partitioning["method"] == "predicate_ast":
            parts[partition] = [selector(row, partitioning["selectors"][partition]) for row in rows]
        else:
            raise _Unsupported("sample partitioning requires its declared reference")
    masks = {f"{role}/{partition}": [a and b for a, b in zip(populations[role], parts[partition], strict=True)]
             for role in populations for partition in parts}
    for role in populations:
        for index in range(len(frame)):
            if sum(masks[f"{role}/{partition}"][index] for partition in parts) != int(populations[role][index]):
                raise ValueError("sample partition overlap or missing population")
    if request["relationship"] == "nested_same_cohort" and any(r and not a for name in parts for r, a in zip(masks[f"risk/{name}"], masks[f"approval/{name}"], strict=True)):
        raise ValueError("nested risk rows outside approval")
    maturity = request["maturity"]
    if maturity["status"] in {"confirmed_matured", "not_matured"}:
        cutoff = date.fromisoformat(maturity["cutoff_date"]) - timedelta(days=maturity["performance_window_days"])
        risk = [any(masks[f"risk/{part}"][i] for part in parts) for i in range(len(frame))]
        dates = [date.fromisoformat(str(v)[:10]) for v in frame[request["field_bindings"]["time_field"]]]
        eligible = sum(chosen and day <= cutoff for chosen, day in zip(risk, dates, strict=True))
        if (maturity["status"] == "confirmed_matured" and eligible != sum(risk)
                or maturity["status"] == "not_matured" and (not sum(risk) or eligible >= sum(risk))):
            raise ValueError("sample declared maturity cutoff differs from original dates")
    return masks


def _artifact(session, artifact_id, kind):
    row = session.conn.execute("SELECT * FROM task_artifacts WHERE id=? AND task_id=?", (artifact_id, session.binding.task_id)).fetchone()
    if row is None or row["kind"] != kind or row["origin_tool"] != "strategy.materialize_sample_design_v2_native":
        raise ValueError("native sample artifact owner or producer differs")
    path = row["path"]
    file = _original_path(session.archive, session.manifest["original_workspace"], path) if path.startswith("/") else _file(session.archive, "workspace/" + path)
    raw = file.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row["content_hash"]:
        raise ValueError("native sample artifact bytes changed")
    return dict(row), raw


def authenticated_membership(session, reference, frame):
    record, raw = _artifact(session, reference["artifact_id"], "strategy_sample_design_v2_json")
    if record["content_hash"] != reference["artifact_content_hash"]:
        raise ValueError("native sample JSON reference differs")
    bundle = json.loads(raw, object_pairs_hook=_pairs)
    proof = json.loads(record["provenance_json"], object_pairs_hook=_pairs)
    request = proof["request"]
    if (request["source_mode"] != "native_active_dataset" or request["dataset_id"] != session.binding.dataset_id
            or request["expected_dataset_content_hash"] != session.binding.dataset_sha256
            or proof["request_hash"] != hashlib.sha256(canonical(request)).hexdigest()
            or reference["sample_design_id"] != bundle["sample_design"]["sample_design_id"]
            or reference["sample_design_content_hash"] != bundle["sample_design"]["content_hash"]):
        raise ValueError("sample V2 source or frozen design changed")
    _, encoded = _artifact(session, proof["membership_artifact_id"], "strategy_sample_membership_v2_native_binary")
    if len(encoded) < 16 or encoded[:8] != b"MRVSMB2\x00":
        raise ValueError("native sample membership magic differs")
    length = struct.unpack_from("<Q", encoded, 8)[0]
    if not 0 < length <= 1024**2 or 16 + length > len(encoded):
        raise ValueError("native membership header outside bound")
    header = json.loads(encoded[16:16 + length], object_pairs_hook=_pairs)
    payload = encoded[16 + length:]
    equal(header, bundle["membership"])
    body = {k: v for k, v in header.items() if k not in {"membership_id", "content_hash"}}
    if (header["membership_id"] != "strategy-sample-membership-" + hashlib.sha256(canonical(body)).hexdigest()[:24]
            or header["content_hash"] != hashlib.sha256(canonical({**body, "membership_id": header["membership_id"]})).hexdigest()
            or header["payload_hash"] != hashlib.sha256(payload).hexdigest() or header["row_count"] != len(frame)
            or header["row_ordinal"] != {"start": 0, "stop": len(frame), "step": 1} or header["bitorder"] != "little"
            or header["mask_order"] != ORDER or header["bytes_per_mask"] != math.ceil(len(frame) / 8)
            or header["schema_version"] != "strategy.sample-membership.v2"
            or header["codec_version"] != "marvis.strategy.sample-membership/2"
            or header["task_id"] != session.binding.task_id
            or header["dataset_ref"] != {"dataset_id": session.binding.dataset_id, "content_hash": session.binding.dataset_sha256}):
        raise ValueError("native sample membership identity contract differs")
    masks = membership_reference(frame, request)
    counts = {role: {part: sum(masks[f"{role}/{part}"]) for part in ("development", "validation", "oot")}
              for role in ("approval", "risk")}
    for role in counts:
        counts[role]["total"] = sum(counts[role].values())
    relationships = {}
    for relation, predicate in (("risk_outside_approval", lambda r, a: r and not a), ("risk_within_approval", lambda r, a: r and a)):
        relationships[relation] = {part: sum(predicate(r, a) for r, a in zip(masks[f"risk/{part}"], masks[f"approval/{part}"], strict=True))
                                   for part in ("development", "validation", "oot")}
        relationships[relation]["total"] = sum(relationships[relation].values())
    equal(header["counts"], {**counts, "analysis_universe": len(frame), "relationship": relationships})
    packed = b"".join(bytes(sum(int(v) << j for j, v in enumerate(masks[name][i:i + 8])) for i in range(0, len(frame), 8)) for name in ORDER)
    if payload != packed or header["payload_bytes"] != len(packed):
        raise ValueError("native membership differs from independently replayed source predicates and dates")
    key = reference["partition"]
    if key not in masks:
        raise ValueError("exact sample population/partition unavailable")
    return frame.iloc[[i for i, selected in enumerate(masks[key]) if selected]], masks, request
