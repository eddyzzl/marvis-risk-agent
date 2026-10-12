"""Signed immutable mapping revisions, using the existing artifact registry."""
from __future__ import annotations

import base64
from datetime import UTC, datetime
import hashlib
import hmac
import json
from pathlib import Path
import time
import uuid

from marvis.artifacts import ArtifactUnitOfWork
from marvis.db_schema import connect
from marvis.packs.strategy.history_mapping import (
    HistoryMappingError, MAPPING_KIND, MAPPING_VERSION, ORIGIN, SOURCE_KIND,
    canonical, digest, load_source_on_connection, parse_local_instruction, source_cell,
    typed_binding, validate_relations,
)
from marvis.repositories.audit import _write_audit_row
from marvis.repositories.task_artifacts import TaskArtifactRepository


MAX_REVISIONS = 128
MAX_BINDINGS = 128
_PLAN_VERSION = "strategy.history-mapping-proposal.v1"


class StrategyHistoryMappingRepository:
    def __init__(self, settings):
        self.settings = settings

    def _sign(self, value):
        secret = self.settings.plugin_admin_token_path.read_bytes().strip()
        if not secret:
            raise HistoryMappingError("history_authority_unavailable")
        return hmac.new(secret, (MAPPING_VERSION + canonical(value)).encode(), hashlib.sha256).hexdigest()

    @staticmethod
    def _task(conn, task_id):
        row = conn.execute("SELECT task_type FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None or row[0] != "strategy":
            raise HistoryMappingError("history_strategy_task_required", "历史策略资料映射仅适用于当前策略任务。")

    def sources(self, task_id):
        with connect(self.settings.db_path) as conn:
            self._task(conn, task_id)
            rows = conn.execute("SELECT id,content_hash,path FROM task_artifacts WHERE task_id=? AND kind=? ORDER BY created_at,id", (task_id, SOURCE_KIND)).fetchall()
            result = []
            for row in rows:
                entry = {"artifact_id": row["id"], "content_hash": row["content_hash"], "name": Path(row["path"]).suffix[1:].upper() + " · " + row["content_hash"][:12]}
                try:
                    source = load_source_on_connection(conn, self.settings, task_id, row["id"], row["content_hash"])
                    entry.update(status="ready", tables=source["tables"], format=source["format"])
                except HistoryMappingError as exc:
                    entry.update(status="unavailable", code=exc.code, next_action=str(exc))
                except Exception:
                    entry.update(status="unavailable", code="history_source_authentication_failed", next_action="材料登记或内容指纹无法验证；请恢复原始材料或重新登记，不能猜测历史指标。")
                result.append(entry)
            return result

    def cells(self, task_id, source_id, *, offset=0, limit=100):
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 200:
            raise HistoryMappingError("history_cell_page_invalid")
        with connect(self.settings.db_path) as conn:
            self._task(conn, task_id)
            source = load_source_on_connection(conn, self.settings, task_id, source_id)
        locations = list(source["cells"])
        page = [source_cell(source, json.loads(location)) for location in locations[offset:offset + limit]]
        return {"source_id": source_id, "source_hash": source["content_hash"], "cells": page, "offset": offset,
                "next_offset": offset + limit if offset + limit < len(locations) else None, "total_cells": len(locations)}

    @staticmethod
    def head(mapping):
        return {"revision": mapping["revision"], "artifact_id": mapping["artifact_id"], "content_hash": mapping["content_hash"]} if mapping else {"revision": 0, "artifact_id": None, "content_hash": None}

    def _record(self, conn, task_id, row, *, verify_sources=True):
        from marvis.packs.strategy.project_context_tools import _artifact_row, _read_verified_registered_artifact
        record = _artifact_row(row, task_id=task_id, tasks_root=self.settings.tasks_dir)
        if record["kind"] != MAPPING_KIND or record["origin_tool"] != ORIGIN:
            raise HistoryMappingError("history_mapping_authentication_failed")
        raw = _read_verified_registered_artifact(record, tasks_root=self.settings.tasks_dir, max_bytes=512 * 1024)
        try:
            body = json.loads(raw)
            provenance = record["provenance"]
            signed = {key: value for key, value in provenance.items() if key != "signature"}
            expected_path = Path(self.settings.tasks_dir).absolute() / task_id / "strategy_history_mappings" / (record["content_hash"] + ".json")
            if raw != canonical(body).encode() or Path(record["path"]) != expected_path or body["task_id"] != task_id or body["schema_version"] != MAPPING_VERSION or provenance["content_hash"] != record["content_hash"] or provenance["mapping_id"] != body["mapping_id"] or provenance["revision"] != body["revision"] or provenance["actor_id"] != body["actor_id"] or not hmac.compare_digest(provenance["signature"], self._sign(signed)):
                raise ValueError
            audit = conn.execute("SELECT 1 FROM audit WHERE kind='strategy.history_mapping.confirmed' AND target_ref=? AND actor=? AND inputs_hash=? AND outcome='succeeded'", (record["id"], body["actor_id"], provenance["proposal_hash"])).fetchone()
            if audit is None:
                raise ValueError
        except Exception:
            raise HistoryMappingError("history_mapping_authentication_failed", "历史映射的确认身份或来源凭据无法验证，报告不能使用该映射。", status=409) from None
        if verify_sources:
            sources = {}
            for binding in body["bindings"]:
                source_id = binding["source_id"]
                if source_id not in sources:
                    sources[source_id] = load_source_on_connection(conn, self.settings, task_id, source_id, binding["source_hash"])
                request = {key: binding[key] for key in ("binding_key", "source_id", "source_hash", "locator", "cell_hash", "target", "period_key", "version_key")}
                if typed_binding(request, sources[source_id]) != binding:
                    raise HistoryMappingError("history_mapping_binding_changed", status=409)
            validate_relations(body["bindings"])
        return {**body, "artifact_id": record["id"], "content_hash": record["content_hash"]}

    def get_on_connection(self, conn, task_id, artifact_id):
        row = conn.execute("SELECT * FROM task_artifacts WHERE task_id=? AND id=? AND kind=?", (task_id, artifact_id, MAPPING_KIND)).fetchone()
        if row is None:
            raise HistoryMappingError("history_mapping_not_registered")
        return self._record(conn, task_id, row)

    def current_on_connection(self, conn, task_id):
        self._task(conn, task_id)
        rows = conn.execute("SELECT * FROM task_artifacts WHERE task_id=? AND kind=? ORDER BY created_at,id LIMIT ?", (task_id, MAPPING_KIND, MAX_REVISIONS + 1)).fetchall()
        if len(rows) > MAX_REVISIONS:
            raise HistoryMappingError("history_mapping_revision_budget", "历史映射修订超过当前认证范围，请建立新的策略任务保留原修订链。")
        mappings = sorted((self._record(conn, task_id, row, verify_sources=False) for row in rows), key=lambda item: item["revision"])
        parent = None
        for number, mapping in enumerate(mappings, 1):
            if mapping["revision"] != number or mapping["parent_ref"] != self.head(parent):
                raise HistoryMappingError("history_mapping_chain_invalid", status=409)
            parent = mapping
        if parent is None:
            return None
        return self.get_on_connection(conn, task_id, parent["artifact_id"])

    def current(self, task_id):
        with connect(self.settings.db_path) as conn:
            conn.execute("BEGIN")
            return self.current_on_connection(conn, task_id)

    def propose(self, *, task_id, actor_id, expected_head, bindings=None, instruction=None, source_id=None):
        with connect(self.settings.db_path) as conn:
            conn.execute("BEGIN")
            current = self.current_on_connection(conn, task_id)
            if expected_head != self.head(current):
                raise HistoryMappingError("history_mapping_revision_conflict", "历史映射版本已变化，请读取最新版本后重新确认。", status=409)
            if current and current["revision"] >= MAX_REVISIONS:
                raise HistoryMappingError("history_mapping_revision_budget", "历史映射修订超过当前认证范围，请建立新的策略任务保留原修订链。")
            if (bindings is None) == (instruction is None):
                raise HistoryMappingError("history_input_required", "请使用表单映射或完整位置指令，不能同时提交两种输入。")
            sources = {}
            if instruction is not None:
                source = load_source_on_connection(conn, self.settings, task_id, source_id)
                sources[source_id] = source
                bindings = parse_local_instruction(instruction, source)
            if not isinstance(bindings, list) or len(bindings) > MAX_BINDINGS:
                raise HistoryMappingError("history_binding_budget", "每次确认最多包含 128 项完整来源映射；空列表明确清空当前映射，其他内容不会被丢弃。")
            typed = []
            for binding in bindings:
                source_id = binding.get("source_id") if isinstance(binding, dict) else None
                if source_id not in sources:
                    sources[source_id] = load_source_on_connection(conn, self.settings, task_id, source_id)
                typed.append(typed_binding(binding, sources[source_id]))
            validate_relations(typed)
        plan = {"schema_version": _PLAN_VERSION, "proposal_id": uuid.uuid4().hex, "task_id": task_id, "actor_id": actor_id,
                "expected_head": expected_head, "bindings": typed, "expires_at": int(time.time()) + 900,
                "instruction_hash": digest(instruction) if instruction is not None else None}
        envelope = canonical({"body": plan, "signature": self._sign(plan)}).encode()
        if len(envelope) > 48000:
            raise HistoryMappingError("history_proposal_too_large", "映射确认超过当前大小范围，请分组建立完整修订；内容不会被截断。")
        return {"status": "needs_confirmation", "proposal_token": base64.urlsafe_b64encode(envelope).decode(), "bindings": typed,
                "expected_head": expected_head, "historical_declaration": "历史材料中的报告值，未经平台复算；缺失期间、版本或单位保持未知，不能据此宣称 OOT 验证或真实收益。"}

    def confirm(self, *, task_id, actor_id, token):
        try:
            if not isinstance(token, str) or len(token) > 65536:
                raise ValueError
            envelope = json.loads(base64.b64decode(token, altchars=b"-_", validate=True))
            plan = envelope["body"]
            if not hmac.compare_digest(envelope["signature"], self._sign(plan)) or plan["schema_version"] != _PLAN_VERSION or plan["task_id"] != task_id or plan["actor_id"] != actor_id:
                raise ValueError
        except Exception:
            raise HistoryMappingError("history_proposal_identity_mismatch", "历史映射确认身份或内容不匹配，请在当前任务重新建立映射。", status=409) from None
        proposal_hash = digest(plan)
        uow = ArtifactUnitOfWork()
        try:
            with connect(self.settings.db_path) as conn:
                conn.execute("BEGIN IMMEDIATE")
                current = self.current_on_connection(conn, task_id)
                if current and current["proposal_hash"] == proposal_hash:
                    return current
                if plan["expires_at"] < time.time():
                    raise HistoryMappingError("history_proposal_expired", status=409)
                if plan["expected_head"] != self.head(current):
                    raise HistoryMappingError("history_mapping_revision_conflict", "历史映射版本已变化，请读取最新版本并重新确认；未覆盖其他人的映射。", status=409)
                if current and current["revision"] >= MAX_REVISIONS:
                    raise HistoryMappingError("history_mapping_revision_budget", "历史映射修订超过当前认证范围，请建立新的策略任务保留原修订链。")
                sources, typed = {}, []
                for binding in plan["bindings"]:
                    source_id = binding["source_id"]
                    if source_id not in sources:
                        sources[source_id] = load_source_on_connection(conn, self.settings, task_id, source_id, binding["source_hash"])
                    request = {key: binding[key] for key in ("binding_key", "source_id", "source_hash", "locator", "cell_hash", "target", "period_key", "version_key")}
                    normalized = typed_binding(request, sources[source_id])
                    if normalized != binding:
                        raise HistoryMappingError("history_mapping_binding_changed", status=409)
                    typed.append(normalized)
                validate_relations(typed)
                body = {"schema_version": MAPPING_VERSION, "task_id": task_id, "revision": self.head(current)["revision"] + 1,
                        "parent_ref": self.head(current), "actor_id": actor_id, "confirmed_at": datetime.now(UTC).isoformat(),
                        "proposal_hash": proposal_hash, "bindings": typed, "validation_status": "historical_reported_unvalidated"}
                body["mapping_id"] = "history-mapping-" + digest(body)[:24]
                raw = canonical(body).encode()
                content_hash = hashlib.sha256(raw).hexdigest()
                directory = Path(self.settings.tasks_dir).absolute() / task_id / "strategy_history_mappings"
                from marvis.packs.strategy.project_context_tools import _prepare_output_directory
                _prepare_output_directory(directory, tasks_root=self.settings.tasks_dir)
                staged = uow.stage_file(directory, content_hash + ".json")
                staged.path.write_bytes(raw)
                uow.promote_all()
                provenance = {"schema_version": MAPPING_VERSION, "content_hash": content_hash, "mapping_id": body["mapping_id"],
                              "revision": body["revision"], "actor_id": actor_id, "proposal_hash": proposal_hash}
                provenance["signature"] = self._sign(provenance)
                record = TaskArtifactRepository(self.settings.db_path).register_on_connection(conn, task_id=task_id, kind=MAPPING_KIND,
                    path=str(staged.final_path), content_hash=content_hash, origin_tool=ORIGIN, provenance=provenance)
                _write_audit_row(conn, kind="strategy.history_mapping.confirmed", target_ref=record["id"], actor=actor_id,
                    inputs_hash=proposal_hash, outcome="succeeded", detail={"mapping_id": body["mapping_id"], "content_hash": content_hash, "revision": body["revision"]})
                result = self.get_on_connection(conn, task_id, record["id"])
            uow.commit()
            return result
        except Exception:
            uow.rollback()
            raise
