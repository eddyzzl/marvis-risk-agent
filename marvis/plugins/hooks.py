from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
import hashlib
import json
import logging

from marvis.repositories.plugins import PluginRepository
from marvis.repositories.hook_deliveries import HookDeliveryRepository
from marvis.plugins.manifest import REQUIRED_HOOK_EVENTS, ToolRef, manifest_to_dict
from marvis.plugins.registry import PluginRegistry
from marvis.plugins.runner import ToolResult


logger = logging.getLogger(__name__)
HookListener = Callable[[str, dict], None]


class HookDispatchResults(list):
    """List-compatible result plus explicit required/optional completion state."""

    def __init__(self):
        super().__init__()
        self.required_failures: list[dict] = []
        self.warnings: list[dict] = []

    def record(self, target: dict, result: ToolResult) -> None:
        if target["kind"] == "plugin" or target["required"]:
            self.append(result)
        if not result.ok:
            failures = self.required_failures if target["required"] else self.warnings
            failures.append({"target_ref": target["ref"], "error_kind": result.error_kind,
                             "error": result.error or "hook delivery failed"})


class HookDispatcher:
    def __init__(
        self,
        plugin_registry: PluginRegistry,
        tool_runner,
        repo: PluginRepository | None = None,
    ):
        self._plugins = plugin_registry
        self._runner = tool_runner
        self._repo = repo
        self._index: dict[str, list[ToolRef]] = {}
        self._listeners: dict[str, list[HookListener]] = defaultdict(list)
        self._required: dict[tuple[str, str], bool] = {}
        self._bindings: dict[str, str] = {}
        self._listener_requirements: dict[tuple[str, str], bool] = {}
        self._listener_bindings: dict[tuple[str, str], str] = {}
        self._deliveries = HookDeliveryRepository(repo.db_path) if repo is not None else None

    def rebuild_index(self) -> None:
        index: dict[str, list[ToolRef]] = defaultdict(list)
        required = {}
        bindings = {}
        for manifest in self._plugins.list():
            manifest_hash = hashlib.sha256(json.dumps(manifest_to_dict(manifest), sort_keys=True).encode()).hexdigest()
            for hook in manifest.hooks:
                if hook.required and hook.event not in REQUIRED_HOOK_EVENTS:
                    raise ValueError(f"required hook event has no durable completion protocol: {hook.event}")
                index[hook.event].append(
                    ToolRef(manifest.name, hook.tool, manifest.version)
                )
                required[(hook.event, _target_ref(index[hook.event][-1]))] = hook.required
                bindings[_target_ref(index[hook.event][-1])] = manifest_hash
        self._index = dict(index)
        self._required = required
        self._bindings = bindings

    def register_listener(self, event: str, listener: HookListener, *, required: bool = False, binding: str = "") -> None:
        if not isinstance(required, bool):
            raise ValueError("required must be a boolean")
        if required and event not in REQUIRED_HOOK_EVENTS:
            raise ValueError(f"required hook event has no durable completion protocol: {event}")
        identity = _listener_ref(listener)
        if any(_listener_ref(item) == identity for item in self._listeners[str(event)]):
            raise ValueError(f"duplicate hook listener identity: {identity}")
        self._listeners[str(event)].append(listener)
        self._listener_requirements[(str(event), identity)] = required
        self._listener_bindings[(str(event), identity)] = str(binding)

    def listener_count(self, event: str) -> int:
        return len(self._listeners.get(str(event), []))

    def prepare(self, event: str, *, event_id: str, task_id: str) -> None:
        """Freeze obligations before review checkpoints or preceding events."""
        self.prepare_many([(event, event_id)], task_id=task_id)

    def prepare_many(self, events: list[tuple[str, str]], *, task_id: str) -> None:
        records = [
            {"event": event, "event_id": identity, "task_id": task_id, "targets": self._targets(event)}
            for event, identity in events
        ]
        if self._deliveries is None:
            if any(target["required"] for item in records for target in item["targets"]):
                raise ValueError("required hook needs durable storage")
            return
        self._deliveries.prepare_events(records)

    def dispatch(self, event: str, payload: dict, *, task_id: str) -> list[ToolResult]:
        if payload.get("event_id"):
            return self._dispatch_durable(event, payload, task_id=task_id)
        # Existing optional event producers retain their previous contract.
        # Required hooks must never run without a persistent event identity.
        targets = self._targets(event)
        if any(target["required"] for target in targets):
            results = HookDispatchResults()
            for target in targets:
                if target["required"]:
                    results.record(target, _delivery_failure("identity", "required hook needs a durable event_id"))
            return results
        return self._dispatch_legacy(event, payload, task_id=task_id)

    def _targets(self, event: str) -> list[dict]:
        return [
            {"ref": _listener_ref(listener), "kind": "listener",
             "required": self._listener_requirements[(event, _listener_ref(listener))],
             **({"binding": self._listener_bindings[(event, _listener_ref(listener))]} if self._listener_bindings.get((event, _listener_ref(listener))) else {})}
            for listener in self._listeners.get(event, [])
        ] + [
            {"ref": _target_ref(ref), "kind": "plugin",
             "binding": self._bindings[_target_ref(ref)],
             "required": self._required[(event, _target_ref(ref))]}
            for ref in self._index.get(event, [])
        ]

    def _dispatch_durable(self, event: str, payload: dict, *, task_id: str) -> HookDispatchResults:
        results = HookDispatchResults()
        targets = self._targets(event)
        event_id = payload["event_id"]
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("hook event_id must be a non-empty string")
        if self._deliveries is None:
            if any(target["required"] for target in targets):
                for target in targets:
                    if target["required"]:
                        results.record(target, _delivery_failure("persistence", "required hook needs durable storage"))
                return results
            return self._dispatch_legacy(event, payload, task_id=task_id)
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
        targets = self._deliveries.prepare_event(
            event_id=event_id, event=event, task_id=task_id,
            payload_hash="sha256:" + hashlib.sha256(encoded).hexdigest(), targets=targets,
            payload=payload,
        )
        listeners = {_listener_ref(item): item for item in self._listeners.get(event, [])}
        plugins = {_target_ref(item): item for item in self._index.get(event, [])}
        for target in targets:
            claimed, delivery = self._deliveries.claim(event_id, target["ref"], required=target["required"])
            if not claimed:
                if delivery["status"] in {"succeeded", "failed", "unknown"} and delivery["result_json"]:
                    result = ToolResult(**json.loads(delivery["result_json"]))
                else:
                    result = _delivery_failure("unknown", "hook delivery has no durable receipt; explicit reconciliation required")
                results.record(target, result)
                continue
            try:
                if target["kind"] == "listener":
                    listener = listeners.get(target["ref"])
                    if listener is None:
                        result = _delivery_failure("binding", "persisted hook listener is unavailable")
                    elif target.get("binding") != (self._listener_bindings.get((event, target["ref"])) or None):
                        result = _delivery_failure("binding", "persisted hook listener binding changed")
                    elif not self._write_listener_started(event, target["ref"], task_id):
                        result = _delivery_failure("audit", "hook start audit failed")
                    else:
                        delivery_payload = payload
                        if target.get("binding"):
                            delivery_payload = {**payload, "_hook_delivery": {"target_ref": target["ref"], "generation": delivery["generation"], "binding": target["binding"]}}
                        listener(event, delivery_payload)
                        result = ToolResult(ok=True, output=None, error=None, error_kind=None, duration_ms=0)
                        self._write_listener_audit(event, target["ref"], task_id, None)
                else:
                    ref = plugins.get(target["ref"])
                    if ref is None or self._bindings.get(target["ref"]) != target.get("binding"):
                        result = _delivery_failure("binding", "persisted hook plugin is unavailable")
                    else:
                        result = self._write_dispatch_started(event, ref, task_id)
                        if result is None:
                            result = self._runner.invoke(ref, payload, task_id=task_id)
                            result = self._write_audit(event, ref, task_id, result) or result
            except Exception as exc:
                # The handler may have acted before raising: failed is not safe
                # to retry. Durable dispatch never retries any claimed attempt.
                result = _delivery_failure("unknown", f"hook execution outcome unknown: {type(exc).__name__}")
            status = "succeeded" if result.ok else ("unknown" if result.error_kind in {"unknown", "audit"} else "failed")
            try:
                receipt = {
                    "ok": result.ok, "output": None,
                    "error": None if result.ok else f"hook delivery failed ({result.error_kind or 'execution'}); inspect hook audit",
                    "error_kind": result.error_kind, "duration_ms": result.duration_ms,
                }
                self._deliveries.finish(event_id, target["ref"], status=status, result=receipt, generation=delivery["generation"])
                result = ToolResult(**receipt)
            except Exception:
                result = _delivery_failure("unknown", "hook receipt could not be persisted; explicit reconciliation required")
            results.record(target, result)
        return results

    def _dispatch_legacy(self, event: str, payload: dict, *, task_id: str) -> list[ToolResult]:
        for listener in self._listeners.get(event, []):
            listener_ref = _listener_ref(listener)
            if not self._write_listener_started(event, listener_ref, task_id):
                continue
            error: Exception | None = None
            try:
                listener(event, payload)
            except Exception as exc:
                error = exc
                logger.warning(
                    "builtin hook listener failed for %s/%s: %s",
                    event,
                    task_id,
                    exc,
                )
            self._write_listener_audit(event, listener_ref, task_id, error)
        results: list[ToolResult] = []
        for ref in self._index.get(event, []):
            start_error = self._write_dispatch_started(event, ref, task_id)
            if start_error is not None:
                results.append(start_error)
                continue
            try:
                result = self._runner.invoke(ref, payload, task_id=task_id)
            except Exception as exc:
                result = ToolResult(
                    ok=False,
                    output=None,
                    error=str(exc),
                    error_kind="hook",
                    duration_ms=0,
                )
            audit_error = self._write_audit(event, ref, task_id, result)
            results.append(audit_error or result)
        return results

    def _write_dispatch_started(
        self,
        event: str,
        ref: ToolRef,
        task_id: str,
    ) -> ToolResult | None:
        if self._repo is None:
            return None
        try:
            self._repo.write_audit(
                kind="hook.dispatch.started",
                target_ref=_target_ref(ref),
                outcome="started",
                detail={
                    "event": event,
                    "task_id": task_id,
                },
            )
        except Exception as exc:
            return _audit_failure_result("start", exc)
        return None

    def _write_audit(
        self,
        event: str,
        ref: ToolRef,
        task_id: str,
        result: ToolResult,
    ) -> ToolResult | None:
        if self._repo is None:
            return None
        try:
            self._repo.write_audit(
                kind="hook.dispatch",
                target_ref=_target_ref(ref),
                outcome="succeeded" if result.ok else "failed",
                detail={
                    "event": event,
                    "task_id": task_id,
                    "error_kind": result.error_kind,
                    "duration_ms": result.duration_ms,
                },
            )
        except Exception as exc:
            return _audit_failure_result("finish", exc, result=result)
        return None

    def _write_listener_started(
        self,
        event: str,
        listener_ref: str,
        task_id: str,
    ) -> bool:
        if self._repo is None:
            return True
        try:
            self._repo.write_audit(
                kind="hook.listener.started",
                target_ref=listener_ref,
                outcome="started",
                detail={
                    "event": event,
                    "task_id": task_id,
                },
            )
        except Exception as exc:
            logger.warning(
                "builtin hook listener checkpoint failed for %s/%s: %s",
                event,
                task_id,
                exc,
            )
            return False
        return True

    def _write_listener_audit(
        self,
        event: str,
        listener_ref: str,
        task_id: str,
        error: Exception | None,
    ) -> None:
        if self._repo is None:
            return
        try:
            self._repo.write_audit(
                kind="hook.listener",
                target_ref=listener_ref,
                outcome="failed" if error else "succeeded",
                detail={
                    "event": event,
                    "task_id": task_id,
                    "error_kind": error.__class__.__name__ if error else None,
                },
            )
        except Exception as exc:
            logger.warning(
                "builtin hook listener audit failed for %s/%s: %s",
                event,
                task_id,
                exc,
            )


def _target_ref(ref: ToolRef) -> str:
    suffix = f"@{ref.version}" if ref.version else ""
    return f"{ref.label()}{suffix}"


def _delivery_failure(kind: str, message: str) -> ToolResult:
    return ToolResult(ok=False, output=None, error=message, error_kind=kind, duration_ms=0)


def _listener_ref(listener: HookListener) -> str:
    module = getattr(listener, "__module__", "")
    name = getattr(listener, "__qualname__", getattr(listener, "__name__", repr(listener)))
    return f"builtin:{module}.{name}".strip(".")


def _audit_failure_result(
    phase: str,
    exc: Exception,
    *,
    result: ToolResult | None = None,
) -> ToolResult:
    detail = {
        "audit_phase": phase,
        "audit_error": str(exc),
    }
    if result is not None:
        detail["result_ok"] = result.ok
        detail["result_error_kind"] = result.error_kind
    return ToolResult(
        ok=False,
        output=None,
        error=f"audit {phase} failed: {exc}",
        error_kind="audit",
        duration_ms=result.duration_ms if result is not None else 0,
        error_detail=detail,
    )
