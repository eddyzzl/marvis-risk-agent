# Governed Plugin and Workflow installation

The default `trusted_local` mode preserves the existing installation and local Workflow discovery behavior. Administrators can explicitly select `governed` mode under `/api/plugins/governance/settings`. This mode rejects unsigned third-party registration, Tool dispatch, local Workflow discovery and ad-hoc/Draft execution. Builtin Tools remain available. Enabled unsigned Plugins and active unsigned local Workflows must be disabled before changing mode. A governed name cannot be replaced through the legacy installer, even in `trusted_local` mode.

The application uses its server-issued local session and the existing workspace Plugin-admin confirmation credential. These do not prove an institution identity. Publisher public-key identity must be checked by the administrator. Governed installation requires the existing `audit` dependency extra (`cryptography`); `/api/plugins/governance` reports signature-verification availability, and missing verification components reject admission with `503`.

## Publisher and archive contract

`POST /api/plugins/governance/publishers` accepts `id`, `display_name` and a Base64-encoded 32-byte Ed25519 public key. Publisher IDs and key fingerprints are unique and cannot be replaced. Revocation is persistent and irreversible; key rotation uses a new publisher/key identity. Revocation prevents new calls of retained versions and their reactivation or rollback. An already dispatched process is not retroactively undone.

A signed ZIP contains one capability directory. A Plugin uses the normal `manifest.json`, its Python package files and `publisher-signature.json`. Its module belongs to its declared package namespace; builtin and platform module shadowing are rejected. A Workflow uses `workflow.json` and `publisher-signature.json`; it is parsed as the existing `WorkflowTemplate`, dry-instantiated and checked by `PlanValidator`. It introduces no additional runtime.

The detached signature document is exactly:

```json
{
  "body": {
    "schema_version": "marvis.capability-signature.v1",
    "kind": "plugin",
    "name": "vendor_pack",
    "version": "1.0.0",
    "publisher_id": "verified_vendor",
    "content_hash": "<64 lowercase hex characters>",
    "declaration_hash": "<64 lowercase hex characters>"
  },
  "signature": "<Base64 Ed25519 signature>"
}
```

Signature bytes are `marvis.capability-signature.v1`, one zero byte, then `canonical(body)` from `marvis.plugins.publisher_trust`. The canonical JSON is UTF-8, sorted keys, compact separators, non-ASCII retained, and rejects non-finite numbers. Duplicate JSON keys are rejected at admission.

`tree_hash(capability_directory)` length-frames every sorted relative file name and exact file bytes. Only the root `publisher-signature.json` is excluded to avoid self-reference. The expanded material is limited to 50 MiB and 1,000 archive members; duplicate members, traversal, symlinks and special files are rejected. The content hash therefore covers code and the exact source manifest or Workflow document.

For Plugins, `declaration_hash` covers `manifest_to_dict(replace(parse_manifest(raw_manifest), checksum='', builtin=False))`. This includes the exact normalized Tools, Hooks, schemas, permissions, policies, execution profiles and release version. Checksums and builtin authority are owned by the platform. For Workflows, it covers the complete `workflow.json` document:

```json
{
  "workflow": {"id": "vendor_flow", "title": "...", "slots": [], "steps": []},
  "tool_bindings": [
    {"plugin": "vendor_pack", "tool": "echo", "version": "1.0.0", "declaration_hash": "<manifest declaration SHA256>"}
  ],
  "permissions": []
}
```

The normal Workflow schema still applies, so `steps` must be nonempty. Every step and granted Tool must name an exact installed version. `tool_bindings` is the sorted, deduplicated list of those references and each current `digest(manifest_to_dict(manifest))`; `permissions` is the sorted union of referenced manifest permissions. Inlined credential keys are refused. Templates must not contain protected records, raw samples, credentials or institution-sensitive document bodies.

## Proposal, confirmation and retained versions

`POST /api/plugins/governance/proposals?kind=plugin|workflow&expected_revision=N` accepts a file upload. `expected_name` can additionally bind an existing target selection. `/api/skills/governance` exposes the same service restricted to Workflows. Signature, publisher, byte hash, declaration, dependency and governance checks precede registration. The response contains the exact proposal hash, current head revision, full declaration and added/removed permissions. A proposal expires after 15 minutes.

`POST .../proposals/{id}/confirm` requires `proposal_hash`, `expected_revision` and the exact `approved_permissions` list. The server session must match the proposal creator and provide workspace-admin confirmation. SQLite binds the immutable version, Plugin registration, new head, confirmation and audit in one transaction. Simultaneous confirmations of different proposals for the same head have one winner; a stale head is rejected. Identical confirmation replay returns the original receipt only while that receipt remains the current head, and still checks permissions, publisher and bytes.

Retained versions are keyed by kind/name/version and exact material hash. An existing version cannot be overwritten. Application-owned files are read-only and verified against the signature and content hash before use. `/rollback-proposals` creates a normal proposal for a retained version, with current-head CAS and a fresh permission confirmation; rollback does not delete other versions. `/enabled` changes availability with head CAS. Managed versions are retained when disabled; the legacy remove/enable/disable paths cannot bypass this policy.

New plans freeze governed Tool versions when saved. A signed Workflow additionally binds its exact immutable version and dependency declarations. Changing current Plugin or Workflow heads leaves these bindings intact. Revocation, disablement, dependency drift and material changes stop dispatch with an explicit admission event. A confirmed plan that has not started is cancelled under the existing state machine; a running plan fails while retaining existing execution evidence. Initial migration of a legacy name is rejected while an unfinished legacy plan still references it.

This is application admission and authenticated byte custody. It does not provide hostile same-UID OS isolation, certify publisher code behavior, establish global resource RBAC, synchronize independent instances, or prove institution identity. Those capabilities use their own explicit contracts and acceptance boundaries.
