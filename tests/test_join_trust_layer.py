"""T3: join dual-path reconciliation + provenance wiring tests.

The join match count is the MVP's core proof point: it is computed two genuinely
independent ways (DuckDB SQL vs a forced pandas recount) and reconciled. These
tests assert:
  - agreement path: real joinable tables -> both paths agree, provenance attached,
    reconcile verdict consistent, no blocking red flag;
  - failure-shape path: an injected divergence between the two match-count paths
    -> a BLOCKING typed red flag in the payload (not a soft warning);
  - the renderer surfaces the red flag + provenance detail.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pandas as pd

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, PluginRepository, init_db
from marvis.packs.data_ops import tools as data_ops_tools
from marvis.plugins.registry import PluginRegistry
from marvis.reconcile import RECONCILE_MISMATCH_FLAG
from marvis.settings import build_settings


def _env(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    # PluginRegistry init is required so DatasetRepository/registry share the db.
    PluginRegistry(PluginRepository(settings.db_path))
    data_repo = DatasetRepository(settings.db_path)
    backend = DataBackend(settings.datasets_dir)
    registry = DatasetRegistry(data_repo, backend, settings.datasets_dir)
    ctx = SimpleNamespace(
        task_id="task-1",
        seed=0,
        workspace=settings.workspace,
        datasets_root=settings.datasets_dir,
    )
    return ctx, registry


def _register(registry, tmp_path, name, frame, *, role):
    path = tmp_path / f"{name}.csv"
    frame.to_csv(path, index=False)
    return registry.register_from_upload("task-1", path, role=role)


def _joinable(registry, tmp_path, n=40):
    phones = [f"138{i:08d}" for i in range(n)]
    anchor = _register(
        registry, tmp_path, "anchor",
        pd.DataFrame({"mobile": phones, "bad_flag": [i % 2 for i in range(n)]}),
        role="sample",
    )
    feature = _register(
        registry, tmp_path, "feature",
        pd.DataFrame({
            "phone_md5": [hashlib.md5(p.encode()).hexdigest() for p in phones],
            "balance": list(range(n)),
        }),
        role="feature",
    )
    return anchor, feature


def _joinable_sha1(registry, tmp_path, n=40):
    """A correct, fully-matching join whose key method is ``hash:sha1`` -- a hash
    algorithm DuckDB cannot express, so the "DuckDB SQL" match-rate path falls back
    to the SAME pure-pandas kernel the "pandas" path uses. The two reconcile paths
    are therefore NOT independent (defect 2)."""
    phones = [f"138{i:08d}" for i in range(n)]
    anchor = _register(
        registry, tmp_path, "anchor",
        pd.DataFrame({"mobile": phones, "bad_flag": [i % 2 for i in range(n)]}),
        role="sample",
    )
    feature = _register(
        registry, tmp_path, "feature",
        pd.DataFrame({
            "phone_sha1": [hashlib.sha1(p.encode()).hexdigest() for p in phones],
            "balance": list(range(n)),
        }),
        role="feature",
    )
    return anchor, feature


def test_propose_join_attaches_reconcile_and_provenance_when_paths_agree(tmp_path):
    ctx, registry = _env(tmp_path)
    anchor, feature = _joinable(registry, tmp_path)

    out = data_ops_tools.tool_propose_join(
        {"anchor_id": anchor.id, "feature_ids": [feature.id]}, ctx
    )

    join = out["joins"][0]
    # Provenance tuple: all four fields present and shaped.
    prov = join["provenance"]
    assert prov["code_version"]  # app version string
    assert prov["dataset_fingerprint"].startswith("sha256:")
    assert prov["params_digest"].startswith("sha256:")
    assert prov["seed"] == 0
    # Reconciliation: two paths, agreeing, non-blocking.
    rec = join["reconcile"]
    assert rec["primary_path"] == "duckdb_sql"
    assert rec["secondary_path"] == "pandas"
    assert rec["consistent"] is True
    assert rec["primary"] == rec["secondary"]
    # Plan-level summary is present and not blocking.
    summary = out["reconcile_summary"]
    assert summary["blocking"] is False
    assert summary["red_flags"] == []


def test_provenance_dataset_fingerprint_reflects_both_source_datasets(tmp_path):
    ctx, registry = _env(tmp_path)
    anchor, feature = _joinable(registry, tmp_path)

    out = data_ops_tools.tool_propose_join(
        {"anchor_id": anchor.id, "feature_ids": [feature.id]}, ctx
    )
    from marvis.provenance import dataset_fingerprint  # noqa: PLC0415

    expected = dataset_fingerprint([
        registry.get(anchor.id).content_hash,
        registry.get(feature.id).content_hash,
    ])
    assert out["joins"][0]["provenance"]["dataset_fingerprint"] == expected


def test_injected_path_divergence_produces_blocking_red_flag(tmp_path, monkeypatch):
    """Failure-shape test: force the pandas path to disagree with the DuckDB path on
    the match count. The reconcile layer MUST turn that into a blocking typed red flag
    in the payload (the whole point: a human sees the disagreement)."""
    ctx, registry = _env(tmp_path)
    anchor, feature = _joinable(registry, tmp_path)

    real_secondary = DataBackend.match_rate_reconcile_secondary

    def diverging_secondary(self, *args, **kwargs):
        matched, sampled = real_secondary(self, *args, **kwargs)
        # Corrupt the second path by one match -> a real divergence, not float noise.
        return matched + 1, sampled

    monkeypatch.setattr(DataBackend, "match_rate_reconcile_secondary", diverging_secondary)

    out = data_ops_tools.tool_propose_join(
        {"anchor_id": anchor.id, "feature_ids": [feature.id]}, ctx
    )

    rec = out["joins"][0]["reconcile"]
    assert rec["consistent"] is False
    summary = out["reconcile_summary"]
    assert summary["blocking"] is True
    flags = summary["red_flags"]
    assert len(flags) == 1
    flag = flags[0]
    assert flag["code"] == RECONCILE_MISMATCH_FLAG
    assert flag["blocking"] is True
    # Both path values must be present so the human sees the divergence.
    assert flag["primary"] != flag["secondary"]
    assert abs(flag["primary"] - flag["secondary"]) == 1.0


def test_renderer_surfaces_reconcile_red_flag_and_provenance(tmp_path, monkeypatch):
    ctx, registry = _env(tmp_path)
    anchor, feature = _joinable(registry, tmp_path)

    real_secondary = DataBackend.match_rate_reconcile_secondary
    monkeypatch.setattr(
        DataBackend,
        "match_rate_reconcile_secondary",
        lambda self, *a, **k: (real_secondary(self, *a, **k)[0] + 1, real_secondary(self, *a, **k)[1]),
    )
    out = data_ops_tools.tool_propose_join(
        {"anchor_id": anchor.id, "feature_ids": [feature.id]}, ctx
    )

    from marvis.agent.renderers import render_tool_output  # noqa: PLC0415

    text, tables = render_tool_output("propose_join", out)
    # Blocking reconcile flag reaches the human as a 🚩 line.
    assert "🚩" in text
    assert "对账不一致" in text
    # Provenance detail table is emitted.
    titles = [t.get("title") for t in tables]
    assert "数字溯源（对账 + 血缘）" in titles


def test_trust_layer_failure_never_breaks_the_proposal(tmp_path, monkeypatch):
    """A trust-layer exception must degrade to "no trust block", never fail the join."""
    ctx, registry = _env(tmp_path)
    anchor, feature = _joinable(registry, tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("trust layer blew up")

    monkeypatch.setattr(data_ops_tools, "_attach_join_trust_layer", boom)
    out = data_ops_tools.tool_propose_join(
        {"anchor_id": anchor.id, "feature_ids": [feature.id]}, ctx
    )
    # Core proposal still intact; trust keys simply absent.
    assert out["joins"][0]["diagnostics"]["fan_out_detected"] is False
    assert "reconcile" not in out["joins"][0]


def _partial_match_large(registry, tmp_path, n=50_000, match_fraction=0.5):
    """A CORRECT join over a large anchor where only a FRACTION of anchor rows match the
    feature. The partial match is what exposes defect 1: if the two reconcile paths sample
    DIFFERENT anchor subsets, they count a different number of the matching rows and diverge
    even though the join itself is perfectly correct."""
    phones = [f"138{i:08d}" for i in range(n)]
    matched = int(n * match_fraction)
    anchor = _register(
        registry, tmp_path, "anchor",
        pd.DataFrame({"mobile": phones, "bad_flag": [i % 2 for i in range(n)]}),
        role="sample",
    )
    feature = _register(
        registry, tmp_path, "feature",
        pd.DataFrame({
            "phone_md5": [hashlib.md5(p.encode()).hexdigest() for p in phones[:matched]],
            "balance": list(range(matched)),
        }),
        role="feature",
    )
    return anchor, feature


def test_correct_large_join_is_not_flagged_as_reconcile_mismatch(tmp_path):
    """Defect 1 (CRITICAL): correct join, anchor in the (5000, 200000] regime with a PARTIAL
    match -> the two reconcile paths must score the SAME sampled anchor keys, so they agree
    and NO blocking red flag is raised.

    With divergent sampling (DuckDB reservoir vs pandas .sample) the two paths counted a
    different number of the matching rows (repro: 2511 vs 2455 over 50k rows) and produced a
    spurious ``reconcile_mismatch_blocking_approval`` flag on a perfectly correct join."""
    ctx, registry = _env(tmp_path)
    anchor, feature = _partial_match_large(registry, tmp_path, n=50_000, match_fraction=0.5)

    out = data_ops_tools.tool_propose_join(
        {"anchor_id": anchor.id, "feature_ids": [feature.id]}, ctx
    )

    rec = out["joins"][0]["reconcile"]
    # A partial match means the count is well below the 5000 sample, so divergent sampling
    # WOULD show up as unequal primary/secondary; identical sampling makes them equal.
    assert rec["primary"] == rec["secondary"], (
        f"paths scored different anchor subsets: {rec['primary']} vs {rec['secondary']}"
    )
    assert rec["consistent"] is True
    summary = out["reconcile_summary"]
    assert summary["blocking"] is False
    assert summary["red_flags"] == []


def test_non_independent_paths_are_not_stamped_as_agreeing(tmp_path):
    """Defect 2 (HIGH): when the DuckDB match-rate path is unavailable (here a
    ``hash:sha1`` method DuckDB cannot express), BOTH reconcile paths collapse to the
    same pure-pandas kernel. They are then guaranteed to agree by construction, so the
    trust layer must NOT present a fake "two paths agree" badge -- it must honestly mark
    the number as NOT independently verified rather than ``consistent=True``."""
    ctx, registry = _env(tmp_path)
    anchor, feature = _joinable_sha1(registry, tmp_path)

    out = data_ops_tools.tool_propose_join(
        {"anchor_id": anchor.id, "feature_ids": [feature.id]}, ctx
    )

    rec = out["joins"][0]["reconcile"]
    # The two paths are NOT independent here -> an honest trust status, not a green agree.
    assert rec.get("trust") == "not_independently_verified", rec
    assert rec.get("consistent") is not True, (
        "same-path self-comparison must not be stamped as an agreeing reconciliation"
    )
    # No blocking flag either -- an unverified number is not a divergence.
    summary = out["reconcile_summary"]
    assert summary["blocking"] is False
    assert summary["red_flags"] == []


def test_renderer_shows_not_independently_verified_instead_of_green_badge(tmp_path):
    """Defect 2 (HIGH): the renderer must display the honest "未独立复核" verdict for a
    non-independent reconciliation, never the ✓ 一致 (agree) badge."""
    ctx, registry = _env(tmp_path)
    anchor, feature = _joinable_sha1(registry, tmp_path)

    out = data_ops_tools.tool_propose_join(
        {"anchor_id": anchor.id, "feature_ids": [feature.id]}, ctx
    )

    from marvis.agent.renderers import render_tool_output  # noqa: PLC0415

    _text, tables = render_tool_output("propose_join", out)
    trust_table = next(t for t in tables if t.get("title") == "数字溯源（对账 + 血缘）")
    verdicts = [row[3] for row in trust_table["rows"]]
    assert any("未独立复核" in v for v in verdicts), verdicts
    assert all("一致" not in v for v in verdicts), verdicts
