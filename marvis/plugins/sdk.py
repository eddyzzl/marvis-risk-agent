"""ARCH-8: shared pack SDK base for tool-runtime construction.

Every pack's ``tools.py`` (and modeling's ``_runtime.py``) built its own
``_Runtime`` class that wires the same five objects from a ``ToolContext``:
``settings``/``datasets_root``/``repo``/``backend``/``registry``, then bolts
on 0-2 pack-specific repositories on top (e.g. strategy's ``strategies``,
data_ops's ``aligner``/``join_engine``, modeling & analysis's
``experiments``/``modeling_repo``). ``PackRuntime`` factors out the common
five-object construction; pack subclasses only declare their own extension
fields via ``_extend()``.

Import-cost note (PERF-5): this module is imported lazily, from inside each
pack's ``tools.py`` (analogous to today's per-pack ``_Runtime``), never from
``marvis.plugins.contracts`` or ``marvis.plugins.subprocess_worker`` at
module load time. ``subprocess_worker.worker_main`` only resolves the pack
module (and transitively this module) inside ``_run_tool``/``_load_module``,
after the job is already dispatched -- so importing ``DatasetRepository``/
``pandas``-adjacent dependencies here carries the same lazy-load profile the
per-pack ``_Runtime`` classes always had. Do not import this module from
``marvis/plugins/contracts.py`` or the worker entrypoint's module-level code.
"""

from __future__ import annotations

from pathlib import Path

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository
from marvis.settings import Settings, build_settings


class PackRuntime:
    """Common five-object construction shared by every pack's tool runtime.

    Subclasses may override ``_extend(ctx)`` to attach additional
    pack-specific repositories/services after the base five are wired.
    """

    settings: Settings
    datasets_root: Path
    repo: DatasetRepository
    backend: DataBackend
    registry: DatasetRegistry

    def __init__(self, ctx):
        self.settings = build_settings(ctx.workspace)
        self.datasets_root = Path(ctx.datasets_root)
        self.repo = DatasetRepository(self.settings.db_path)
        self.backend = DataBackend(self.datasets_root)
        self.registry = DatasetRegistry(self.repo, self.backend, self.datasets_root)
        self._extend(ctx)

    def _extend(self, ctx) -> None:
        """Hook for subclasses to attach pack-specific repositories/services.

        No-op by default -- packs without extensions (e.g. feature) need not
        override this.
        """
