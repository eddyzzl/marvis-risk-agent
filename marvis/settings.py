from dataclasses import dataclass
import os
from pathlib import Path


DEFAULT_REPORT_TEMPLATE_NAME = "default.docx"
# Backward-compat fallback for workspaces that still have the legacy template
# file name from the v1 sample project.
_LEGACY_REPORT_TEMPLATE_NAME = "04_贷前评分卡MOB3验证模板_带占位符.docx"

# TST-2: upload guardrails. Excel parsing (openpyxl -> pandas DataFrame) has a
# much higher in-memory amplification factor per byte-on-disk than a streamed
# CSV/parquet read, so it gets a much lower default ceiling. Both are
# overridable via env var (without a code change) for large single-machine
# deployments.
DEFAULT_MAX_CSV_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
DEFAULT_MAX_EXCEL_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB
DEFAULT_MAX_EXCEL_ROWS = 2_000_000
_MAX_CSV_UPLOAD_BYTES_ENV = "MARVIS_MAX_CSV_UPLOAD_BYTES"
_MAX_EXCEL_UPLOAD_BYTES_ENV = "MARVIS_MAX_EXCEL_UPLOAD_BYTES"
_MAX_EXCEL_ROWS_ENV = "MARVIS_MAX_EXCEL_ROWS"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


@dataclass(frozen=True)
class Settings:
    workspace: Path
    report_template_name: str = DEFAULT_REPORT_TEMPLATE_NAME
    max_csv_upload_bytes: int = DEFAULT_MAX_CSV_UPLOAD_BYTES
    max_excel_upload_bytes: int = DEFAULT_MAX_EXCEL_UPLOAD_BYTES
    max_excel_rows: int = DEFAULT_MAX_EXCEL_ROWS

    @classmethod
    def from_workspace(
        cls,
        workspace: str | Path,
        *,
        report_template_name: str = DEFAULT_REPORT_TEMPLATE_NAME,
    ) -> "Settings":
        return build_settings(workspace, report_template_name=report_template_name)

    @property
    def tasks_dir(self) -> Path:
        return self.workspace / "tasks"

    @property
    def plugins_dir(self) -> Path:
        return self.workspace / "plugins"

    @property
    def datasets_dir(self) -> Path:
        return self.workspace / "datasets"

    @property
    def report_template_path(self) -> Path:
        templates_dir = self.workspace / "report_templates"
        primary = templates_dir / self.report_template_name
        if primary.exists():
            return primary
        legacy = templates_dir / _LEGACY_REPORT_TEMPLATE_NAME
        if legacy.exists():
            return legacy
        return primary

    @property
    def db_path(self) -> Path:
        return self.workspace / "marvis.sqlite"


def build_settings(
    workspace: str | Path,
    *,
    report_template_name: str = DEFAULT_REPORT_TEMPLATE_NAME,
) -> Settings:
    path = Path(workspace).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    (path / "tasks").mkdir(parents=True, exist_ok=True)
    (path / "plugins").mkdir(parents=True, exist_ok=True)
    (path / "datasets").mkdir(parents=True, exist_ok=True)
    return Settings(
        workspace=path,
        report_template_name=report_template_name,
        max_csv_upload_bytes=_env_int(_MAX_CSV_UPLOAD_BYTES_ENV, DEFAULT_MAX_CSV_UPLOAD_BYTES),
        max_excel_upload_bytes=_env_int(
            _MAX_EXCEL_UPLOAD_BYTES_ENV, DEFAULT_MAX_EXCEL_UPLOAD_BYTES
        ),
        max_excel_rows=_env_int(_MAX_EXCEL_ROWS_ENV, DEFAULT_MAX_EXCEL_ROWS),
    )
