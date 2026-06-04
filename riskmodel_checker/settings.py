from dataclasses import dataclass
from pathlib import Path


DEFAULT_REPORT_TEMPLATE_NAME = "default.docx"
# Backward-compat fallback for workspaces that still have the legacy template
# file name from the v1 sample project.
_LEGACY_REPORT_TEMPLATE_NAME = "04_贷前评分卡MOB3验证模板_带占位符.docx"


@dataclass(frozen=True)
class Settings:
    workspace: Path
    report_template_name: str = DEFAULT_REPORT_TEMPLATE_NAME

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
        return self.workspace / "riskmodel_checker.sqlite"


def build_settings(
    workspace: str | Path,
    *,
    report_template_name: str = DEFAULT_REPORT_TEMPLATE_NAME,
) -> Settings:
    path = Path(workspace).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    (path / "tasks").mkdir(parents=True, exist_ok=True)
    return Settings(workspace=path, report_template_name=report_template_name)
