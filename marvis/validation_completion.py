"""Read-only compatibility for validation tasks completed before status split.

Older report writers used review_required for findings or missing narrative
metadata even after producing both final files. Execution completion says
nothing about the model's risk findings or business acceptance.
"""

from functools import lru_cache
import json
from pathlib import Path
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile, ZipFile
from zlib import error as CompressionError

from defusedxml.ElementTree import fromstring
from defusedxml.common import DefusedXmlException

from marvis.domain import TASK_TYPE_VALIDATION, TaskRecord, TaskStatus
from marvis.repositories.tasks import AGENT_REPORT_CONCLUSION_KEYS, TaskRepository
from marvis.safe_paths import resolve_fixed_task_output
from marvis.validation.results import validation_results_from_dict

_LEGACY_DELIVERED_MESSAGES = frozenset({
    "报告已生成，需人工复核", "验证已完成，需人工复核报告",
})


def legacy_validation_delivery_complete(
    repo: TaskRepository,
    task: TaskRecord,
    tasks_dir: Path | None,
    *,
    active_job_kind: str | None,
    workflow_status: str | None,
    awaiting_input_confirmation: bool,
) -> bool:
    if (
        task.task_type != TASK_TYPE_VALIDATION
        or task.status is not TaskStatus.REVIEW_REQUIRED
        or task.status_message not in _LEGACY_DELIVERED_MESSAGES
        or task.status_reason_code
        or tasks_dir is None
        or active_job_kind
        or workflow_status not in {None, "done"}
        or awaiting_input_confirmation
    ):
        return False
    if task.run_mode == "agent":
        values, _revision = repo.get_report_values(task.id)
        if not all(str(values.get(key) or "").strip() for key in AGENT_REPORT_CONCLUSION_KEYS):
            return False
        if repo.latest_agent_report_stage(task.id) not in {None, "word_report_ready"}:
            return False
    fingerprints = []
    for filename in ("validation_report.docx", "validation.xlsx", "validation_results.json"):
        path = resolve_fixed_task_output(tasks_dir, task.id, filename)
        if path is None:
            return False
        try:
            stat = path.stat()
        except OSError:
            return False
        if not stat.st_size:
            return False
        fingerprints.append((str(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size))
    return _delivery_files_valid(tuple(fingerprints))


@lru_cache(maxsize=128)
def _delivery_files_valid(fingerprints: tuple[tuple[str, int, int, int], ...]) -> bool:
    """Cache by file identity so task polling doesn't repeatedly parse OOXML."""
    try:
        packages = (
            ("word/document.xml", "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}document"),
            ("xl/workbook.xml", "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}workbook"),
        )
        for fingerprint, (member, root_tag) in zip(fingerprints, packages):
            with ZipFile(fingerprint[0]) as archive:
                if archive.getinfo(member).file_size > 32_000_000 or archive.getinfo("[Content_Types].xml").file_size > 1_000_000:
                    return False
                if fromstring(archive.read(member), forbid_dtd=True).tag != root_tag:
                    return False
                if fromstring(archive.read("[Content_Types].xml"), forbid_dtd=True).tag != (
                    "{http://schemas.openxmlformats.org/package/2006/content-types}Types"
                ):
                    return False
        payload = json.loads(Path(fingerprints[2][0]).read_text(encoding="utf-8"))
        validation_results_from_dict(payload)
    except (
        OSError, BadZipFile, KeyError, ValueError, TypeError, ParseError,
        RuntimeError, NotImplementedError, CompressionError, DefusedXmlException,
    ):
        return False
    return True
