"""Atomically register immutable feature reports with their native input binding."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from marvis.artifacts import ArtifactUnitOfWork
from marvis.feature.errors import FeatureError
from marvis.files import sha256_file
from marvis.output.feature_report import render_feature_report
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.settings import build_settings


def registered_feature_report(inputs, ctx, *, metrics, collinear, binning):
    settings = build_settings(ctx.workspace)
    out_dir = Path(settings.tasks_dir) / ctx.task_id / "feature_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    repository = TaskArtifactRepository(settings.db_path)
    input_hash = hashlib.sha256(json.dumps(inputs, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    with tempfile.TemporaryDirectory(prefix=".feature-report-", dir=out_dir) as temporary:
        rendered = render_feature_report(metrics, Path(temporary) / "report.xlsx", collinear=collinear, binning=binning)
        content_hash = sha256_file(rendered)
        uow = ArtifactUnitOfWork()
        artifact = uow.stage_file(out_dir, f"feature_report_{content_hash[:16]}.xlsx")
        try:
            shutil.copyfile(rendered, artifact.path)
            if sha256_file(artifact.path) != content_hash:
                raise FeatureError("特征报告暂存内容摘要不一致。")

            def register(conn):
                if sha256_file(artifact.final_path) != content_hash:
                    raise FeatureError("特征报告登记前内容摘要不一致。")
                return repository.register_on_connection(conn, task_id=ctx.task_id,
                    kind="feature_report_xlsx", path=str(artifact.final_path), content_hash=content_hash,
                    origin_tool="feature.generate_feature_report", provenance={
                        "schema_version": "feature-report-artifact.v1", "producer_version": "feature.generate_feature_report.v1",
                        "task_id": ctx.task_id, "input_hash": input_hash, "feature_count": len(metrics)})

            record = uow.finalize_with_connection(repository.transaction, register)
        except Exception:
            uow.rollback()
            raise
    return {"report_path": str(artifact.final_path), "artifact_id": record["id"], "artifact_content_hash": content_hash}
