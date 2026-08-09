"""Durable ownership and cleanup for browser-staged validation materials."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import logging
from pathlib import Path
import shutil
import sqlite3
import uuid

from marvis.db_schema import connect


logger = logging.getLogger(__name__)

DEFAULT_VALIDATION_BATCH_UPLOAD_TTL_SECONDS = 60 * 60
_TOKEN_LENGTH = 32


class ValidationBatchUploadClaimError(RuntimeError):
    """Raised when a staging token is absent, expired, or path-inconsistent."""


@dataclass(frozen=True)
class ValidationBatchUploadRecoveryReport:
    examined: int
    removed: int
    failed: int


class ValidationBatchUploadGarbageCollector:
    """Own staged material tokens until a batch transaction consumes them.

    The ledger makes response-loss observable and bounded.  Startup treats all
    surviving staging claims as abandoned because no HTTP request from the
    previous process can still own them.  Normal upload traffic also sweeps TTL
    expiry so a lost browser response does not require a restart to recover.
    """

    def __init__(
        self,
        db_path: Path,
        *,
        material_uploads_root: Path,
        ttl_seconds: int = DEFAULT_VALIDATION_BATCH_UPLOAD_TTL_SECONDS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if isinstance(ttl_seconds, bool) or int(ttl_seconds) <= 0:
            raise ValueError("validation batch upload TTL must be positive")
        self.db_path = Path(db_path)
        self.material_uploads_root = Path(material_uploads_root).absolute()
        self.ttl_seconds = int(ttl_seconds)
        self._clock = clock or (lambda: datetime.now(UTC))

    def register(self, upload_token: str) -> Path:
        token = _validate_upload_token(upload_token)
        path = self._staging_path(token, require_directory=True)
        now = _utc(self._clock())
        expires_at = now + timedelta(seconds=self.ttl_seconds)
        relative_path = f"staging/{token}"
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO validation_batch_material_uploads(
                    upload_token, relative_path, expires_at, created_at, updated_at,
                    claim_state, claim_owner
                ) VALUES (?, ?, ?, ?, ?, 'staged', '')
                """,
                (
                    token,
                    relative_path,
                    expires_at.isoformat(),
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
        return path

    def claimable_path(self, upload_token: str) -> Path:
        token = _validate_upload_token(upload_token)
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT relative_path, expires_at, claim_state, claim_owner
                  FROM validation_batch_material_uploads
                 WHERE upload_token = ?
                """,
                (token,),
            ).fetchone()
        if row is None:
            raise ValidationBatchUploadClaimError(
                "validation batch upload_token is missing or already claimed"
            )
        if str(row["relative_path"]) != f"staging/{token}":
            raise ValidationBatchUploadClaimError(
                "validation batch upload_token has invalid durable ownership"
            )
        if str(row["claim_state"]) != "staged" or str(row["claim_owner"]):
            raise ValidationBatchUploadClaimError(
                "validation batch upload_token is missing or already claimed"
            )
        if _parse_timestamp(str(row["expires_at"])) <= _utc(self._clock()):
            raise ValidationBatchUploadClaimError(
                "validation batch upload_token has expired"
            )
        try:
            return self._staging_path(token, require_directory=True)
        except (OSError, RuntimeError) as exc:
            raise ValidationBatchUploadClaimError(
                "validation batch upload_token is unavailable"
            ) from exc

    def reserve(
        self,
        upload_tokens: Sequence[str],
        claim_owner: str,
    ) -> dict[str, Path]:
        """Atomically reserve every token for one batch-create request."""

        return self._reserve_uploads(
            upload_tokens,
            claim_owner,
            allow_expired=False,
            require_directory=True,
        )

    def _reserve_uploads(
        self,
        upload_tokens: Sequence[str],
        claim_owner: str,
        *,
        allow_expired: bool,
        require_directory: bool,
    ) -> dict[str, Path]:
        tokens = tuple(
            dict.fromkeys(_validate_upload_token(token) for token in upload_tokens)
        )
        owner = _validate_claim_owner(claim_owner)
        if not tokens:
            return {}
        placeholders = ",".join("?" for _ in tokens)
        now = _utc(self._clock())
        paths: dict[str, Path] = {}
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                f"""
                SELECT upload_token, relative_path, expires_at,
                       claim_state, claim_owner
                  FROM validation_batch_material_uploads
                 WHERE upload_token IN ({placeholders})
                """,  # noqa: S608 - placeholders are generated from token count only.
                tokens,
            ).fetchall()
            by_token = {str(row["upload_token"]): row for row in rows}
            for token in tokens:
                row = by_token.get(token)
                if row is None:
                    raise ValidationBatchUploadClaimError(
                        "validation batch upload_token is missing or already claimed"
                    )
                if (
                    str(row["relative_path"]) != f"staging/{token}"
                    or str(row["claim_state"]) != "staged"
                    or str(row["claim_owner"])
                ):
                    raise ValidationBatchUploadClaimError(
                        "validation batch upload_token is missing or already claimed"
                    )
                if (
                    not allow_expired
                    and _parse_timestamp(str(row["expires_at"])) <= now
                ):
                    raise ValidationBatchUploadClaimError(
                        "validation batch upload_token has expired"
                    )
                try:
                    paths[token] = self._staging_path(
                        token,
                        require_directory=require_directory,
                    )
                except (OSError, RuntimeError) as exc:
                    raise ValidationBatchUploadClaimError(
                        "validation batch upload_token is unavailable"
                    ) from exc
            cursor = conn.execute(
                f"""
                UPDATE validation_batch_material_uploads
                   SET claim_state = 'reserved', claim_owner = ?, updated_at = ?
                 WHERE upload_token IN ({placeholders})
                   AND claim_state = 'staged' AND claim_owner = ''
                """,  # noqa: S608 - placeholders are generated from token count only.
                (owner, now.isoformat(), *tokens),
            )
            if cursor.rowcount != len(tokens):
                raise ValidationBatchUploadClaimError(
                    "validation batch upload_token is missing or already claimed"
                )
        return paths

    def abandon_reservation(self, upload_token: str, claim_owner: str) -> bool:
        """Delete one reservation only when the exact request still owns it."""

        token = _validate_upload_token(upload_token)
        owner = _validate_claim_owner(claim_owner)
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                """
                DELETE FROM validation_batch_material_uploads
                 WHERE upload_token = ?
                   AND claim_state = 'reserved'
                   AND claim_owner = ?
                """,
                (token, owner),
            )
        if cursor.rowcount != 1:
            return False
        return self._remove_controlled_directory(
            self._staging_path(token, require_directory=False)
        )

    def discard(self, upload_token: str) -> bool:
        token = _validate_upload_token(upload_token)
        owner = uuid.uuid4().hex
        try:
            self._reserve_uploads(
                (token,),
                owner,
                allow_expired=True,
                require_directory=False,
            )
        except ValidationBatchUploadClaimError:
            return False
        return self.abandon_reservation(token, owner)

    def forget(self, upload_token: str) -> None:
        token = _validate_upload_token(upload_token)
        with connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM validation_batch_material_uploads WHERE upload_token = ?",
                (token,),
            )

    def sweep_expired(self, *, limit: int = 100) -> ValidationBatchUploadRecoveryReport:
        if isinstance(limit, bool) or not 1 <= int(limit) <= 500:
            raise ValueError("validation batch upload sweep limit must be 1 to 500")
        now = _utc(self._clock()).isoformat()
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT upload_token
                  FROM validation_batch_material_uploads
                 WHERE claim_state = 'staged' AND expires_at <= ?
                 ORDER BY expires_at ASC, upload_token ASC
                 LIMIT ?
                """,
                (now, int(limit)),
            ).fetchall()
        return self._remove_ledger_rows(rows, reserve_staged=True)

    def reconcile_startup(self) -> ValidationBatchUploadRecoveryReport:
        """Remove every claim/orphan left by the previous server process."""

        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT upload_token
                  FROM validation_batch_material_uploads
                 ORDER BY created_at ASC, upload_token ASC
                """
            ).fetchall()
            live_source_dirs = {
                Path(str(row["source_dir"] or "")).absolute()
                for row in conn.execute("SELECT source_dir FROM tasks").fetchall()
                if str(row["source_dir"] or "")
            }
        report = self._remove_ledger_rows(rows)
        examined = report.examined
        removed = report.removed
        failed = report.failed

        try:
            batch_root, staging_root = self._roots()
        except RuntimeError:
            logger.exception("validation batch upload startup root validation failed")
            return ValidationBatchUploadRecoveryReport(
                examined=examined,
                removed=removed,
                failed=failed + 1,
            )

        # A crash can happen after mkdir/write but before ledger registration.
        for candidate in sorted(staging_root.iterdir()):
            if not _is_upload_token(candidate.name):
                continue
            examined += 1
            if self._remove_controlled_directory(candidate):
                removed += 1
            else:
                failed += 1

        # A crash can also happen after staging -> batch-root rename but before
        # the DB batch transaction commits.  Preserve any tree referenced by a
        # live task (including child source paths); every other controlled UUID
        # directory is an orphan owned by this subsystem.
        for candidate in sorted(batch_root.iterdir()):
            if candidate == staging_root or not _is_upload_token(candidate.name):
                continue
            absolute = candidate.absolute()
            if any(
                source == absolute or absolute in source.parents
                for source in live_source_dirs
            ):
                continue
            examined += 1
            if self._remove_controlled_directory(candidate):
                removed += 1
            else:
                failed += 1
        return ValidationBatchUploadRecoveryReport(examined, removed, failed)

    def _remove_ledger_rows(
        self,
        rows: Sequence[sqlite3.Row],
        *,
        reserve_staged: bool = False,
    ) -> ValidationBatchUploadRecoveryReport:
        examined = 0
        removed = 0
        failed = 0
        for row in rows:
            token = str(row["upload_token"])
            examined += 1
            if reserve_staged:
                owner = uuid.uuid4().hex
                try:
                    self._reserve_uploads(
                        (token,),
                        owner,
                        allow_expired=True,
                        require_directory=False,
                    )
                except ValidationBatchUploadClaimError:
                    failed += 1
                    continue
                if self.abandon_reservation(token, owner):
                    removed += 1
                else:
                    failed += 1
                continue
            try:
                path = self._staging_path(token, require_directory=False)
                succeeded = self._remove_controlled_directory(path)
            except (ValueError, RuntimeError):
                logger.exception(
                    "refused invalid validation batch staging ledger row %s",
                    token,
                )
                succeeded = False
            if succeeded:
                self.forget(token)
                removed += 1
            else:
                failed += 1
        return ValidationBatchUploadRecoveryReport(examined, removed, failed)

    def _roots(self) -> tuple[Path, Path]:
        uploads_root = self.material_uploads_root
        if uploads_root.is_symlink():
            raise RuntimeError("material uploads root must not be a symlink")
        uploads_root.mkdir(parents=True, exist_ok=True)
        if uploads_root.resolve() != uploads_root:
            raise RuntimeError("material uploads root resolved outside owned path")

        batch_root = uploads_root / "validation-batches"
        if batch_root.is_symlink():
            raise RuntimeError("validation batch material root must not be a symlink")
        batch_root.mkdir(parents=True, exist_ok=True)
        if batch_root.resolve() != batch_root.absolute():
            raise RuntimeError("validation batch material root resolved outside owned path")

        staging_root = batch_root / "staging"
        if staging_root.is_symlink():
            raise RuntimeError("validation batch staging root must not be a symlink")
        staging_root.mkdir(parents=True, exist_ok=True)
        if staging_root.resolve() != staging_root.absolute():
            raise RuntimeError("validation batch staging root resolved outside owned path")
        return batch_root, staging_root

    def _staging_path(self, upload_token: str, *, require_directory: bool) -> Path:
        token = _validate_upload_token(upload_token)
        _, staging_root = self._roots()
        path = staging_root / token
        if path.is_symlink():
            raise RuntimeError("validation batch staging token must not be a symlink")
        if require_directory and not path.is_dir():
            raise RuntimeError("validation batch staging token directory is missing")
        if path.exists() and path.resolve() != path.absolute():
            raise RuntimeError("validation batch staging token resolved outside owned path")
        return path

    @staticmethod
    def _remove_controlled_directory(path: Path) -> bool:
        try:
            if path.is_symlink():
                path.unlink()
            elif path.exists():
                shutil.rmtree(path)
            return True
        except OSError as exc:
            logger.warning(
                "validation batch staged material cleanup failed for %s: %s",
                path,
                exc,
            )
            return False


def consume_validation_batch_uploads_on_connection(
    conn: sqlite3.Connection,
    upload_tokens: Sequence[str],
    *,
    claim_owner: str | None = None,
    now: datetime | None = None,
) -> None:
    """Consume staging claims inside the owning batch-create transaction."""

    tokens = tuple(dict.fromkeys(_validate_upload_token(token) for token in upload_tokens))
    if not tokens:
        return
    owner = _validate_claim_owner(claim_owner)
    placeholders = ",".join("?" for _ in tokens)
    rows = conn.execute(
        f"""
        SELECT upload_token, relative_path, expires_at, claim_state, claim_owner
          FROM validation_batch_material_uploads
         WHERE upload_token IN ({placeholders})
        """,  # noqa: S608 - placeholders are generated from token count only.
        tokens,
    ).fetchall()
    by_token = {str(row["upload_token"]): row for row in rows}
    current = _utc(now or datetime.now(UTC))
    for token in tokens:
        row = by_token.get(token)
        if row is None:
            raise ValidationBatchUploadClaimError(
                "validation batch upload_token is missing or already claimed"
            )
        if str(row["relative_path"]) != f"staging/{token}":
            raise ValidationBatchUploadClaimError(
                "validation batch upload_token has invalid durable ownership"
            )
        if (
            str(row["claim_state"]) != "reserved"
            or str(row["claim_owner"]) != owner
        ):
            raise ValidationBatchUploadClaimError(
                "validation batch upload_token is missing or owned by another request"
            )
        if _parse_timestamp(str(row["expires_at"])) <= current:
            raise ValidationBatchUploadClaimError(
                "validation batch upload_token has expired"
            )
    cursor = conn.execute(
        f"""
        DELETE FROM validation_batch_material_uploads
         WHERE upload_token IN ({placeholders})
           AND claim_state = 'reserved' AND claim_owner = ?
        """,  # noqa: S608 - placeholders are generated from token count only.
        (*tokens, owner),
    )
    if cursor.rowcount != len(tokens):
        raise ValidationBatchUploadClaimError(
            "validation batch upload_token ownership changed before consume"
        )


def _validate_upload_token(upload_token: str) -> str:
    token = str(upload_token or "")
    if not _is_upload_token(token):
        raise ValueError("invalid validation batch upload_token")
    return token


def _validate_claim_owner(claim_owner: str | None) -> str:
    owner = str(claim_owner or "")
    if not _is_upload_token(owner):
        raise ValueError("invalid validation batch upload claim owner")
    return owner


def _is_upload_token(value: str) -> bool:
    return (
        len(value) == _TOKEN_LENGTH
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationBatchUploadClaimError(
            "validation batch upload_token has invalid expiry"
        ) from exc
    return _utc(parsed)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = [
    "DEFAULT_VALIDATION_BATCH_UPLOAD_TTL_SECONDS",
    "ValidationBatchUploadClaimError",
    "ValidationBatchUploadGarbageCollector",
    "ValidationBatchUploadRecoveryReport",
    "consume_validation_batch_uploads_on_connection",
]
