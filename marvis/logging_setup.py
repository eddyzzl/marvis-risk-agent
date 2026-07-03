from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path


# GAP-10: the backend had no logging.basicConfig/FileHandler/dictConfig
# anywhere, and uvicorn.run was called with no log_config -- close the
# terminal that launched `marvis serve` and every log line is gone. For a
# single-machine product delivered to non-developers, an on-disk log is
# practically the only remote troubleshooting channel.
LOG_LEVEL_ENV = "MARVIS_LOG_LEVEL"
DEFAULT_ROOT_LEVEL = "INFO"
DEFAULT_MARVIS_LEVEL = "DEBUG"
LOG_FILE_NAME = "marvis.log"
# 10MB x 3 backups, per the review's sizing (docs/reviews/... GAP-10 fix item 1).
MAX_BYTES = 10 * 1024 * 1024
BACKUP_COUNT = 3

_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"

_configured_workspaces: set[str] = set()


def configure_logging(workspace: str | Path) -> Path:
    """Idempotent logging setup: console + rotating file handler under
    workspace/logs/marvis.log.

    Root logger is set to MARVIS_LOG_LEVEL (default INFO); the marvis.*
    hierarchy is set to DEBUG by default so module loggers created with
    logging.getLogger(__name__) are verbose on disk without needing every
    third-party library at DEBUG too. Safe to call more than once per
    workspace (e.g. from tests or multiple entry points) -- later calls are a
    no-op for a workspace whose handlers are already attached.
    """
    workspace_path = Path(workspace).resolve()
    key = str(workspace_path)
    if key in _configured_workspaces:
        return workspace_path / "logs" / LOG_FILE_NAME

    logs_dir = workspace_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / LOG_FILE_NAME

    root_level_name = os.environ.get(LOG_LEVEL_ENV, DEFAULT_ROOT_LEVEL).strip().upper()
    root_level = logging.getLevelName(root_level_name)
    if not isinstance(root_level, int):
        root_level = logging.INFO

    formatter = logging.Formatter(_LOG_FORMAT)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(root_level)
    root_logger.addHandler(file_handler)

    # marvis.* stays verbose on disk (DEBUG) independent of the root/console
    # level, unless the operator explicitly overrides via MARVIS_LOG_LEVEL --
    # in that case honor a single explicit knob rather than two.
    marvis_logger = logging.getLogger("marvis")
    marvis_logger.setLevel(
        root_level if LOG_LEVEL_ENV in os.environ else logging.getLevelName(DEFAULT_MARVIS_LEVEL)
    )

    _configured_workspaces.add(key)
    return log_path


def uvicorn_log_config(workspace: str | Path) -> dict:
    """A log_config for uvicorn.run() so its access/error logs land in the
    same rotating file instead of only the launching terminal's stdout."""
    log_path = str(Path(workspace).resolve() / "logs" / LOG_FILE_NAME)
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {"format": _LOG_FORMAT},
        },
        "handlers": {
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": log_path,
                "maxBytes": MAX_BYTES,
                "backupCount": BACKUP_COUNT,
                "encoding": "utf-8",
                "formatter": "default",
            },
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["console", "file"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"handlers": ["console", "file"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": ["console", "file"], "level": "INFO", "propagate": False},
        },
    }


__all__ = ["configure_logging", "uvicorn_log_config", "LOG_FILE_NAME", "LOG_LEVEL_ENV"]
