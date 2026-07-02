import logging

import pytest

from marvis.logging_setup import (
    LOG_FILE_NAME,
    LOG_LEVEL_ENV,
    configure_logging,
    uvicorn_log_config,
)


@pytest.fixture(autouse=True)
def _reset_logging_state(monkeypatch):
    import marvis.logging_setup as logging_setup

    monkeypatch.setattr(logging_setup, "_configured_workspaces", set())
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    yield
    for handler in list(root_logger.handlers):
        if handler not in original_handlers:
            root_logger.removeHandler(handler)
            handler.close()
    root_logger.setLevel(original_level)


def test_configure_logging_creates_log_file_and_writes_records(tmp_path):
    log_path = configure_logging(tmp_path)

    assert log_path == tmp_path / "logs" / LOG_FILE_NAME
    assert log_path.parent.is_dir()

    marker_logger = logging.getLogger("marvis.test_logging_setup_marker")
    marker_logger.info("hello from test")
    for handler in logging.getLogger().handlers:
        handler.flush()

    assert log_path.exists()
    assert "hello from test" in log_path.read_text(encoding="utf-8")


def test_configure_logging_is_idempotent_per_workspace(tmp_path):
    root_logger = logging.getLogger()
    handlers_before = len(root_logger.handlers)
    configure_logging(tmp_path)
    handlers_after_first = len(root_logger.handlers)
    configure_logging(tmp_path)
    handlers_after_second = len(root_logger.handlers)

    assert handlers_after_first == handlers_before + 1
    assert handlers_after_second == handlers_after_first


def test_configure_logging_honors_log_level_env(tmp_path, monkeypatch):
    monkeypatch.setenv(LOG_LEVEL_ENV, "WARNING")

    configure_logging(tmp_path)

    assert logging.getLogger().level == logging.WARNING
    assert logging.getLogger("marvis").level == logging.WARNING


def test_uvicorn_log_config_points_at_workspace_log_file(tmp_path):
    config = uvicorn_log_config(tmp_path)

    assert config["handlers"]["file"]["filename"] == str((tmp_path / "logs" / LOG_FILE_NAME).resolve())
    assert config["handlers"]["file"]["maxBytes"] == 10 * 1024 * 1024
    assert config["handlers"]["file"]["backupCount"] == 3
    assert "uvicorn.access" in config["loggers"]
