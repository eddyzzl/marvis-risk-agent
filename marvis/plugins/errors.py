from __future__ import annotations


class PluginError(Exception):
    """Base class for plugin runtime errors."""


class ManifestError(PluginError):
    """Plugin manifest is missing required fields or has invalid values."""


class SchemaValidationError(PluginError):
    """Tool inputs or output do not match the declared JSON Schema."""

    def __init__(self, label: str, detail: str):
        super().__init__(f"{label} schema validation failed: {detail}")
        self.label = label
        self.detail = detail


class PluginNotFoundError(PluginError):
    """Requested plugin is not registered."""


class ToolNotFoundError(PluginError):
    """Requested tool is not registered."""


class DuplicatePluginError(PluginError):
    """A plugin with the same identity already exists."""


class ToolExecutionError(PluginError):
    """Tool function raised in the worker process."""

    def __init__(self, message: str, traceback_text: str):
        super().__init__(message)
        self.traceback_text = traceback_text


class ToolTimeoutError(PluginError):
    """Tool execution exceeded its timeout."""
