"""Shared exception types for Chronicle."""


class SessionValidationError(RuntimeError):
    """Raised when a session input bundle is invalid."""


class StageExecutionError(RuntimeError):
    """Raised when a pipeline stage cannot execute successfully."""

