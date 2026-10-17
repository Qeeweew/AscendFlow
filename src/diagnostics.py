"""Compiler diagnostics independent of IR extraction and execution models."""

from enum import StrEnum


class ErrorKind(StrEnum):
    INVALID = "invalid-input"
    UNSUPPORTED = "unsupported"
    RESOURCE = "resource-exhausted"
    ANALYSIS_LIMIT = "analysis-limit"


class CompileError(ValueError):
    """An actionable failure, with optional source and physical witnesses.

    A missing proof due to an unsupported representation is not a proof that
    the program is unsafe. Callers must select the corresponding category.
    """

    def __init__(self, message, *, kind=ErrorKind.INVALID, phase=None, location=None, witness=()):
        self.message = message
        self.kind = ErrorKind(kind)
        self.phase = phase
        self.location = location
        self.witness = tuple(witness)
        context = [value for value in (phase, location) if value is not None]
        suffix = "" if not context else " [" + ", ".join(map(str, context)) + "]"
        if self.witness:
            suffix += ": " + "; ".join(map(str, self.witness))
        super().__init__(message + suffix)


def require(condition, message, **context):
    if not condition:
        raise CompileError(message, **context)
