"""Restricted source capture and AST compilation into typed source IR."""

from .parser import FrontendError, InlineFunction, PrimFunction, from_source

__all__ = ["FrontendError", "InlineFunction", "PrimFunction", "from_source"]
