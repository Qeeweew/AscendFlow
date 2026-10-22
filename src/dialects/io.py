"""Dialect registration and verified text IO for NPU programs."""

from io import StringIO

from xdsl.context import Context
from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects.scf import Scf
from xdsl.parser import Parser
from xdsl.printer import Printer

from .physical import NPU
from .source import SOURCE


def create_context() -> Context:
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(NPU)
    ctx.load_dialect(SOURCE)
    ctx.load_dialect(Arith)
    ctx.load_dialect(Scf)
    return ctx


class _StringPrinter(Printer):
    def print_string_literal(self, string: str):
        if string.isascii() and string.isprintable() and '"' not in string and "\\" not in string:
            self.print_string('"' + string + '"')
        else:
            super().print_string_literal(string)

    def print_string(self, text: str, *, indent: int | None = None) -> None:
        if "\n" not in text:
            self._current_column += len(text)
            self.stream.write(text)
        else:
            super().print_string(text, indent=indent)


def dump(module: ModuleOp) -> str:
    output = StringIO()
    _StringPrinter(stream=output).print_op(module)
    return output.getvalue() + "\n"


def parse(text: str) -> ModuleOp:
    module = Parser(create_context(), text).parse_module()
    module.verify()
    return module
