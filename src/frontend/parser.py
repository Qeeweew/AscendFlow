"""Restricted Python AST -> typed xDSL regions, without executing a kernel body.

The typed operation builders are internal to this frontend. Control flow is
owned here: host iterators and arbitrary Python calls are never invoked.
Unsupported structured lowering is diagnosed, never sent to finite expansion.
"""

from __future__ import annotations

import ast
import inspect
import operator
import textwrap
from contextlib import ExitStack
from dataclasses import dataclass, replace
from types import MappingProxyType

from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import FileLineColLoc, IndexType, IntAttr, ModuleOp, StringAttr, i1
from xdsl.ir import Block, OpResult, Region, SSAValue, SSAValues

from ..diagnostics import CompileError, require
from ..dialects.source import PipelineOp, YieldOp
from . import builders, contracts, syntax


class FrontendError(CompileError):
    """A diagnostic tied to the original DSL source, not a Python traceback."""


@dataclass(frozen=True)
class _Symbol:
    name: str


_DSL_NAMESPACE = object()


_CALLS = {
    "Tensor",
    "Kernel",
    "Stage",
    "Cube",
    "Vector",
    "alloc_ub",
    "alloc_l1",
    "alloc_l0a",
    "alloc_l0b",
    "alloc_l0c",
    "copy",
    "mul",
    "add",
    "sub",
    "maximum",
    "exp",
    "cast",
    "fill",
    "fill_upper",
    "matmul",
    "sqrt",
    "rsqrt",
    "div",
    "load_scalar",
    "reduce_sum",
    "reduce_max",
}
_SYNTAX = {
    "serial",
    "unroll",
    "tile_ranges",
    "Pipelined",
    "ceildiv",
    "min",
    "max",
    "int64",
    "constexpr",
}
_CONSTANTS = {"f32", "f16", "bf16"}
_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Div: operator.truediv,
}
_COMPARE = {
    ast.Eq: ("eq", operator.eq),
    ast.NotEq: ("ne", operator.ne),
    ast.Lt: ("slt", operator.lt),
    ast.LtE: ("sle", operator.le),
    ast.Gt: ("sgt", operator.gt),
    ast.GtE: ("sge", operator.ge),
}


def _constant(value):
    if type(value) in {int, float, str, bool, type(None)} or isinstance(value, syntax.DType):
        return value
    if type(value) is tuple:
        return tuple(_constant(item) for item in value)
    raise CompileError("captured values must be immutable scalar constants or tuples")


@dataclass(frozen=True)
class _SourceFunction:
    source: str
    filename: str
    first_line: int
    environment: object


@dataclass(frozen=True)
class InlineFunction(_SourceFunction):
    """Lexically captured DSL helper, interpreted at each call, never executed."""


@dataclass(frozen=True)
class PrimFunction(_SourceFunction):
    disjoint_external: bool = False
    specializations: tuple = ()

    def to_ir(self):
        return _Parser(self).build()

    def specialize(self, **values):
        bound = dict(self.specializations)
        bound.update((name, _constant(value)) for name, value in values.items())
        return replace(self, specializations=tuple(bound.items()))


def capture(function, *, namespace, disjoint_external=False, inline=False):
    require(type(disjoint_external) is bool, "disjoint_external must be a boolean contract")
    try:
        lines, first = inspect.getsourcelines(function)
    except (OSError, TypeError) as error:
        raise CompileError(
            "DSL source is unavailable; use language.from_source explicitly"
        ) from error
    source = textwrap.dedent("".join(lines))
    closure = inspect.getclosurevars(function)
    available = {**function.__globals__, **closure.nonlocals}
    names = {node.id for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Name)}
    environment = {}
    for name in names & available.keys():
        value = available[name]
        if value is namespace:
            environment[name] = _DSL_NAMESPACE
        elif isinstance(value, InlineFunction):
            environment[name] = value
        else:
            # Only names resolved from the host are frozen; parameters/local
            # bindings shadow them in the AST parser.
            try:
                environment[name] = _constant(value)
            except CompileError:
                # Do not retain a callable/mutable object. A reference to this
                # name will receive a source diagnostic at compilation.
                environment[name] = _Symbol("unsupported host binding: " + name)
    arguments = (
        source,
        inspect.getsourcefile(function) or "<dsl>",
        first,
        MappingProxyType(environment),
    )
    return InlineFunction(*arguments) if inline else PrimFunction(*arguments, disjoint_external)


def from_source(
    source, *, constants=None, functions=None, filename="<dsl>", disjoint_external=False
):
    """Parse explicit source; this function does not use exec/eval or import it."""
    require(type(source) is str, "DSL source must be a string")
    require(type(disjoint_external) is bool, "disjoint_external must be a boolean contract")
    environment = {"N": _DSL_NAMESPACE, "T": _DSL_NAMESPACE}
    for name, value in (constants or {}).items():
        require(name not in {"N", "T"} and name.isidentifier(), "invalid captured constant name")
        environment[name] = _constant(value)
    for name, value in (functions or {}).items():
        require(name.isidentifier() and name not in environment, "invalid captured function name")
        require(
            isinstance(value, InlineFunction), "functions must contain captured DSL inline helpers"
        )
        environment[name] = value
    return PrimFunction(
        textwrap.dedent(source), filename, 1, MappingProxyType(environment), disjoint_external
    )


class _Parser:
    def __init__(self, definition):
        self.definition = definition
        self.env = dict(definition.environment)
        self.trace = None
        self.steps = 0
        self.inline_stack = []

    def fail(self, node, message):
        line = self.definition.first_line + getattr(node, "lineno", 1) - 1
        column = getattr(node, "col_offset", 0) + 1
        raise FrontendError(f"{self.definition.filename}:{line}:{column}: {message}")

    def locate(self, node):
        self.trace.location = FileLineColLoc(
            StringAttr(self.definition.filename),
            IntAttr(self.definition.first_line + node.lineno - 1),
            IntAttr(node.col_offset + 1),
        )

    def function(self, decorator_name):
        tree = ast.parse(self.definition.source, filename=self.definition.filename)
        require(sum(1 for _ in ast.walk(tree)) <= 10000, "DSL AST exceeds 10000-node budget")
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        require(len(functions) == 1 and len(tree.body) == 1, "expected exactly one DSL function")
        func = functions[0]
        if len(func.decorator_list) > 1:
            self.fail(func, f"a DSL definition supports only its {decorator_name} decorator")
        for decorator in func.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if not (
                isinstance(target, ast.Attribute)
                and target.attr == decorator_name
                and self.expr(target.value) is _DSL_NAMESPACE
            ):
                self.fail(decorator, "unsupported decorator; it will not be executed or ignored")
            if isinstance(decorator, ast.Call):
                values, options = self.arguments(decorator)
                if decorator_name == "inline":
                    if values or options:
                        self.fail(decorator, "inline accepts no options")
                    continue
                if values or set(options) - {"disjoint_external"}:
                    self.fail(decorator, "prim_func accepts only disjoint_external")
                if "disjoint_external" in options and (
                    type(options["disjoint_external"]) is not bool
                    or options["disjoint_external"] != self.definition.disjoint_external
                ):
                    self.fail(
                        decorator, "source decorator conflicts with the explicit input contract"
                    )
        return func

    def build(self):
        func = self.function("prim_func")
        args = func.args
        if args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg or func.returns:
            self.fail(func, "use named tensor/scalar parameters; no variadic or return annotation")
        self.trace = builders._Trace(func.name, self.definition.disjoint_external)
        self.trace.op.attributes["frontend"] = StringAttr("ast")
        token = builders._current.set(self.trace)
        try:
            self.locate(func)
            self.signature(args)
            self.body(func.body)
            require(self.trace.kernel_entered, "source function must contain Kernel")
            self.trace.add(YieldOp.build(operands=[[]]))
            module = ModuleOp([self.trace.op])
            module.verify()
            return module
        finally:
            builders._current.reset(token)

    def inline(self, definition, node, values, options):
        self.steps += 1
        if self.steps > 10000:
            self.fail(node, "DSL expansion exceeds 10000-statement budget")
        if len(self.inline_stack) >= 16 or any(definition is item for item in self.inline_stack):
            self.fail(node, "inline recursion or depth exceeds the 16-call budget")
        original = self.definition, self.env, self.trace.location
        self.definition, self.env = definition, dict(definition.environment)
        self.inline_stack.append(definition)
        func = node
        try:
            func = self.function("inline")
            args = func.args
            if args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg or func.returns:
                self.fail(
                    func, "inline helpers require named parameters without return annotations"
                )
            if any(arg.annotation is not None for arg in args.args):
                self.fail(func, "inline helper parameters take DSL values without annotations")
            names = [arg.arg for arg in args.args]
            if len(values) > len(names) or set(options) - set(names):
                self.fail(func, "invalid inline arguments")
            bound = dict(zip(names, values))
            if bound.keys() & options.keys():
                self.fail(func, "inline parameter supplied more than once")
            bound.update(options)
            defaults = dict(zip(names[len(names) - len(args.defaults) :], args.defaults))
            for name in names:
                if name not in bound:
                    if name not in defaults:
                        self.fail(func, f"missing inline argument {name}")
                    bound[name] = self.expr(defaults[name])
            self.env.update(bound)
            result = func.body[-1] if isinstance(func.body[-1], ast.Return) else None
            statements = func.body[:-1] if result is not None else func.body
            if any(isinstance(item, ast.Return) for stmt in statements for item in ast.walk(stmt)):
                self.fail(func, "inline helpers support only one final return")
            self.body(statements)
            if result is None or result.value is None:
                return None
            self.locate(result)
            return self.expr(result.value)
        except FrontendError:
            raise
        except (CompileError, TypeError, ValueError, ZeroDivisionError) as error:
            self.fail(func, str(error))
        finally:
            self.inline_stack.pop()
            self.definition, self.env, self.trace.location = original

    def signature(self, args):
        defaults = dict(zip([arg.arg for arg in args.args[-len(args.defaults) :]], args.defaults))
        annotations, constexpr, runtime = {}, {}, {}
        for arg in args.args:
            annotation = arg.annotation
            if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
                annotation = ast.parse(annotation.value, mode="eval").body
            if annotation is None:
                self.fail(arg, "expected Tensor, int64, or constexpr annotation")
            annotations[arg.arg] = annotation
            target = annotation.func if isinstance(annotation, ast.Call) else annotation
            if isinstance(target, ast.Attribute) and target.attr in {"int64", "constexpr"}:
                kind = self.expr(target)
                if kind == _Symbol("int64"):
                    if arg.arg in defaults:
                        self.fail(arg, "runtime scalars cannot have specialization defaults")
                    runtime[arg.arg] = annotation
                else:
                    if isinstance(annotation, ast.Call):
                        self.fail(annotation, "constexpr annotation does not accept arguments")
                    constexpr[arg.arg] = arg
        bound = dict(self.definition.specializations)
        unknown = set(bound) - constexpr.keys()
        if unknown:
            self.fail(args, f"only constexpr parameters may specialize: {sorted(unknown)}")
        for name, arg in constexpr.items():
            if name in bound:
                self.env[name] = bound[name]
            elif name in defaults:
                self.env[name] = _constant(self.expr(defaults[name]))
            else:
                self.fail(arg, f"constexpr {name} requires an explicit value")
        # Contracts may use constexpr parameters declared later in the signature.
        # Runtime expressions are never executed as host annotation values.
        for name, annotation in runtime.items():
            options = {}
            if isinstance(annotation, ast.Call):
                values, options = self.arguments(annotation)
                if values or set(options) - {"minimum", "multiple_of"}:
                    self.fail(annotation, "int64 accepts only minimum and multiple_of keywords")
                if len(options) != len(annotation.keywords):
                    self.fail(annotation, "int64 contract keywords cannot repeat")
            try:
                self.env[name] = contracts.dynamic(name, **options)
            except CompileError as error:
                self.fail(annotation, str(error))
            self.trace.size(self.env[name])
        for arg in args.args:
            if arg.arg in constexpr or arg.arg in runtime:
                continue
            if arg.arg in defaults:
                self.fail(arg, "only constexpr parameters may have defaults")
            annotation = annotations[arg.arg]
            typ = self.expr(annotation)
            if not isinstance(typ, syntax.Tensor):
                self.fail(arg, "expected Tensor shape/dtype/access annotation")
            self.env[arg.arg] = builders._allocate(
                typ.shape,
                typ.dtype,
                "GM",
                name=arg.arg,
                access=typ.access,
                cache=typ.cache,
                write_disjoint=typ.write_disjoint,
            )

    def body(self, nodes):
        for node in nodes:
            self.steps += 1
            if self.steps > 10000:
                self.fail(node, "DSL expansion exceeds 10000-statement budget")
            self.locate(node)
            try:
                self.statement(node)
            except FrontendError:
                raise
            except (CompileError, TypeError, ValueError, ZeroDivisionError) as error:
                self.fail(node, str(error))

    def assign(self, target, value):
        if isinstance(target, ast.Name):
            self.env[target.id] = value
        elif isinstance(target, ast.Tuple) and isinstance(value, tuple):
            if len(target.elts) != len(value):
                self.fail(target, "tuple assignment arity mismatch")
            for item, element in zip(target.elts, value):
                self.assign(item, element)
        else:
            self.fail(target, "assign local names; write storage using typed operations")

    def statement(self, node):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            self.assign(node.targets[0], self.expr(node.value))
        elif isinstance(node, ast.Expr):
            self.expr(node.value)
        elif isinstance(node, ast.Pass):
            return
        elif isinstance(node, ast.With):
            with ExitStack() as scopes:
                for item in node.items:
                    scope = self.expr(item.context_expr)
                    if not isinstance(
                        scope, (builders.Kernel, builders.Stage, builders.Cube, builders.Vector)
                    ):
                        self.fail(item.context_expr, "expected Kernel/Stage/Cube/Vector scope")
                    value = scopes.enter_context(scope)
                    if item.optional_vars:
                        self.assign(item.optional_vars, value)
                self.body(node.body)
        elif isinstance(node, ast.For):
            self.loop(node)
        elif isinstance(node, ast.If):
            self.branch(node)
        elif isinstance(node, ast.Assert):
            # Interpret the captured AST, never a Python assert statement:
            # host optimization must not remove a DSL compile-time contract.
            condition = self.expr(node.test)
            if type(condition) is not bool:
                self.fail(
                    node.test,
                    "assert requires a constexpr bool; runtime assertions are unsupported",
                )
            message = None if node.msg is None else self.expr(node.msg)
            if node.msg is not None and type(message) is not str:
                self.fail(node.msg, "assert message must be a constexpr str")
            if not condition:
                detail = "" if message is None else f": {message}"
                self.fail(node, f"compile-time assertion failed{detail}")
        else:
            self.fail(node, f"unsupported device statement {type(node).__name__}")

    def loop(self, node):
        if node.orelse or not isinstance(node.iter, ast.Call):
            self.fail(node, "device for requires one index and no for/else")
        call = node.iter
        symbol = self.expr(call.func)
        if not isinstance(symbol, _Symbol) or symbol.name not in {
            "Pipelined",
            "serial",
            "unroll",
            "tile_ranges",
        }:
            self.fail(call, "use T.serial, T.Pipelined, T.unroll, or T.tile_ranges")
        if symbol.name != "tile_ranges" and not isinstance(node.target, ast.Name):
            self.fail(node.target, "device for requires one index")
        args, kwargs = self.arguments(call)
        original = dict(self.env)
        if symbol.name in {"unroll", "tile_ranges"}:
            if symbol.name == "unroll":
                if (
                    kwargs
                    or not all(type(value) is int for value in args)
                    or not 1 <= len(args) <= 3
                ):
                    self.fail(call, "unroll requires constexpr integer range bounds")
                iterations = range(*args)
                if len(iterations) > 64:
                    self.fail(call, "unroll exceeds 64-iteration budget; retain a device loop")
                targets = (node.target.id,)
            else:
                if kwargs or len(args) != 2 or not all(type(value) is int for value in args):
                    self.fail(call, "tile_ranges requires constexpr integer extent and tile width")
                extent, width = args
                if extent < 0 or width <= 0:
                    self.fail(
                        call, "tile_ranges requires nonnegative extent and positive tile width"
                    )
                if not (
                    isinstance(node.target, ast.Tuple)
                    and len(node.target.elts) == 3
                    and all(isinstance(item, ast.Name) for item in node.target.elts)
                    and len({item.id for item in node.target.elts}) == 3
                ):
                    self.fail(
                        node.target, "tile_ranges binds three distinct names: base, count, valid"
                    )
                targets = tuple(item.id for item in node.target.elts)
                whole, tail = divmod(extent, width)
                iterations = ([(0, whole, width)] if whole else []) + (
                    [(whole * width, 1, tail)] if tail else []
                )
            for value in iterations:
                self.assign(node.target, value)
                self.body(node.body)
            for name in targets:
                if name in original:
                    self.env[name] = original[name]
                else:
                    self.env.pop(name, None)
            return
        if symbol.name == "Pipelined":
            if (
                len(args) != 1
                or set(kwargs) - {"stages", "order", "issue_order"}
                or "stages" not in kwargs
            ):
                self.fail(call, "Pipelined requires one bound and explicit stages")
            limit = args[0]
            if type(limit) is int or isinstance(limit, contracts._Dimension):
                operation = builders.Pipelined(limit, **kwargs).op
            else:
                operation = PipelineOp.create(
                    operands=[self.scalar(limit)],
                    attributes={
                        "uid": StringAttr(self.trace.name("pipeline")),
                        "stages": IntAttr(kwargs["stages"]),
                        "order": StringAttr(kwargs.get("order", "source")),
                        **builders._issue_order(kwargs.get("issue_order")),
                    },
                    regions=[Region(Block(arg_types=[IndexType()]))],
                )
            self.trace.enter(operation)
            self.trace.pipelines.append(operation)
            self.env[node.target.id] = builders._coordinate(operation.body.block.args[0])
            try:
                self.body(node.body)
            finally:
                self.trace.pipelines.pop()
                self.trace.leave(operation)
        else:
            if kwargs or not 1 <= len(args) <= 3:
                self.fail(call, "serial requires one to three range bounds")
            lb, ub, step = (
                (0, args[0], 1) if len(args) == 1 else (*args, 1) if len(args) == 2 else args
            )
            if type(step) is not int or step <= 0:
                self.fail(call, "serial step must be a positive constexpr")
            assigned = {
                item.id
                for statement in node.body
                for item in ast.walk(statement)
                if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
            }
            carried = sorted((assigned & original.keys()) - {node.target.id})
            initial = [self.scalar_result(original[name], node) for name in carried]
            block = Block(arg_types=[IndexType(), *(value.type for value in initial)])
            operation = scf.ForOp(
                self.scalar(lb), self.scalar(ub), self.scalar(step), initial, block
            )
            self.trace.add(operation)
            self.trace.blocks.append(block)
            self.trace.values.append([])
            self.env.update(zip(carried, block.args[1:]))
            self.env[node.target.id] = builders._coordinate(block.args[0])
            try:
                self.body(node.body)
                results = [self.scalar_result(self.env[name], node) for name in carried]
                if tuple(value.type for value in results) != tuple(value.type for value in initial):
                    self.fail(node, "loop-carried scalar types must remain invariant")
                self.trace.add(scf.YieldOp(*results))
            finally:
                self.trace.values.pop()
                self.trace.blocks.pop()
            self.env = original
            self.env.update(zip(carried, operation.results))
            return
        # Explicit state writes are supported; scalar/tile loop results require
        # iter_args/yields, never the last Python iteration's binding.
        for name, value in original.items():
            if name != node.target.id and self.env.get(name) is not value:
                self.fail(node, f"loop-carried binding {name} needs structured iter_args support")
        self.env = original

    def branch(self, node):
        condition = self.expr(node.test)
        if type(condition) is bool:
            self.body(node.body if condition else node.orelse)
            return
        if not isinstance(condition, SSAValue) or condition.type != i1:
            self.fail(node.test, "if requires a scalar comparison")
        before = dict(self.env)
        blocks = [Block(), Block()]
        operation = scf.IfOp(condition, [], *(Region(block) for block in blocks))
        self.trace.add(operation)
        environments = []
        for block, statements in zip(blocks, (node.body, node.orelse)):
            self.env = dict(before)
            self.trace.blocks.append(block)
            self.trace.values.append([])
            try:
                self.body(statements)
                environments.append(dict(self.env))
            finally:
                self.trace.values.pop()
                self.trace.blocks.pop()
        left, right = environments
        merged = sorted(
            name
            for name in left.keys() & right.keys()
            if left[name] is not before.get(name) or right[name] is not before.get(name)
        )
        results = []
        for block, values in zip(blocks, environments):
            self.trace.blocks.append(block)
            try:
                returned = [self.scalar_result(values[name], node) for name in merged]
                self.trace.add(scf.YieldOp(*returned))
                results.append(returned)
            finally:
                self.trace.blocks.pop()
        if tuple(value.type for value in results[0]) != tuple(value.type for value in results[1]):
            self.fail(node, "if branches must produce matching scalar types")
        operation.results = SSAValues(
            OpResult(value.type, operation, i) for i, value in enumerate(results[0])
        )
        self.env = before
        self.env.update(zip(merged, operation.results))

    def scalar_result(self, value, node):
        if type(value) is bool:
            return self.trace.add(arith.ConstantOp.from_int_and_width(int(value), i1)).result
        if isinstance(value, SSAValue) and value.type == i1:
            return value
        try:
            return self.scalar(value)
        except CompileError:
            self.fail(
                node,
                "structured SSA results must be integer/boolean scalars; write buffers explicitly",
            )

    def scalar(self, value):
        return self.trace.scalar(value)

    def arguments(self, node):
        if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
            k.arg is None for k in node.keywords
        ):
            self.fail(node, "argument unpacking is not part of the DSL")
        if len({k.arg for k in node.keywords}) != len(node.keywords):
            self.fail(node, "call keywords cannot repeat")
        return [self.expr(arg) for arg in node.args], {
            k.arg: self.expr(k.value) for k in node.keywords
        }

    def expr(self, node):
        if isinstance(node, ast.BoolOp):
            # Specialization policies use Python's ordered short-circuiting.
            # Device predicates remain explicit control regions.
            for item in node.values:
                value = self.expr(item)
                if type(value) is not bool:
                    self.fail(item, "boolean expressions require constexpr booleans")
                if value == isinstance(node.op, ast.Or):
                    return value
            return value
        if isinstance(node, ast.Constant):
            return _constant(node.value)
        if isinstance(node, ast.Name):
            if node.id not in self.env:
                self.fail(node, f"undefined DSL name {node.id}")
            value = self.env[node.id]
            if isinstance(value, _Symbol) and value.name.startswith("unsupported host"):
                self.fail(node, value.name)
            return value
        if isinstance(node, ast.Attribute):
            if self.expr(node.value) is not _DSL_NAMESPACE:
                self.fail(node, "attribute access is restricted to registered DSL names")
            if node.attr in _CONSTANTS:
                return getattr(syntax, node.attr)
            if node.attr in _CALLS | _SYNTAX:
                return _Symbol(node.attr)
            self.fail(node, f"unknown DSL operation {node.attr}")
        if isinstance(node, ast.Tuple):
            return tuple(self.expr(item) for item in node.elts)
        if isinstance(node, ast.IfExp):
            condition = self.expr(node.test)
            if type(condition) is not bool:
                self.fail(
                    node,
                    "conditional expressions require a constexpr boolean; "
                    "use structured if/else for device conditions",
                )
            return self.expr(node.body if condition else node.orelse)
        if isinstance(node, ast.Dict):
            if any(key is None for key in node.keys):
                self.fail(node, "policy mappings cannot unpack host dictionaries")
            keys = [self.expr(key) for key in node.keys]
            if not all(type(key) is str for key in keys) or len(set(keys)) != len(keys):
                self.fail(node, "policy mappings need unique literal string keys")
            return dict(zip(keys, (self.expr(value) for value in node.values)))
        if isinstance(node, ast.Slice):
            return slice(
                *(
                    self.expr(item) if item is not None else None
                    for item in (node.lower, node.upper, node.step)
                )
            )
        if isinstance(node, ast.Subscript):
            value, index = self.expr(node.value), self.expr(node.slice)
            if not isinstance(value, (builders._Buffer, builders._View, builders._Value, tuple)):
                self.fail(node, "only tensor views and constexpr tuples support indexing")
            return value[index]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = self.expr(node.operand)
            return value * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            lhs, rhs = self.expr(node.left), self.expr(node.right)
            if isinstance(node.op, ast.Div):
                if type(lhs) not in {int, float} or type(rhs) not in {int, float}:
                    self.fail(node, "device integer division requires // or N.ceildiv")
                return lhs / rhs
            if type(lhs) not in {int, float} or type(rhs) not in {int, float}:
                lhs, rhs = builders._index(lhs), builders._index(rhs)
            return _BINARY[type(node.op)](lhs, rhs)
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _COMPARE:
            lhs, rhs = self.expr(node.left), self.expr(node.comparators[0])
            name, compare = _COMPARE[type(node.ops[0])]
            if type(lhs) in {int, float, bool} and type(rhs) in {int, float, bool}:
                return compare(lhs, rhs)
            if (
                type(lhs) is str
                and type(rhs) is str
                and isinstance(node.ops[0], (ast.Eq, ast.NotEq))
            ):
                return compare(lhs, rhs)
            return self.trace.index_op(arith.CmpiOp, lhs, rhs, comparison=name)
        if isinstance(node, ast.Call):
            symbol = self.expr(node.func)
            if isinstance(symbol, InlineFunction):
                args, kwargs = self.arguments(node)
                return self.inline(symbol, node, args, kwargs)
            if not isinstance(symbol, _Symbol):
                self.fail(node, "calling arbitrary Python functions is unsupported")
            args, kwargs = self.arguments(node)
            name = symbol.name
            if name in {"ceildiv", "min", "max"}:
                if len(args) != 2 or kwargs:
                    self.fail(node, f"{name} requires two scalar arguments")
                a, b = args
                if name == "ceildiv" and (type(b) is not int or b <= 0):
                    self.fail(node, "ceildiv requires a positive constexpr divisor")
                if all(type(arg) is int for arg in args):
                    return (
                        (a + b - 1) // b
                        if name == "ceildiv"
                        else min(args)
                        if name == "min"
                        else max(args)
                    )
                cls = {"ceildiv": arith.CeilDivUIOp, "min": arith.MinSIOp, "max": arith.MaxSIOp}[
                    name
                ]
                return self.trace.index_op(cls, a, b)
            if name not in _CALLS:
                self.fail(node, f"{name} is only valid in its declaration/control position")
            if name == "Tensor":
                return syntax.Tensor(*args, **kwargs)
            return getattr(builders, name)(*args, **kwargs)
        self.fail(node, f"unsupported device expression {type(node).__name__}")
