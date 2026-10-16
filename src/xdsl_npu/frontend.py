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

from . import language as N
from .model import CompileError, require
from .source_ir import PipelineOp, YieldOp


class FrontendError(CompileError):
    """A diagnostic tied to the original DSL source, not a Python traceback."""


@dataclass(frozen=True)
class _Symbol:
    name: str


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
    "matmul",
    "sqrt",
    "rsqrt",
    "div",
    "load_scalar",
    "reduce_sum",
    "reduce_max",
}
_SYNTAX = {"serial", "unroll", "Pipelined", "ceildiv", "min", "max", "int64", "constexpr"}
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
    if type(value) in {int, float, str, bool, type(None)} or isinstance(value, N.DType):
        return value
    if type(value) is tuple:
        return tuple(_constant(item) for item in value)
    raise CompileError("captured values must be immutable scalar constants or tuples")


@dataclass(frozen=True)
class PrimFunction:
    source: str
    filename: str
    first_line: int
    environment: object
    disjoint_external: bool = False
    specializations: tuple = ()

    def to_ir(self):
        return _Parser(self).build()

    def specialize(self, **values):
        bound = dict(self.specializations)
        bound.update((name, _constant(value)) for name, value in values.items())
        return replace(self, specializations=tuple(bound.items()))


def capture(function, *, disjoint_external=False):
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
        if value is N:
            environment[name] = N
        else:
            # Only names resolved from the host are frozen; parameters/local
            # bindings shadow them in the AST parser.
            try:
                environment[name] = _constant(value)
            except CompileError:
                # Do not retain a callable/mutable object. A reference to this
                # name will receive a source diagnostic at compilation.
                environment[name] = _Symbol("unsupported host binding: " + name)
    return PrimFunction(
        source,
        inspect.getsourcefile(function) or "<dsl>",
        first,
        MappingProxyType(environment),
        disjoint_external,
    )


def from_source(source, *, constants=None, filename="<dsl>", disjoint_external=False):
    """Parse explicit source; this function does not use exec/eval or import it."""
    require(type(source) is str, "DSL source must be a string")
    require(type(disjoint_external) is bool, "disjoint_external must be a boolean contract")
    environment = {"N": N, "T": N}
    for name, value in (constants or {}).items():
        require(name not in {"N", "T"} and name.isidentifier(), "invalid captured constant name")
        environment[name] = _constant(value)
    return PrimFunction(
        textwrap.dedent(source), filename, 1, MappingProxyType(environment), disjoint_external
    )


class _Parser:
    def __init__(self, definition):
        self.definition = definition
        self.env = dict(definition.environment)
        self.trace = None
        self.steps = 0

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

    def build(self):
        tree = ast.parse(self.definition.source, filename=self.definition.filename)
        require(sum(1 for _ in ast.walk(tree)) <= 10000, "DSL AST exceeds 10000-node budget")
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        require(len(functions) == 1 and len(tree.body) == 1, "expected exactly one DSL function")
        func = functions[0]
        if len(func.decorator_list) > 1:
            self.fail(func, "a DSL definition supports only its prim_func decorator")
        for decorator in func.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if not (
                isinstance(target, ast.Attribute)
                and target.attr == "prim_func"
                and self.expr(target.value) is N
            ):
                self.fail(decorator, "unsupported decorator; it will not be executed or ignored")
            if isinstance(decorator, ast.Call):
                values, options = self.arguments(decorator)
                if values or set(options) - {"disjoint_external"}:
                    self.fail(decorator, "prim_func accepts only disjoint_external")
                if "disjoint_external" in options and (
                    type(options["disjoint_external"]) is not bool
                    or options["disjoint_external"] != self.definition.disjoint_external
                ):
                    self.fail(
                        decorator, "source decorator conflicts with the explicit input contract"
                    )
        args = func.args
        if args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg or func.returns:
            self.fail(func, "use named tensor/scalar parameters; no variadic or return annotation")
        self.trace = N._Trace(func.name, self.definition.disjoint_external)
        self.trace.op.attributes["frontend"] = StringAttr("ast")
        token = N._current.set(self.trace)
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
            N._current.reset(token)

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
                self.env[name] = N.dynamic(name, **options)
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
            if not isinstance(typ, N.Tensor):
                self.fail(arg, "expected Tensor shape/dtype/access annotation")
            self.env[arg.arg] = N._allocate(
                typ.shape, typ.dtype, "GM", name=arg.arg, access=typ.access
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
                    if not isinstance(scope, (N.Kernel, N.Stage, N.Cube, N.Vector)):
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
        if (
            node.orelse
            or not isinstance(node.target, ast.Name)
            or not isinstance(node.iter, ast.Call)
        ):
            self.fail(node, "device for requires one index and no for/else")
        call = node.iter
        symbol = self.expr(call.func)
        if not isinstance(symbol, _Symbol) or symbol.name not in {"Pipelined", "serial", "unroll"}:
            self.fail(call, "use N.serial, N.Pipelined, or explicit N.unroll")
        args, kwargs = self.arguments(call)
        original = dict(self.env)
        if symbol.name == "unroll":
            if kwargs or not all(type(value) is int for value in args) or not 1 <= len(args) <= 3:
                self.fail(call, "unroll requires constexpr integer range bounds")
            iterations = range(*args)
            if len(iterations) > 64:
                self.fail(call, "unroll exceeds 64-iteration budget; retain a device loop")
            for value in iterations:
                self.env[node.target.id] = value
                self.body(node.body)
            if node.target.id in original:
                self.env[node.target.id] = original[node.target.id]
            else:
                self.env.pop(node.target.id, None)
            return
        if symbol.name == "Pipelined":
            if (
                len(args) != 1
                or set(kwargs) - {"stages", "order", "issue_order"}
                or "stages" not in kwargs
            ):
                self.fail(call, "Pipelined requires one bound and explicit stages")
            limit = args[0]
            if type(limit) is int or isinstance(limit, N._Dimension):
                operation = N.Pipelined(limit, **kwargs).op
            else:
                operation = PipelineOp.create(
                    operands=[self.scalar(limit)],
                    attributes={
                        "uid": StringAttr(self.trace.name("pipeline")),
                        "stages": IntAttr(kwargs["stages"]),
                        "order": StringAttr(kwargs.get("order", "source")),
                        **N._issue_order(kwargs.get("issue_order")),
                    },
                    regions=[Region(Block(arg_types=[IndexType()]))],
                )
            self.trace.enter(operation)
            self.trace.pipelines.append(operation)
            self.env[node.target.id] = N._coordinate(operation.body.block.args[0])
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
            self.env[node.target.id] = N._coordinate(block.args[0])
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
        return [self.expr(arg) for arg in node.args], {
            k.arg: self.expr(k.value) for k in node.keywords
        }

    def expr(self, node):
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
            if self.expr(node.value) is not N:
                self.fail(node, "attribute access is restricted to registered DSL names")
            if node.attr in _CONSTANTS:
                return getattr(N, node.attr)
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
            if not isinstance(value, (N._Buffer, N._View, N._Value, tuple)):
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
                lhs, rhs = N._index(lhs), N._index(rhs)
            return _BINARY[type(node.op)](lhs, rhs)
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _COMPARE:
            lhs, rhs = self.expr(node.left), self.expr(node.comparators[0])
            name, compare = _COMPARE[type(node.ops[0])]
            if type(lhs) in {int, float, bool} and type(rhs) in {int, float, bool}:
                return compare(lhs, rhs)
            return self.trace.index_op(arith.CmpiOp, lhs, rhs, comparison=name)
        if isinstance(node, ast.Call):
            symbol = self.expr(node.func)
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
            return getattr(N, name)(*args, **kwargs)
        self.fail(node, f"unsupported device expression {type(node).__name__}")
