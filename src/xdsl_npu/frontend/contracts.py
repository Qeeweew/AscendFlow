"""Pure runtime input contracts and symbolic tensor extent declarations."""

from dataclasses import dataclass, replace

from ..diagnostics import require


@dataclass(frozen=True)
class IntegerContract:
    minimum: int = 0
    multiple_of: int = 1

    def __post_init__(self):
        require(
            type(self.minimum) is int and 0 <= self.minimum <= (1 << 63) - 1,
            "int64 minimum must be a nonnegative int64 integer",
        )
        require(
            type(self.multiple_of) is int and 0 < self.multiple_of <= (1 << 63) - 1,
            "int64 multiple_of must be a positive int64 integer",
        )
        first = ((self.minimum + self.multiple_of - 1) // self.multiple_of) * self.multiple_of
        require(first <= (1 << 63) - 1, "int64 input contract has no representable value")


@dataclass(frozen=True)
class _Dimension:
    name: str
    multiple: int = 1
    scale: int = 1
    minimum: int = 0

    def __mul__(self, factor):
        if isinstance(factor, (_Dimension, _ExtentProduct)):
            return _extent_product(self, factor)
        require(
            type(factor) is int and factor >= 0,
            "one runtime extent times a static integer required",
        )
        return 0 if factor == 0 else replace(self, scale=self.scale * factor)

    __rmul__ = __mul__

    def __bool__(self):
        raise TypeError("runtime lengths cannot control Python branches")


@dataclass(frozen=True)
class _ExtentProduct:
    """Uniform row-major stride, with the static byte coefficient kept separate.

    This is a source address expression, not a flattened loop coordinate. Its
    factors become a shared arith.muli DAG in the program's parameter scope.
    """

    factors: tuple[_Dimension, ...]
    scale: int = 1

    def __mul__(self, factor):
        if type(factor) is int:
            return 0 if factor == 0 else replace(self, scale=self.scale * factor)
        require(isinstance(factor, (_Dimension, _ExtentProduct)), "unsupported extent product")
        return _extent_product(self, factor)

    __rmul__ = __mul__


def _extent_product(left, right):
    factors = tuple(
        replace(dimension, scale=1)
        for part in (left, right)
        for dimension in (part.factors if isinstance(part, _ExtentProduct) else (part,))
    )
    require(len(factors) <= 8, "tensor address exceeds eight dynamic stride factors")
    return _ExtentProduct(tuple(sorted(factors, key=lambda d: d.name)), left.scale * right.scale)


def dynamic(name, *, minimum=0, multiple_of=1):
    """Uniform runtime length. Local/native tile dimensions remain static."""
    require(type(name) is str and name.isidentifier(), "runtime size needs an identifier")
    contract = IntegerContract(minimum, multiple_of)
    return _Dimension(name, contract.multiple_of, minimum=contract.minimum)
