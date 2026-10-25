"""Host device facts bound before source construction and dependency analysis."""

from dataclasses import dataclass

from ..diagnostics import require


@dataclass(frozen=True)
class DeviceResources:
    cube_cores: int
    vector_cores: int

    def __post_init__(self):
        require(
            all(type(n) is int and 1 <= n <= 1024 for n in (self.cube_cores, self.vector_cores)),
            "device core counts must be positive integers in [1,1024]",
        )

    def core_count(self, kind):
        require(kind in {"cube", "vector", "mixed"}, "core kind must be cube, vector, or mixed")
        return self.vector_cores if kind == "vector" else self.cube_cores
