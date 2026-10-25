"""AscendFlow: an Ascend 910B pipeline compiler built on xDSL."""

from .compilation import Compilation, compile
from .runtime.torch import TorchRuntime
from .target.resources import DeviceResources

__all__ = ["Compilation", "DeviceResources", "TorchRuntime", "compile"]
