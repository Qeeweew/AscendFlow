"""Physical memory hazard kinds."""

from enum import Enum


class HazardKind(str, Enum):
    RAW = "RAW"
    WAR = "WAR"
    WAW = "WAW"
