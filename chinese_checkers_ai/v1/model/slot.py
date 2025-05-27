from dataclasses import dataclass
from typing import Optional, Tuple

from chinese_checkers_ai.v1.model.color import Color


@dataclass
class Slot:
    position: Tuple[int, int]
    enabled: bool
    floor_color: Optional[Color]
    peice_color: Optional[Color]