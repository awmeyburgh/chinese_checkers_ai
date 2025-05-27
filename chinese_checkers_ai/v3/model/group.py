from dataclasses import dataclass
from typing import Iterator, List, Set, Union

from chinese_checkers_ai.v3.model.color import Color
from chinese_checkers_ai.v3.model.player import Player
from chinese_checkers_ai.v3.model.position import Position


@dataclass(frozen=True)
class Group:
    positions: Set[Position]
    color: Color
    is_player: bool

    def rotate(self, steps: int) -> "Group":
        return Group(
            {position.rotate(steps) for position in self.positions},
            self.color,
            self.is_player
        )
    
    def translate(self, x_steps: int, y_steps: int) -> "Group":
        return Group(
            {position.translate(x_steps, y_steps) for position in self.positions},
            self.color,
            self.is_player
        )

    def scale(self, factor: float) -> "Group":
        return Group(
            {position.scale(factor) for position in self.positions},
            self.color,
            self.is_player
        )
    
    def __eq__(self, other: "Group") -> bool:
        return self.positions == other.positions

    def __contains__(self, position: Position) -> bool:
        return position in self.positions
    
    def __iter__(self) -> Iterator[Position]:
        return iter(self.positions)

    @classmethod
    def triangle(
        cls,
        color: Union[Color, Player],
        is_player: bool,
        size: int = 4
    ) -> "Group":
        if isinstance(color, Player):
            color = color.to_color()

        center = Position.new_center()
        positions = set()

        for y in range(size):
            for x in range(y+1):
                positions.add(center.translate(-y+2*x, -y))
        return cls(positions, color, is_player)
    
    @classmethod
    def start_triangle(
        cls,
        player: Player,
        is_player: bool
    ) -> "Group":
        return cls.triangle(player, is_player, size=4) \
            .rotate(3) \
            .translate(0, -8) \
            .rotate(player.value)
