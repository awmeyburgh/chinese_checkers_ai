from dataclasses import dataclass

from chinese_checkers_ai.v3.model.player import Player
from chinese_checkers_ai.v3.model.position import Position


@dataclass(frozen=True)
class Move:
    player: Player
    from_position: Position
    to_position: Position

    def __hash__(self):
        return hash((self.player, self.from_position, self.to_position))
    
    def __eq__(self, other):
        return self.player == other.player and self.from_position == other.from_position and self.to_position == other.to_position