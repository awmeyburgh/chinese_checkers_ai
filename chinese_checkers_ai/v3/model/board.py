from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Union

from chinese_checkers_ai.v3.model.color import Color
from chinese_checkers_ai.v3.model.group import Group
from chinese_checkers_ai.v3.model.move import Move
from chinese_checkers_ai.v3.model.player import Player
from chinese_checkers_ai.v3.model.position import Position
import torch
import random as rand


@dataclass(frozen=True)
class Board:
    players: Dict[Player, Group]
    rotation: int

    # transformation

    def rotate(self, steps: int) -> "Board":
        return Board(
            {player: group.rotate(steps) for player, group in self.players.items()},
            (self.rotation + steps) % 6
        )
    
    def focus(self, player: Player) -> "Board":
        return self.rotate(6 - (player.value + self.rotation))

    def floor(self) -> List[Group]:
        return self.rotated_floor(self.rotation)
    
    @property
    def focused(self) -> Player:
        return Player((6-self.rotation)%6)
    
    # position check
    
    def is_player_position(self, position: Position) -> bool:
        return any(position in group for group in self.players.values())
    
    def is_floor_position(self, position: Position) -> bool:
        return any(position in group for group in self.floor())
    
    # move calculation 

    def side_positions(self, position: Position) -> List[Position]:
        return [
            sibling for sibling in position.siblings()
                if not self.is_player_position(sibling) \
                    and self.is_floor_position(sibling)
        ]
    
    def jump_positions(self, position: Position) -> List[Position]:
        side_positions = position.siblings()
        jump_positions = position.siblings(2)

        result = []

        for i in range(6):
            side_position = side_positions[i]
            jump_position = jump_positions[i]
            
            if self.is_player_position(side_position):
                if self.is_floor_position(jump_position) and not self.is_player_position(jump_position):
                    result.append(jump_position)

        return result
        
    def moves(self, player: Player, position: Position, has_jumped=False, explored=None) -> List[Move]:
        result = []

        if not has_jumped:
            result.extend([
                Move(player, position, side_position)
                for side_position in self.side_positions(position)
            ])

        if explored is None:
            explored = {position}

        for jump_position in self.jump_positions(position):
            if jump_position not in explored:
                result.append(Move(player, position, jump_position))
                explored.add(jump_position)
                jump_moves = self.moves(player, jump_position, has_jumped=True, explored=explored)
                result.extend([Move(player, position, jump_move.to_position) for jump_move in jump_moves])
                explored.update(jump_moves)

        return result

    def all_moves(self, player: Player) -> List[Move]:
        result = []
        for position in self.players[player]:
            result.extend(self.moves(player, position))
        return result
    
    # move execution

    def validate_move(self, move: Move) -> bool:
        if self.is_player_position(move.to_position):
            raise ValueError("Invalid move, cannot move to a player position")
        if not self.is_floor_position(move.to_position):
            raise ValueError("Invalid move, out of bounds")
        return True

    def move(self, move: Move, validate=True) -> "Board":
        if validate and not self.validate_move(move):
            raise ValueError("Invalid move")
        
        players = {}
        for player, group in self.players.items():
            if player == move.player:
                new_positions = group.positions.copy()
                new_positions.remove(move.from_position)
                new_positions.add(move.to_position)
                players[player] = Group(new_positions, group.color, group.is_player)
            else:
                players[player] = Group(group.positions.copy(), group.color, group.is_player)
        
        return Board(players, self.rotation)
    
    def has_won(self, player: Player) -> bool:
        winning_group = self.rotated_winning_group(self.rotation, player)
        return winning_group == self.players[player]

    def goal(self, player: Player) -> Group:
        return self.rotated_winning_group(self.rotation, player)

    # static methods

    def __eq__(self, other: "Board") -> bool:
        if self.rotation != other.rotation:
            return False
        for player in self.players:
            if self.players[player] != other.players[player]:
                return False
        return True

    @classmethod
    def to_players(cls, players: Optional[Union[List[Union[int, Color, Player]], int]] = None) -> List[Player]:
        if players is None:
            return list(map(Player, range(6)))
        if isinstance(players, int):
            return [Player(i//2 if i % 2 == 0 else i//2+3) for i in range(players)]
        if isinstance(players[0], int):
            return list(map(Player, players))
        if isinstance(players[0], Color):
            return [color.to_player() for color in players]
        return players
    
    @classmethod
    def new(cls, players: Optional[Union[List[Union[int, Color, Player]], int]] = None, random = False) -> "Board":
        players = cls.to_players(players)
        if not random:
            players = {
                player: Group.start_triangle(player, is_player=True)
                for player in players
            }
        else:
            positions = {position for group in cls.rotated_floor(0) for position in group.positions}
            _players = players
            players = {}
            for player in _players:
                samples = set()
                for _ in range(10):
                    sample = rand.choice(list(positions))
                    samples.add(sample)
                    positions.remove(sample)
                players[player] = Group(samples, player.to_color(), True)
        return cls(players, 0)
    
    __ROTATED_FLOORS = {}

    @classmethod
    def rotated_floor(cls, steps: int) -> List[Group]:
        steps = (steps + 36) % 6
        if steps not in cls.__ROTATED_FLOORS:
            if steps > 0:
                cls.__ROTATED_FLOORS[steps] = [
                    group.rotate(steps)
                    for group in cls.rotated_floor(0)
                ]
            else:
                middle = set()
                for i in range(6):
                    triangle = Group.triangle(
                            Color(0),
                            False,
                            5
                        ) \
                        .rotate(i)
                    middle.update(triangle.positions)
                middle = Group(middle, color=Color.NONE, is_player=False)

                floor = [middle]
                for i in range(6):
                    floor.append(
                        Group.start_triangle(
                            Player(i),
                            False
                        )
                    )
                cls.__ROTATED_FLOORS[steps] = floor

        return cls.__ROTATED_FLOORS[steps]

    __ROTATED_WINNING_GROUPS = {}
    __POSITION_TO_IDX = None

    @classmethod
    def get_position_to_idx(cls) -> Dict[Position, int]:
        if cls.__POSITION_TO_IDX is None:
            # Get floor positions from unrotated board (rotation 0)
            floor_positions = sorted(
                {pos for group in cls.rotated_floor(0) for pos in group.positions},
                key=lambda pos: pos.key
            )
            cls.__POSITION_TO_IDX = {pos: idx for idx, pos in enumerate(floor_positions)}
        return cls.__POSITION_TO_IDX

    @classmethod
    def rotated_winning_group(cls, rotation: int, player: Player) -> Group:
        if (rotation, player) not in cls.__ROTATED_WINNING_GROUPS:
            cls.__ROTATED_WINNING_GROUPS[(rotation, player)] = Group.start_triangle(player, is_player=False)\
                .rotate(rotation+3)
        return cls.__ROTATED_WINNING_GROUPS[(rotation, player)]

    def to_tensor(self, player: Player, moves:Optional[List[Move]]=None) -> torch.Tensor:
        """Convert board state to tensor representation from perspective of current_player.
        
        Returns a tensor with shape (C, N) where:
        - C is number of channels (8 channels):
            - Channel 0: Current player's pieces
            - Channel 1: Opposite player's pieces (if exists)
            - Channel 2-6: Other players' pieces (clockwise from current)
            - Channel 7: Valid moves for current player's pieces
        - N is the number of valid positions on the board (based on floor positions)
        
        Args:
            player: The player whose perspective to use
        """
        if self.focused != player:
            focused_board = self.focus(player)
        else:
            focused_board = self
        
        # Get the cached position mapping
        pos_to_idx = self.get_position_to_idx()
        num_positions = len(pos_to_idx)
        
        # Create empty tensor
        board_tensor = torch.zeros((8, num_positions), dtype=torch.float32)
        
        # Fill channel 0 with current player's pieces
        for pos in focused_board.players[player].positions:
            if pos in pos_to_idx:
                board_tensor[0, pos_to_idx[pos]] = 1.0
            
        # Calculate opposite player (value + 3)
        opposite_player = Player((player.value + 3) % 6)
        
        # Fill channel 1 with opposite player if they exist in the game
        if opposite_player in focused_board.players:
            for pos in focused_board.players[opposite_player].positions:
                if pos in pos_to_idx:
                    board_tensor[1, pos_to_idx[pos]] = 1.0
        
        # Fill channels 2-6 with other players' pieces (excluding current and opposite)
        channel = 2
        for p in focused_board.players:
            if p != player and p != opposite_player and channel < 7:
                for pos in focused_board.players[p].positions:
                    if pos in pos_to_idx:
                        board_tensor[channel, pos_to_idx[pos]] = 1.0
                channel += 1
                    
        # Fill channel 7 with valid moves
        moves = focused_board.all_moves(player) if moves is None else moves
        for move in moves:
            if move.to_position in pos_to_idx:
                board_tensor[7, pos_to_idx[move.to_position]] = 1.0
            
        return board_tensor