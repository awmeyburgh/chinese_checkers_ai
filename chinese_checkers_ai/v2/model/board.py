from dataclasses import dataclass
from turtle import pos
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import torch

from chinese_checkers_ai.v2.model.color import Color

@dataclass
class BoardSnapshot:
    state: torch.Tensor
    rotation: int = 0

    def __init__(self, state: torch.Tensor, rotation: int = 0):
        self.state = state
        self.rotation = rotation
        self.__floor = None
        self.__pieces = None

    def __convert(self, tensor: torch.Tensor) -> List[Tuple[torch.Tensor, Optional[Color]]]:
        result = []

        for i in range(tensor.size(0)):
            if tensor[i, 2] == 0:
                color = None
            else:
                color = Color(int(tensor[i,2]-1))

            result.append((tensor[i, :2], color))

        return result

    @property
    def floor(self) -> List[Tuple[torch.Tensor, Optional[Color]]]:
        if self.__floor is None:
            board = Board.board()
            board[:, :2] @= Board.rotate_matrix(self.rotation)
            self.__floor = self.__convert(board)
        return self.__floor

    @property
    def pieces(self) -> List[Tuple[torch.Tensor, Optional[Color]]]:
        if self.__pieces is None:
            self.__pieces = self.__convert(self.state)
        return self.__pieces

    __top_left_center_offset = None
    __dimensions = None
    
    @classmethod
    def top_left_center_offset(cls) -> torch.Tensor:
        if cls.__top_left_center_offset is None:
            offset = torch.zeros(2)
            board = Board.board()

            for i in range(board.size(0)):
                offset[0] = min(offset[0], board[i, 0])
                offset[1] = min(offset[1], board[i, 1])

            offset *= -1
            offset += torch.tensor((1, np.sin(np.pi/3)))

            cls.__top_left_center_offset = offset
        return cls.__top_left_center_offset
    
    @classmethod
    def dimensions(cls) -> torch.Tensor:
        if cls.__dimensions is None:
            bottom_right_offset = torch.zeros(2)
            board = Board.board()

            for i in range(board.size(0)):
                bottom_right_offset[0] = max(bottom_right_offset[0], board[i, 0])
                bottom_right_offset[1] = max(bottom_right_offset[1], board[i, 1])

            bottom_right_offset += torch.tensor((1, np.sin(np.pi/3)))

            top_left_center_offset = cls.top_left_center_offset()
            cls.__dimensions = bottom_right_offset + top_left_center_offset

        return cls.__dimensions

@dataclass
class Board:
    state: torch.Tensor

    def __init__(self, state: torch.Tensor):
        self.state = state
        self.__players = None
        self.__playing = None
        self.__snapshot = None
        self.__winner = None
        self.__players_keys_lookup = None
        self.__playing_keys = None
        self.__players_winning_keys = None

    @property
    def players(self) -> List[Tuple[int, int]]:
        if self.__players is None:
            red_player = None
            players = []
            for i in range(6):
                if not torch.all(self.state[i, :10, :] == 0):
                    players.append(i)
                if self.state[i, 10, 1] == 1:
                    red_player = i
            self.__players = [(player, (player-red_player+6)%6) for player in players]

        return self.__players
    
    @property
    def playing(self) -> Tuple[int, int]:
        if self.__playing is None:
            for i, player in self.players:
                if self.state[i, 10, 0] == 1:
                    self.__playing = (i, player)
        return self.__playing
    
    def __change(self):
        self.__snapshot = None
        self.__playing = None
        self.__playing_keys = None

    def rotate(self, steps=1, player:Optional[Union[Color,int]]=None) -> int:
        if player is not None:
            if isinstance(player, Color):
                player = player.value

            rotation = self.rotation
            steps = 6 - (player + rotation)

        steps = (steps+6)%6

        if steps > 0:
            rotation_matrix = self.rotate_matrix(steps)

            for i, player in self.players:
                self.state[i, :10, :] @= rotation_matrix

            self.state = torch.concat([self.state[6-steps:6], self.state[0:6-steps]])

            self.__players = None
            self.__players_keys_lookup = None
            self.__players_winning_keys = None
            self.__change()
        return steps

    def snapshot(self) -> BoardSnapshot:
        if self.__snapshot is None:
            state = torch.zeros(10*len(self.players), 3) # each point (x, y, piece)

            for j, (i, player) in enumerate(self.players):
                colors = torch.tensor([((player+1),) for _ in range(10)], dtype=torch.get_default_dtype())
                state[10*j:10*j+10] = torch.concat([self.state[i, :10, :], colors], dim=1)

            rotation = self.rotation

            self.__snapshot = BoardSnapshot(state, rotation)

        return self.__snapshot
    
    @property
    def winner(self) -> Optional[Color]:
        return self.__winner
    
    @winner.setter
    def winner(self, value):
        self.__winner = value
    
    def move(self, from_position: torch.Tensor, to_position: torch.Tensor, check=True, rotate=False) -> Optional[Color]:
        # performs move if valid and returns if 

        from_key = self.to_key(from_position)
        to_key = self.to_key(to_position)

        if check:
            if self.winner is not None:
                raise Exception(f'invalid move, game already complete')
            
            if to_key in self.players_keys:
                raise Exception(f"invalid move to {from_key} {to_key} (Occupied)")
            
            if to_key not in self.board_keys():
                raise Exception(f"invalid move from {from_key} {to_key}")
        
            if from_key not in self.playing_keys:
                raise Exception(f"Invalid player moving")
        
        (i, j) = self.players_keys_lookup[from_key]
        self.state[i, j, :] = to_position
        del self.players_keys_lookup[from_key]
        self.players_keys_lookup[to_key] = (i,j)

        if hasattr(self, '__playing_keys'):
            delattr(self, '__playing_keys')

        if self.winning_keys(self.playing[1]) == self.playing_keys:
            self.winner = Color(self.playing[1])

        self.state[self.next_player()[0], 10, 0] = 1
        self.state[i, 10, 0] = 0
        
        self.__change()

        if rotate:
            self.rotate(player = self.playing[1])

        return self.winner

    def move_new(self, from_position: torch.Tensor, to_position: torch.Tensor, check=True, rotate=False) -> "Board":
        board = Board(self.state.clone())
        board.move(from_position, to_position, check=check, rotate=rotate)
        return board
    
    @property
    def players_keys_lookup(self) -> Dict[Tuple[int, int], Tuple[int, int]]:
        if self.__players_keys_lookup is None:
            players_keys_lookup = {}
            for i, _ in self.players:
                keys = self.to_key(self.state[i, :10, :])
                for j in range(10):
                    players_keys_lookup[keys[j]] = (i, j)
            self.__players_keys_lookup = players_keys_lookup
        return self.__players_keys_lookup
    
    def player_keys(self, player) -> Set[Tuple[int, int]]:
        i = (6-(player+self.rotation))%6
        return set(self.to_key(self.state[i, :10, :]))
    
    @property
    def players_keys(self) -> Set[Tuple[int, int]]:
        return set(self.players_keys_lookup.keys())
    
    @property
    def playing_keys(self) -> Set[Tuple[int, int]]:
        if self.__playing_keys is None:
            playing_keys = set(self.to_key(self.state[self.playing[0], :10, :]))
            self.__playing_keys = playing_keys
        return self.__playing_keys
    
    def next_player(self) -> Tuple[int, int]:
        for _i in range(1, 6):
            i = (self.playing[0] + _i)%6
            if not torch.all(self.state[i, :10, :] == 0):
                return (i, None)
            
    @property
    def rotation(self):
        return torch.max(self.state[:, 10, 1], dim=0).indices

    def winning_keys(self, player) -> Set[Tuple[int, int]]:
        if self.__players_winning_keys is None:
            rotation = self.rotation

            winning_keys = []

            for player in range(6):
                triangle = self.start_traingle(player)
                triangle @= self.rotate_matrix(3 + rotation)
                
                winning_keys.append(set(self.to_key(triangle)))
            
            self.__players_winning_keys = winning_keys

        return self.__players_winning_keys[player]

    def __next_positions(self, position: torch.Tensor, step_size = 1) -> List[torch.Tensor]:
        delta = torch.tensor((2*step_size, 0), dtype=torch.get_default_dtype())
        step_matrix = self.rotate_matrix(1)

        result = []
        for i in range(6):
            result.append(position + delta)
            delta @= step_matrix

        return result

    def __step_moves(self, position: torch.Tensor) -> List[torch.Tensor]:
        result = []

        for to_position in self.__next_positions(position, step_size=1):
            key = self.to_key(to_position)
            if key in self.board_keys() and key not in self.players_keys:
                result.append(to_position)

        return result
    def __jump_moves(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        result = []

        step_positions = self.__next_positions(position, step_size=1)
        jump_positions = self.__next_positions(position, step_size=2)

        for i in range(len(step_positions)):
            step_key = self.to_key(step_positions[i])
            jump_key = self.to_key(jump_positions[i])

            if step_key in self.players_keys and jump_key in self.board_keys() and jump_key not in self.players_keys:
                result.append(jump_positions[i])


        return result
    
    def moves(self, position: torch.Tensor, has_jumped=False, already_searched=None) -> List[torch.Tensor]:
        already_searched = already_searched or set()
        result = []

        if not has_jumped:
            result.extend(self.__step_moves(position))

        for to_position in self.__jump_moves(position):
            key = self.to_key(to_position)

            if key not in already_searched:
                result.append(to_position)
                already_searched.add(key)

                moves = self.moves(to_position, has_jumped=True, already_searched=already_searched)

                result.extend(moves)
                already_searched.update(map(Board.to_key, moves))

        return result
        

    def all_moves(self) -> List[Tuple[torch.Tensor, List[torch.Tensor]]]:
        result = []

        for i in range(10):
            from_position = self.state[self.playing[0], i, :]
            result.append((from_position, self.moves(from_position)))

        return result

    
    @classmethod
    def to_key(self, position: torch.Tensor) -> Union[Tuple[int, int], List[Tuple[int, int]]]:
        reduce = False
        if len(position.shape) == 1:
            reduce = True
            position = position.view(1, -1)

        position = position.clone()

        position[:, 1] = position[:, 1]/np.sin(np.pi/3)
        position = torch.round(position)

        result = list(map(lambda k: tuple(map(int, k)), position.tolist()))
        if reduce:
            return result[0]
        return result
    
    @classmethod
    def board_keys(cls) -> Set[Tuple[int, int]]:
        if cls.__board_keys is None:
            cls.__board_keys = set(cls.to_key(cls.board()[:, :2]))
        return cls.__board_keys
        
    __board_keys = None
    __BOARD = None

    @classmethod
    def board(cls) -> torch.Tensor:
        if cls.__BOARD is None:
            board = torch.zeros(121, 3) # each point (x, y, floor)
            offset = 0
            for i in range(6):
                start_traingle = cls.start_traingle(i)
                slots = torch.concat([start_traingle, torch.tensor([((i+1),) for _ in range(10)], dtype=torch.get_default_dtype())], dim=1)
                board[offset:offset+10] = slots
                offset += 10
            
            empty = list()
            for i in range(6):
                triangle = cls.triangle(5)
                triangle @= cls.rotate_matrix(i)
                
                for j in range(triangle.size(0)):
                    left = triangle[j:j+1, :]
                    min_dist = 999
                    for right in empty:
                        min_dist = min(min_dist, torch.cdist(left, right))
                        
                    if min_dist > 0.1:
                        empty.append(left)

            for position in empty:
                board[offset] = torch.tensor((*position[0], 0), dtype=torch.get_default_dtype())
                offset += 1
            
            assert offset == board.size(0)
            cls.__BOARD = board

        return cls.__BOARD.clone()
            

    @classmethod
    def triangle(cls, height) -> torch.Tensor:
        # generate a top directed traingle
        result = torch.zeros((height*(height+1)//2, 2))
        row = torch.tensor([(2*x, 0) for x in range(0, height)], dtype=torch.get_default_dtype())
        delta = torch.tensor((-1, -2*np.sin(np.pi/3)), dtype=torch.get_default_dtype())
        i = 0

        for y in range(1, height+1):
            result[i: i+y] = row[0:y]
            row += delta
            i += y

        return result
    
    @classmethod
    def rotate_matrix(cls, steps=1) -> torch.Tensor:
        angle_radians = -steps * np.pi / 3

        return torch.Tensor([
            [np.cos(angle_radians), -np.sin(angle_radians)],
            [np.sin(angle_radians), np.cos(angle_radians)]
        ])
    
    @classmethod
    def start_traingle(cls, player=0) -> torch.Tensor:
        # generate player start triangle
        triangle = cls.triangle(4)
        triangle @= cls.rotate_matrix(3)
        triangle += torch.Tensor((0, -8*2*np.sin(np.pi/3)))
        triangle @= cls.rotate_matrix(player)

        return triangle


    @classmethod
    def new(cls, players: Optional[Iterable[Color]] = None, rotate=False):
        players = players or map(Color, range(6))
        players = set(map(lambda c: c.value, players))

        state = torch.zeros((6, 11, 2))

        state[0, 10, 1] = 1 # set red player
        state[min(players), 10, 0] = 1 # set current player

        for player in range(6):
            if player in players:
                state[player, :10, :] = cls.start_traingle(player)

        board = Board(state)
        if rotate:
            board.rotate(player=board.playing[1])
        return board