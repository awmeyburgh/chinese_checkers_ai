import copy
from dataclasses import dataclass
import math
from random import Random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from chinese_checkers_ai.v1.model.color import Color
from chinese_checkers_ai.v1.model.slot import Slot

@dataclass
class Board:
    TOP_ENABLED = [
        (6, 0),
        (5, 1),
        (6, 1),
        (5, 2),
        (6, 2),
        (7, 2),
        (4, 3),
        (5, 3),
        (6, 3),
        (7, 3),
    ]

    LEFT_ENABLED = [
        (0, 4),
        (1, 4),
        (0, 5),
        (1, 5),
        (1, 6),
        (1, 7),
        (1, 9),
        (1, 10),
        (1, 11),
        (0, 11),
        (1, 12),
        (0, 12),
    ]

    RIGHT_ENABLED = [
        (11, 4),
        (12, 4),
        (11, 5),
        (11, 6),
        (11, 10),
        (11, 11),
        (11, 12),
        (12, 12),
    ]

    TOP_LEFT_ENABLED = [
        (0,4),
        (1,4),
        (2,4),
        (3,4),
        (0,5),
        (1,5),
        (2,5),
        (1,6),
        (2,6),
        (1,7),
    ]

    TOP_RIGHT_ENABLED = [
        (12,4),
        (11,4),
        (10,4),
        (9,4),
        (11,5),
        (10,5),
        (9,5),
        (11,6),
        (10,6),
        (10,7),
    ]

    slots: Dict[Tuple[int, int], Slot]
    players: Dict[Color, Set[Tuple[int, int]]]
    player: Color
    winner: Optional[Color]

    def has_piece(self, position: Tuple[int, int], is_none=None) -> Optional[bool]:
        if position not in self.slots or not self.slots[position].enabled:
            return is_none
        
        return self.slots[position].peice_color is not None

    def step_sides(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        # return all step sides regardless if they are on the board or valid moves
        if position[1]%2==0:
            return  [
                (position[0]-1, position[1]-1),
                (position[0], position[1]-1),
                (position[0]+1, position[1]),
                (position[0], position[1]+1),
                (position[0]-1, position[1]+1),
                (position[0]-1, position[1]),
            ]
        else:
            return [
                (position[0], position[1]-1),
                (position[0]+1, position[1]-1),
                (position[0]+1, position[1]),
                (position[0]+1, position[1]+1),
                (position[0], position[1]+1),
                (position[0]-1, position[1]),
            ]

    def jump_sides(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        # return all jump sides regardless if they are on the board or valid moves
        return [
            (position[0]-1, position[1]-2),
            (position[0]+1, position[1]-2),
            (position[0]+2, position[1]),
            (position[0]+1, position[1]+2),
            (position[0]-1, position[1]+2),
            (position[0]-2, position[1]),
        ]

    def step_moves(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        # return all step moves that are on the board and valid (no occupation by other piece)
        sides = self.step_sides(position)

        result = []

        for side in sides:
            if not self.has_piece(side, is_none=True):
                result.append(side)
        
        return result
    
    def jump_moves(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        result = []

        step_sides = self.step_sides(position)
        jump_sides = self.jump_sides(position)

        for i in range(len(step_sides)):
            if self.has_piece(step_sides[i], is_none=False) and not self.has_piece(jump_sides[i], is_none=True):
                result.append(jump_sides[i])

        return result

    def moves(self, position: Tuple[int, int], has_jumped:bool=False, already_seached:Optional[Set[Tuple[int, int]]]=None) -> Set[Tuple[int, int]]:
        already_seached = already_seached or {position}

        result = set()

        if not has_jumped:
            result.update(self.step_moves(position))
        
        for jump_position in self.jump_moves(position):
            if jump_position not in already_seached:
                result.add(jump_position)
                already_seached.add(jump_position)
                
                postions = self.moves(jump_position, has_jumped=True, already_seached=already_seached)

                result.update(postions)
                already_seached.update(postions)

        self.last_moves = (position, result)
        
        return result
    
    def all_moves(self) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
        result = {}

        for position in self.players[self.player]:
            result[position] = self.moves(position)

        return result
    
    def has_won(self, player: Color) -> bool:
        for position in self.players[player]:
            if self.slots[position].floor_color is None or self.slots[position].floor_color.value != (player.value+3)%6:
                return False
        return True
    
    def move(self, from_position: Tuple[int, int], to_position: Tuple[int, int], check=True) -> Optional[Color]:
        # performs move if valid and returns if 

        if check:
            if self.winner is not None:
                raise Exception(f'invalid move, game already complete')

            if not hasattr(self, 'last_moves') or self.last_moves[0] != from_position:
                self.moves(from_position)
            if to_position not in self.last_moves[1]:
                raise Exception(f"invalid move to {from_position} {to_position} (Invalid Route)")
            
            if self.has_piece(to_position, is_none=True):
                raise Exception(f"invalid move to {from_position} {to_position} (Occupied)")
            
            if not self.has_piece(from_position, is_none=False):
                raise Exception(f"invalid move from {from_position} {to_position}")
        
        player = self.slots[from_position].peice_color
        if player != self.player and check:
            raise Exception(f"Invalid player moving {player} expected {self.player}")
        
        self.slots[from_position].peice_color = None
        self.slots[to_position].peice_color = player

        self.players[player].remove(from_position)
        self.players[player].add(to_position)

        if self.has_won(player):
            self.winner = player

        for i in range(1, 6):
            player = Color((self.player.value+i)%6)
            if player in self.players:
                self.player = player
                break

        return self.winner


    def move_new(self, from_position: Tuple[int, int], to_position: Tuple[int, int], check=True) -> "Board":
        clone = self.clone()
        clone.move(from_position, to_position, check=check)
        return clone

    def clone(self) -> "Board":
        return copy.deepcopy(self)

    @property
    def order(self):
        if not hasattr(self, '__order'):
            self.__order = list(self.slots.keys())
            self.__order.sort()
        return self.__order
    
    @property
    def state(self) -> torch.Tensor:
        result = torch.zeros((len(self.order)+1, 6))
        top_color_value = self.slots[self.TOP_ENABLED[0]].floor_color.value
        
        result[0, :] = 1 if top_color_value == self.player else 0 # show in state if current player can move

        for i, position in enumerate(self.order):
            slot = self.slots[position]
            if slot.peice_color is not None:
                result[i+1, (slot.peice_color.value-top_color_value+6)%6] = 1

        return result


    def rotate(self, color: Optional[Color] = None):
        if color is None:
            angle_radians = 1
        else:
            top_color_value = self.slots[self.TOP_ENABLED[0]].floor_color.value
            angle_radians = ((top_color_value-color.value+6)%6)

        if angle_radians == 0:
            return
        
        angle_radians = angle_radians * np.pi/3
            

        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians)],
            [np.sin(angle_radians), np.cos(angle_radians)]
        ])
        
        transform = {
            position:
            self.from_orthogonal(
                rotation_matrix@self.to_orthogonal(position)
            )
            for position in self.order
        }

        slots = {}
        players = {player: set() for player in self.players}

        for slot in self.slots.values():
            slot.position = transform[slot.position]
            slots[slot.position] = slot
        
        for player in self.players:
            for position in self.players[player]:
                players[player].add(transform[position])

        self.slots = slots
        self.players = players

    def to_orthogonal(self, position: Tuple[int, int]) -> np.ndarray:
        result = np.zeros(2)

        result[1] = (position[1] - 8) * np.sin(np.pi/3)

        if position[1]%2 == 0:
            result[0] = position[0] - 6
        else:
            if position[0] >= 6:
                x = position[0] - 5
            else:
                x = position[0] - 6
            
            mag = abs(x)
            sign = x/mag

            if mag > 0:
                result[0] = sign*0.5
                mag -= 1
            
            result[0] += sign*mag
        
        return result
    
    def from_orthogonal(self, position: np.ndarray) -> Tuple[int, int]:
        y = np.round(position[1]/np.sin(np.pi/3)) + 8

        x_parts = np.round(position[0]/0.5)

        if y%2 == 0:
            x = x_parts//2 + 6
        else:
            mag = abs(x_parts)
            sign = x_parts/mag

            x = sign
            mag -= 1
            x += (mag//2)*sign

            if x > 0:
                x += 5
            else:
                x += 6

        return (int(x), int(y))


    @classmethod
    def new(cls, players: Optional[List[Color]] = None, random=False):
        players = players or [Color(i) for i in range(6)]
        players.sort()
        player = players[0]
        players: Dict[Color, Set[Tuple[int, int]]] = {player:set() for player in players}

        slots: Dict[Tuple[int, int], Slot] = {}

        for x in range(13):
            for y in range(17):
                slots[(x, y)] = Slot((x, y), True, None, None)

        #top trim
        for y in range(4):
            for x in range(13):
                slots[(x,y)].enabled = False
        for pos in cls.TOP_ENABLED:
            slots[pos].enabled = True

        #bottom trim
        for y in range(17-4,17):
            for x in range(13):
                slots[(x,y)].enabled = False
        for (x,y) in cls.TOP_ENABLED:
            slots[(x, 16-y)].enabled = True

        #left trim
        for y in range(4, 17-4):
            for x in range(2):
                slots[(x,y)].enabled = False
        for pos in cls.LEFT_ENABLED:
            slots[pos].enabled = True

        #right trim
        for y in range(4, 17-4):
            for x in range(13-2,13):
                slots[(x,y)].enabled = False
        for pos in cls.RIGHT_ENABLED:
            slots[pos].enabled = True

        for slot in list(slots.values()):
            if not slot.enabled:
                del slots[slot.position]

        #top color
        for pos in cls.TOP_ENABLED:
            slots[pos].floor_color = Color(0)
            if Color(0) in players and not random:
                slots[pos].peice_color = Color(0)
                players[Color(0)].add(pos)

        #top right color
        for pos in cls.TOP_RIGHT_ENABLED:
            slots[pos].floor_color = Color(1)
            if Color(1) in players and not random:
                slots[pos].peice_color = Color(1)
                players[Color(1)].add(pos)

        #bottom right color
        for (x,y) in cls.TOP_RIGHT_ENABLED:
            slots[(x, 16-y)].floor_color = Color(2)
            if Color(2) in players and not random:
                slots[(x, 16-y)].peice_color = Color(2)
                players[Color(2)].add((x, 16-y))

        #bottom color
        for (x,y) in cls.TOP_ENABLED:
            slots[(x, 16-y)].floor_color = Color(3)
            if Color(3) in players and not random:
                slots[(x, 16-y)].peice_color = Color(3)
                players[Color(3)].add((x, 16-y))

        #bottom left color
        for (x,y) in cls.TOP_LEFT_ENABLED:
            slots[(x, 16-y)].floor_color = Color(4)
            if Color(4) in players and not random:
                slots[(x, 16-y)].peice_color = Color(4)
                players[Color(4)].add((x, 16-y))

        #top left color
        for pos in cls.TOP_LEFT_ENABLED:
            slots[pos].floor_color = Color(5)
            if Color(5) in players and not random:
                slots[pos].peice_color = Color(5)
                players[Color(5)].add(pos)

        if random:
            rng = Random()
            open = list(slots.keys())

            for color in players:
                for i in range(10):
                    position = rng.choice(open)
                    open.remove(position)
                    slots[position].peice_color = color
                    players[color].add(position)

        return Board(slots, players, player, None)
