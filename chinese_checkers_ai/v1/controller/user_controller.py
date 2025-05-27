from typing import Tuple, Union

import pygame
from chinese_checkers_ai.input import Input
from chinese_checkers_ai.v1.model.color import Color
from chinese_checkers_ai.v1.model.slot import Slot
from chinese_checkers_ai.v1.view.board import Board
from chinese_checkers_ai.v1.view.node import Node


class UserController(Node):
    def __init__(self, board: Board, players=None):
        super().__init__()
        self.board = board
        self.players = players or self.board.board.players

    def is_mouse_pressed(self, position: Union[Slot, Tuple[int, int]]) -> bool:
        slot_position = self.board.slot_position(position)
        dimensions = self.board.dimensions()
        slot_position = (
            self.board.offset[0]-dimensions[0]/2+slot_position[0],
            self.board.offset[1]-dimensions[1]/2+slot_position[1],
        )
        mouse_position = Input.mouse_position()

        return (slot_position[0]-mouse_position[0])**2 + (slot_position[1]-mouse_position[1])**2 <= self.board.FLOOR_RADIUS**2

    def process(self, delta: float):
        if self.board.board.player in self.players:
            self.board.board.rotate(Color((self.board.board.player.value+3)%6))

        if Input.mouse_pressed() and self.board.board.winner is None:
            pressed_slot = None
            for slot in self.board.board.slots.values():
                if self.is_mouse_pressed(slot):
                    pressed_slot = slot
                    break

            if pressed_slot is not None and pressed_slot.peice_color == self.board.board.player:
                if pressed_slot.peice_color in self.players:
                    self.board.moves = (slot.position, self.board.board.moves(slot.position))
            elif pressed_slot is not None and pressed_slot.position in self.board.moves[1]:
                if (winner:=self.board.board.move(self.board.moves[0], pressed_slot.position)) is not None:
                    print(f'winner {winner.name}')
                self.board.moves = (-1, set())
            else:
                self.board.moves = (-1, set())

        if Input.key_pressed(pygame.K_r):
            self.board.board.rotate()

    