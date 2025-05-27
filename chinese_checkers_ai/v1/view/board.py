from ctypes import pointer
from turtle import position
from typing import Tuple, Union
import pygame
from chinese_checkers_ai.input import Input
from chinese_checkers_ai.v1.model.board import Board as BoardModel
from chinese_checkers_ai.v1.model.slot import Slot
from chinese_checkers_ai.v1.view.node import Node

class Board(Node):
    FLOOR_COLOR = pygame.colordict.THECOLORS['burlywood']
    MOVE_COLOR = pygame.colordict.THECOLORS['cadetblue3']
    FLOOR_RADIUS = 20
    PIECE_RADIUS = 18

    def __init__(self, board: BoardModel, offset=(0,0)):
        super().__init__()

        self.board = board
        self.offset = offset
        self.moves = (-1, set())
        self.font = pygame.font.SysFont('Comic Sans MS', 12)

        for position in self.board.slots:
            assert position == self.board.from_orthogonal(self.board.to_orthogonal(position))

    def slot_position(self, position: Union[Slot, Tuple[int, int]]):
        if isinstance(position, Slot):
            position = position.position

        x = 2*self.FLOOR_RADIUS*position[0] + self.FLOOR_RADIUS
        y = 2*self.FLOOR_RADIUS*position[1] + self.FLOOR_RADIUS

        if position[1]%2==1:
            x += self.FLOOR_RADIUS

        return (x, y)
    
    def dimensions(self) -> Tuple[int, int]:
        return (13*2*self.FLOOR_RADIUS, 17*2*self.FLOOR_RADIUS)

    def draw(self, screen: pygame.Surface):
        surface = pygame.Surface(self.dimensions(), pygame.SRCALPHA)

        for slot in self.board.slots.values():
            if slot.enabled:
                slot_position = self.slot_position(slot)

                # draw floor
                floor_color = self.FLOOR_COLOR
                if slot.floor_color is not None:
                    floor_color = slot.floor_color.floor()

                pygame.draw.circle(
                    surface,
                    floor_color,
                    slot_position,
                    self.FLOOR_RADIUS,
                )

                #draw piece
                if slot.peice_color is not None:
                    pygame.draw.circle(
                        surface,
                        slot.peice_color.piece(),
                        slot_position,
                        self.PIECE_RADIUS
                    )

                #debug position
                text_surface = self.font.render(
                    ','.join(f'{x:.2f}' for x in self.board.to_orthogonal(slot.position)),
                    True,
                    pygame.colordict.THECOLORS['black']
                )
                surface.blit(
                    text_surface,
                    (slot_position[0]-text_surface.get_width()/2, slot_position[1]-text_surface.get_height()/2)
                )

        for position in self.moves[1]:
            pygame.draw.circle(
                surface,
                self.MOVE_COLOR,
                self.slot_position(position),
                self.FLOOR_RADIUS,
            )


        screen.blit(surface, (self.offset[0]-surface.get_width()/2, self.offset[1]-surface.get_height()/2))

                    