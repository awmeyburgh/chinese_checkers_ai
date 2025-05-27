from typing import List, Optional, Tuple

import numpy as np
import pygame
from chinese_checkers_ai.v3.model.board import Board
from chinese_checkers_ai.v3.model.group import Group
from chinese_checkers_ai.v3.model.move import Move
from chinese_checkers_ai.v3.model.position import Position
from chinese_checkers_ai.v3.node import Node


class BoardView(Node):
    SCALE = 20
    __FLOOR_RADUIS = SCALE
    __PIECE_RADUIS = 17
    __BACKGROUND_COLOR = pygame.colordict.THECOLORS['saddlebrown']
    __MOVE_COLOR = pygame.colordict.THECOLORS['cadetblue3']
    
    def __init__(
        self,
        board: Board,
        position: Optional[np.ndarray] = None,
        debug: bool = False
    ):
        super().__init__(position=position)

        self.__board = board
        self.__rendered_board = None
        self.__dimensions = None
        self.__offset = None
        self.__moves = None
        self.__font = None
        self.debug = debug

    @property
    def board(self) -> Board:
        return self.__board
    
    @board.setter
    def board(self, board: Board):
        if board != self.__board:
            self.__board = board
            self.__moves = None
            self.__rendered_board = None

    @property
    def moves(self) -> List[Move]:
        if self.__moves is None:
            return []
        return self.__moves

    @moves.setter
    def moves(self, moves: List[Move]):
        self.__moves = moves
        self.__rendered_board = None

    @property
    def rendered_board(self) -> pygame.Surface:
        if self.__rendered_board is None:
            self.__rendered_board = self.render()
        return self.__rendered_board
    
    def render_move(self, surface: pygame.Surface, move: Move):
        position = move.to_position.scale(self.SCALE)

        pygame.draw.circle(
            surface, 
            self.__MOVE_COLOR, 
            (position.euclid + self.offset), 
            self.__FLOOR_RADUIS
        )
    
    def render_group(self, surface: pygame.Surface, group: Group):
        group = group.scale(self.SCALE)
        for position in group.positions:
            radius = self.__PIECE_RADUIS if group.is_player else self.__FLOOR_RADUIS
            color = group.color.to_player().to_pygame() if group.is_player else group.color.to_pygame()
            pygame.draw.circle(
                surface, 
                color, 
                (position.euclid + self.offset), 
                radius
            )

    def font(self) -> pygame.font.Font:
        if self.__font is None:
            self.__font = pygame.font.SysFont('arial', 10)
        return self.__font

    def render_debug(self, surface: pygame.Surface, group: Group):
        group = group.scale(self.SCALE)
        for position in group.positions:
            text = self.font().render(
                str(position), 
                True, 
                pygame.colordict.THECOLORS['black']
            )
            surface.blit(text, (position.euclid + self.offset - np.array([text.get_width()/2, text.get_height()/2])))

    @property
    def offset(self) -> np.ndarray:
        if self.__offset is None:
            self.__offset = np.array([0, 0])
            for group in self.board.floor():
                group = group.scale(self.SCALE)
                for position in group.positions:
                    self.__offset[0] = min(self.__offset[0], position.euclid[0])
                    self.__offset[1] = min(self.__offset[1], position.euclid[1])
            self.__offset -= np.array([self.__FLOOR_RADUIS, self.__FLOOR_RADUIS])
            self.__offset = -self.__offset
        return self.__offset
    
    @property
    def dimensions(self) -> np.ndarray:
        if self.__dimensions is None:
            self.__dimensions = np.array([0, 0])
            for group in self.board.floor():
                group = group.scale(self.SCALE)
                for position in group.positions:
                    self.__dimensions[0] = max(self.__dimensions[0], position.euclid[0])
                    self.__dimensions[1] = max(self.__dimensions[1], position.euclid[1])
            self.__dimensions += np.array([self.__FLOOR_RADUIS, self.__FLOOR_RADUIS])
            self.__dimensions += self.offset
        return self.__dimensions
    
    def render(self) -> pygame.Surface:
        surface = pygame.Surface(self.dimensions, pygame.SRCALPHA)

        for group in self.board.floor():
            self.render_group(surface, group)
        for player in self.board.players.values():
            self.render_group(surface, player)
        for move in self.moves:
            self.render_move(surface, move)

        if self.debug:
            for group in self.board.floor():
                self.render_debug(surface, group)

        return surface
        

    def draw(self, surface: pygame.Surface):
        surface.fill(self.__BACKGROUND_COLOR)
        surface.blit(
            self.rendered_board, 
            self.position
        )
    
