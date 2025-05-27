import pygame
import torch
from chinese_checkers_ai.v2.model.board import Board, BoardSnapshot
from chinese_checkers_ai.v2.node import Node


class BoardView(Node):
    FLOOR_COLOR = pygame.colordict.THECOLORS['burlywood']
    MOVE_COLOR = pygame.colordict.THECOLORS['cadetblue3']
    FLOOR_RADIUS = 20
    PIECE_RADIUS = 17

    def __init__(self, board: Board, position=None, debug=False):
        super().__init__(position=position)

        self.board = board
        self.moves = set()
        self.offset = torch.zeros(2)
        self.debug = debug

    @classmethod
    def dimensions(cls) -> torch.Tensor:
        return BoardSnapshot.dimensions() * cls.FLOOR_RADIUS
    
    def transform(self, position: torch.Tensor) -> torch.Tensor:
        return position*self.FLOOR_RADIUS + self.offset

    def draw(self, surface: pygame.Surface):
        snapshot = self.board.snapshot()
        self.offset = snapshot.top_left_center_offset()*self.FLOOR_RADIUS + self.position

        for position, color in snapshot.floor:
            if color is None:
                color = self.FLOOR_COLOR
            else:
                color = color.floor()

            position = self.transform(position)

            pygame.draw.circle(
                surface,
                color,
                position.tolist(),
                self.FLOOR_RADIUS
            )

        for position, color in snapshot.pieces:
            if color is not None:
                color = color.piece()
                position = self.transform(position)

                pygame.draw.circle(
                    surface,
                    color,
                    position.tolist(),
                    self.PIECE_RADIUS
                )

        for position in self.moves:
            position = self.transform(position)

            pygame.draw.circle(
                surface,
                self.MOVE_COLOR,
                position.tolist(),
                self.FLOOR_RADIUS
            )

        for position, _ in snapshot.floor:
            if self.debug:
                font = pygame.font.SysFont('Comic Sans MS', 12) 
                text = font.render(
                    str(self.board.to_key(position)),
                    True,
                    pygame.colordict.THECOLORS['black']
                )
                transform = (
                    self.transform(position) 
                    - torch.tensor(
                        (text.get_width(), text.get_height()), 
                        dtype=torch.get_default_dtype()
                    )/2
                )
                surface.blit(
                    text,
                    transform.tolist()
                )