from enum import IntEnum

import pygame


class Player(IntEnum):
    RED = 0
    BLUE = 1
    GREEN = 2
    WHITE = 3
    BLACK = 4
    YELLOW = 5

    def to_pygame(self) -> pygame.Color:
        match self:
            case Player.RED:
                return pygame.colordict.THECOLORS['red']
            case Player.BLUE:
                return pygame.colordict.THECOLORS['blue']
            case Player.GREEN:
                return pygame.colordict.THECOLORS['green']
            case Player.WHITE:
                return pygame.colordict.THECOLORS['white']
            case Player.BLACK:
                return pygame.colordict.THECOLORS['gray2']
            case Player.YELLOW:
                return pygame.colordict.THECOLORS['yellow']

    def to_color(self):
        from chinese_checkers_ai.v3.model.color import Color
        return Color(self.value+1)