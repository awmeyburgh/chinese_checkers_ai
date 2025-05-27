from enum import IntEnum

import pygame

from chinese_checkers_ai.v3.model.player import Player


class Color(IntEnum):
    NONE = 0
    RED = 1
    BLUE = 2
    GREEN = 3
    WHITE = 4
    BLACK = 5
    YELLOW = 6

    def to_pygame(self) -> pygame.Color:
        match self:
            case Color.NONE:
                return pygame.colordict.THECOLORS['burlywood']
            case Color.RED:
                return pygame.colordict.THECOLORS['red4']
            case Color.BLUE:
                return pygame.colordict.THECOLORS['blue4']
            case Color.GREEN:
                return pygame.colordict.THECOLORS['green4']
            case Color.WHITE:
                return pygame.colordict.THECOLORS['gray80']
            case Color.BLACK:
                return pygame.colordict.THECOLORS['gray10']
            case Color.YELLOW:
                return pygame.colordict.THECOLORS['yellow4']

    def to_player(self) -> Player:
        return Player(self.value-1)