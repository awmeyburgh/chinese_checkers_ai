from enum import IntEnum

import pygame


class Color(IntEnum):
    RED = 0
    BLUE = 1
    GREEN = 2
    WHITE = 3
    BLACK = 4
    YELLOW = 5

    def floor(self) -> pygame.Color:
        match self:
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

    def piece(self) -> pygame.Color:
        match self:
            case Color.RED:
                return pygame.colordict.THECOLORS['red']
            case Color.BLUE:
                return pygame.colordict.THECOLORS['blue']
            case Color.GREEN:
                return pygame.colordict.THECOLORS['green']
            case Color.WHITE:
                return pygame.colordict.THECOLORS['white']
            case Color.BLACK:
                return pygame.colordict.THECOLORS['gray2']
            case Color.YELLOW:
                return pygame.colordict.THECOLORS['yellow']