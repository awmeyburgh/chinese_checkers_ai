from typing import Tuple
import pygame


class Input:
    __SINGLETON = None

    @classmethod
    def get(cls) -> "Input":
        if cls.__SINGLETON is None:
            cls.__SINGLETON = Input()
        return cls.__SINGLETON

    def __init__(self):
        self.events = []

    def process(self, events):
        self.events = events
        self.__mouse_pressed = False

        for event in self.events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.__mouse_pressed = True
                

    @classmethod
    def mouse_pressed(cls) -> bool:
        self = cls.get()
        return self.__mouse_pressed
    
    @classmethod
    def mouse_position(cls) -> Tuple[int, int]:
        return pygame.mouse.get_pos()
    
    @classmethod
    def key_pressed(cls, key) -> bool:
        self = cls.get()
        for event in self.events:
            if event.type == pygame.KEYUP:
                if event.key == key:
                    return True
        return False