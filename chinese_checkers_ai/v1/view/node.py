from typing import List, Optional
import pygame


class Node:
    def __init__(self, children: Optional[List["Node"]] = None):
        self.children = children or []

    def draw(self, surface: pygame.Surface):
        for child in self.children:
            child.draw(surface)

    def process(self, delta: float):
        for child in self.children:
            child.process(delta)