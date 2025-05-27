from typing import List, Optional
import numpy as np
import pygame


class Node:
    def __init__(self, position: Optional[np.ndarray] = None, children: Optional[List["Node"]] = None):
        self.position = np.zeros(2) if position is None else position
        self.children = children or []

    def draw(self, surface: pygame.Surface):
        for child in self.children:
            child.draw(surface)

    def process(self, delta: float):
        for child in self.children:
            child.process(delta)