from typing import List, Optional
import pygame
import torch


class Node:
    def __init__(self, position: Optional[torch.Tensor] = None, children: Optional[List["Node"]] = None):
        self.position = torch.zeros(2) if position is None else position
        self.children = children or []

    def draw(self, surface: pygame.Surface):
        for child in self.children:
            child.draw(surface)

    def process(self, delta: float):
        for child in self.children:
            child.process(delta)