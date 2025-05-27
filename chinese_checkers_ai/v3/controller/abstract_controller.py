from abc import ABC, abstractmethod
from chinese_checkers_ai.v2.model.board import Board
from chinese_checkers_ai.v3.node import Node


class AbstractController(ABC):
    @abstractmethod
    def process(self, board: Board) -> Board:
        pass