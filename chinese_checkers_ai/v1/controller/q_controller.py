from pathlib import Path
from typing import Callable
from pygame import Color
import torch
from chinese_checkers_ai.v1.model.board import Board
from chinese_checkers_ai.v1.trainer.neat import QNetwork
from chinese_checkers_ai.v1.view import board
from chinese_checkers_ai.v1.view.node import Node
import torch.nn as nn

class QController(Node):
    def __init__(
            self, 
            board: Board, 
            player: Color, 
            q_function: Callable[[torch.Tensor], float]
        ):
        super().__init__()

        self.board = board
        self.player = player
        self.q_function = q_function

    def process(self, delta):
        if self.board.player == self.player:
            board = self.board.clone()
            board.rotate(self.player)

            next = (None, None, -1)

            for from_position, to_positions in board.all_moves().items():
                for to_position in to_positions:
                    next_board = board.move_new(from_position, to_position, check=False)
                    quality = self.q_function(next_board.state)
                    if quality > next[2]:
                        next = (from_position, to_position, quality)

            self.board.move(next[0], next[1])

    @classmethod
    def stupid_player(cls, board: Board, player: Color):
        layers = nn.Sequential(
            nn.Flatten(0),
            nn.Linear(board.state.numel(), 1),
            nn.Sigmoid()
        )

        return cls(
            board,
            player,
            lambda state: layers(state)[0]
        )
    
    @classmethod
    def neat(cls, board: Board, player: Color, file: Path):
        network = torch.load(file, weights_only=False)

        return cls(
            board,
            player,
            lambda state: network.q(state)
        )