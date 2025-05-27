from typing import List, Optional
import pygame
import torch
from chinese_checkers_ai.input import Input
from chinese_checkers_ai.v2.model.board import Board
from chinese_checkers_ai.v2.model.color import Color
from chinese_checkers_ai.v2.node import Node
from chinese_checkers_ai.v2.trainer.deepq import calculate_board_score as score
from chinese_checkers_ai.v2.view.board_view import BoardView


class UserBoardController(Node):
    def __init__(self, view: BoardView, players: Optional[List[Color]] = None, autorestart=True):
        super().__init__()
        self.view = view
        self.board = view.board
        self.from_position = None
        self.__moves = set()
        self.players = players or [Color(p[1]) for p in self.board.players]
        self.autorestart = autorestart

    def moves(self, from_position=None):
        if from_position is None:
            self.view.moves = set()
            self.__moves = set()
        else:
            self.view.moves = set(self.board.moves(from_position))
            self.__moves = set(map(Board.to_key, self.view.moves))
        self.from_position = from_position

    def process(self, delta: float):
        if Input.key_pressed(pygame.K_r):
            self.board.rotate()


        if Input.mouse_pressed() and Color(self.board.playing[1]) in self.players and self.board.winner is None:
            board = Board.board()
            board = board[:, :2]
            transformed_board = self.view.transform(board)

            mouse = torch.tensor(Input.mouse_position(), dtype=torch.get_default_dtype()).view(1, -1)

            triggers = torch.cdist(transformed_board, mouse) <= self.view.FLOOR_RADIUS

            if torch.any(triggers):
                index = torch.max(triggers, dim=0).indices[0].item()
                key = self.board.to_key(board[index, :])

                if key in self.board.playing_keys:
                    self.moves(board[index, :])
                elif key in self.__moves:
                    # playing = self.board.playing[1]
                    self.board.move(self.from_position, board[index, :])
                    # print(score(self.board, playing))
                    # self.board.rotate(player=self.board.playing[1])
                    self.moves()
                else:
                    self.moves()
            else:
                self.moves()

        if self.autorestart and self.board.winner is not None:
            self.view.board = Board.new([Color(p[1]) for p in self.board.players])
            self.board = self.view.board