from typing import Optional, Tuple
import numpy as np
import pygame
from chinese_checkers_ai.input import Input
from chinese_checkers_ai.v3.model.color import Color
from chinese_checkers_ai.v3.model.player import Player
from chinese_checkers_ai.v3.view.board_view import BoardView
from chinese_checkers_ai.v3.controller.abstract_controller import AbstractController
from chinese_checkers_ai.v3.model.board import Board
from chinese_checkers_ai.v3.model.position import Position


class UserController(AbstractController):
    def __init__(self, game, players: list[Player]):
        super().__init__()
        self.game = game
        self.players = players

    def get_mouse_position(self) -> Optional[Position]:
        for group in self.game.view.board.floor():
            group = group.scale(BoardView.SCALE)
            for position in group.positions:
                euclid = position.euclid + self.game.view.offset + self.game.view.position
                if np.linalg.norm(euclid - np.array(Input.mouse_position())) < BoardView.SCALE:
                    return position.scale(1/BoardView.SCALE)
        return None

    def set_moves(self, board: Board, position: Position) -> bool:
        if self.game.playing in self.players:
            if position in board.players[self.game.playing]:
                self.game.view.moves = board.moves(self.game.playing, position)
            return True
        return False

    def move(self, board: Board, position: Position) -> Tuple[Board, bool]:
        for move in self.game.view.moves:
            if move.to_position == position:
                board = board.move(move)
                self.game.next_player()
                return board, True
        return board, False

    def process(self, board: Board) -> Board:
        if Input.key_pressed(pygame.K_r):
            board = board.rotate(1)
        if Input.mouse_pressed():
            position = self.get_mouse_position()
            if position is not None:
                if len(self.game.view.moves) == 0:
                    self.set_moves(board, position)
                else:
                    board, moved = self.move(board, position)
                    if not moved:
                        if not self.set_moves(board, position):
                            self.game.view.moves = None
                    return board

        if Input.key_pressed(pygame.K_f):
            position = self.get_mouse_position()
            if position is not None:
                for group in board.floor():
                    if position in group and group.color != Color.NONE:
                        board = board.focus(group.color.to_player())
                        break

        return board