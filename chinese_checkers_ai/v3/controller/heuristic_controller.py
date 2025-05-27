import random
from typing import Dict, List, Set, Tuple
import numpy as np
from collections import deque
from scipy.spatial.distance import cdist

from chinese_checkers_ai.v3.controller.abstract_controller import AbstractController
from chinese_checkers_ai.v3.model.board import Board
from chinese_checkers_ai.v3.model.group import Group
from chinese_checkers_ai.v3.model.player import Player
from chinese_checkers_ai.v3.model.position import Position
from chinese_checkers_ai.v3.model.move import Move


class HeuristicController(AbstractController):
    def __init__(self, game, players: list[Player], history: int = 10, depth: int = 7, top_depth_size = 12, dropout = 0.1):
        """Initialize the heuristic controller.
        
        Args:
            players: List of players this controller controls
        """
        super().__init__()
        self.game = game
        self.players = players
        self.history = history
        self.move_history = {player: [] for player in players}
        self.depth = depth
        self.top_depth_size = top_depth_size
        self.dropout = dropout

    def group_positions(self, group: Group, size = 10) -> np.ndarray:
        positions = np.zeros((size, 2))
        for i, position in enumerate(group.positions):
            positions[i] = position.euclid
        return positions

    def evaluate_goal_distance(self, board: Board, player: Player) -> float:
        goal = board.goal(player)

        player_positions = self.group_positions(board.players[player])
        goal_positions = self.group_positions(goal)
        
        return 15 - np.linalg.norm(player_positions - goal_positions, axis=1).mean()
    
    def evaluate_home_retrieval(self, board: Board, player: Player) -> float:
        player_positions = self.group_positions(board.players[player])
        back = np.clip((player_positions - player_positions.mean(axis=0))[:, 1], -100, 0)
        return back.mean()
    
    def evaluate_goal_equality(self, board: Board, player: Player) -> float:
        goal = board.goal(player)
        for position in board.players[player].positions:
            if position not in goal:
                return 0
        return 500


    def evaluate_board(self, board: Board, player: Player) -> float:
        goal_distance = self.evaluate_goal_distance(board, player)
        home_retrieval = self.evaluate_home_retrieval(board, player)
        goal_equality = self.evaluate_goal_equality(board, player)
        return goal_distance * 1.5 \
            + home_retrieval * 2.5 \
            + goal_equality
        
    def sparse_boards(self, player: Player, boards: List[Tuple[Board, Move]]) -> List[Tuple[Board, Move]]:
        one_moves = [boards[0]]
        boards = [board for board in boards if random.random() > self.dropout]
        boards = sorted(boards, key=lambda x: self.evaluate_board(x[0], player), reverse=True)[:self.top_depth_size]
        return boards if len(boards) > 0 else one_moves

    def get_boards(self, board: Board, player: Player) -> List[Tuple[Board, Move]]:
        boards = [(board.move(move, validate=False), move) for move in board.all_moves(player) if move not in self.move_history[player]]
        for _ in range(1, self.depth):

            for _ in range(len(boards)):
                board, move = boards.pop(0)
                for next_move in board.all_moves(player):
                    if next_move not in self.move_history[player]:
                        boards.append((board.move(next_move, validate=False), move))
        return boards

    def get_move(self, board: Board, player: Player) -> Move:
        moves = []

        for (board, move) in self.get_boards(board, player):
            moves.append((self.evaluate_board(board, player), move))

        return max(moves, key=lambda x: x[0])[1]

    def process(self, board: Board) -> Board:
        if self.game.playing in self.players:
            prev_focus = board.focused
            board = board.focus(self.game.playing)
            move = self.get_move(board, self.game.playing)
            self.move_history[self.game.playing].append(move)
            if len(self.move_history[self.game.playing]) > self.history:
                self.move_history[self.game.playing].pop(0)
            board = board.move(move)
            board = board.focus(prev_focus)
            self.game.next_player()
        return board