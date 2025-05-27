from typing import List
from chinese_checkers_ai.v2.model.board import Board
from chinese_checkers_ai.v2.model.color import Color
from chinese_checkers_ai.v2.node import Node
from chinese_checkers_ai.v2.trainer.deepq import calculate_board_score as score
from chinese_checkers_ai.v2.view.board_view import BoardView


class NaiveController(Node):
    def __init__(self, view: BoardView, players: List[Color], lookahead=1):
        super().__init__()
        self.view = view
        self.players = players
        self.lookahead = 1

    @property
    def board(self) -> Board:
        return self.view.board

    def process(self, delta: float):
        if self.board.winner is None:
            if Color(self.board.playing[1]) in self.players:
                best = (None, None, -1)

                states = []

                for fro, tos in self.board.all_moves():
                    for to in tos:
                        states.append((fro, to, self.board.move_new(fro, to, check=False)))

                for j in range(1, self.lookahead):
                    new_states = []
                    for (fro, to, state) in states:
                        for nfro, ntos in  state.all_moves():
                            for nto in ntos:
                                new_states.append((fro, to, self.board.move_new(nfro, nto, check=False)))
                    states = new_states

                for fro, to, state in states:
                    _score = score(state, self.board.playing[1])
                    if _score > best[2]:
                        best = (fro, to, _score)

                self.board.move(best[0], best[1])

