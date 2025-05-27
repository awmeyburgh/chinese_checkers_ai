from pathlib import Path
from typing import List
import torch
import pygame

from chinese_checkers_ai.v2.model.board import Board
from chinese_checkers_ai.v2.model.color import Color
from chinese_checkers_ai.v2.node import Node
from chinese_checkers_ai.v2.trainer.deepq import ChineseCheckersTrainer
from chinese_checkers_ai.v2.trainer.deepq import calculate_board_score as score
from chinese_checkers_ai.v2.view.board_view import BoardView


class VisualDeepQController(Node):
    def __init__(
            self, 
            view: BoardView,
            versus: int = 2,
            max_turns: int = 10,
            gamma: float = 0.2
        ):
        super().__init__()

        self.view = view
        self.board = self.view
        self.versus = versus
        self.max_turns = max_turns
        self.gamma = gamma
        self.state = 'new'
        self.__font = None
        self.turn = 0
        self.players: List[Color] = []
        self.last_score = 0
        self.trainer = None
        self.init_network()

    def init_network(self):
        filename = Path('chinese_checkers_model.pth')
        self.trainer = ChineseCheckersTrainer(gamma=self.gamma)
        if filename.exists():
            self.trainer.load(str(filename))

    def set_board(self, board: Board):
        self.board = board
        self.view.board = board

    def process(self, delta: float):
        if self.state == 'new':
            self.players = []
            for i in range(self.versus//2):
                self.players.extend([Color(i), Color(i+3)])
            self.set_board(Board.new(self.players, rotate=True))

            self.state = 'turn'
            self.turn = 1
        if self.state == 'turn':
            if self.turn <= self.max_turns:
                for i in range(self.versus):
                    loss, done = self.trainer.train_step(self.board)
                    if done:
                        self.state = 'done'
                        break
                if self.board.winner is not None:
                    self.state = 'done'
                self.turn += 1
            else:
                self.state = 'done'
        if self.state == 'done':
            if self.board.winner is not None:
                self.last_score = score(self.board, self.board.winner.value)
            else:
                mean_score = 0
                for player in self.players:
                    mean_score += score(self.board, player.value)
                mean_score /= len(self.players)
                self.last_score = mean_score
            self.state = 'new'
            # Save the model after each game
            self.trainer.save('chinese_checkers_model.pth')
            
    @property
    def font(self):
        if self.__font is None:
            self.__font = pygame.font.SysFont('Comic Sans MS', 16) 
        return self.__font
            
    def draw(self, surface: pygame.Surface):
        delta_y = 20
        render = [
            f'{player.name}: {score(self.board, player.value)}'
                for player in self.players
        ]
        render.insert(0, f'Turn: {self.turn}')
        render.append(f'Last Score: {self.last_score:.2f}')
        render.append(f'Epsilon: {self.trainer.epsilon:.3f}')

        for i in range(len(render)):
            text = self.font.render(
                render[i],
                True,
                pygame.colordict.THECOLORS['white']
            )

            surface.blit(
                text,
                (500, 20+delta_y*i)
            )