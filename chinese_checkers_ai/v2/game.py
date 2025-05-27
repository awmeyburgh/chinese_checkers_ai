import os
from pathlib import Path
import sys
from chinese_checkers_ai.v2.controller.ai_controller import AIController
from chinese_checkers_ai.v2.controller.heuristic_controller import HeuristicController
import pygame
import torch

from chinese_checkers_ai.input import Input
from chinese_checkers_ai.v2.controller.naive_controller import NaiveController
from chinese_checkers_ai.v2.controller.user_board_controller import UserBoardController
from chinese_checkers_ai.v2.model.board import Board
from chinese_checkers_ai.v2.model.color import Color
from chinese_checkers_ai.v2.node import Node
from chinese_checkers_ai.v2.trainer import deepq, evolution, deepq2
from chinese_checkers_ai.v2.view.board_view import BoardView
from chinese_checkers_ai.v2.controller.visual_deep_q_controller import VisualDeepQController


class Game(Node):
    BACKGROUND_COLOR = pygame.colordict.THECOLORS['saddlebrown']

    __SINGLETON = None

    @classmethod
    def get(cls) -> "Game":
        if cls.__SINGLETON is None:
            cls.__SINGLETON = Game()
        return cls.__SINGLETON

    def __init__(self):
        self.running = False
        self.board = Board.new(players=[Color(0), Color(3)])
        self.input = Input.get()
        self.dimensions = torch.tensor((600,600))
        self.board_view = BoardView(
            self.board,
            position=(self.dimensions-BoardView.dimensions())/2
        )
        self.mode = 'normal'
        if len(sys.argv) > 1:
            self.mode = sys.argv[1]
        mode_children = []
        if self.mode == 'normal':
            mode_children.extend([
                # UserBoardController(self.board_view, [Color(0)]),
                # NaiveController(self.board_view, list(map(Color, range(6))))
                HeuristicController(self.board_view, [Color(0), Color(3)])
            ])
        if self.mode == 'deepq-visual':
            mode_children.append(
                VisualDeepQController(self.board_view)
            )

        super().__init__(children=[
            self.board_view,
            *mode_children
        ])

    def draw(self, surface: pygame.Surface):
        surface.fill(self.BACKGROUND_COLOR)
        super().draw(surface)

    def run_game(self):

        pygame.init()
        pygame.font.init()

        screen = pygame.display.set_mode(self.dimensions.tolist())
        clock = pygame.time.Clock()

        self.running = True

        while self.running:
            events = []

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                else:
                    events.append(event)

            self.draw(screen)
            pygame.display.flip()

            self.input.process(events)

            delta = clock.tick(60)/1000
            self.process(delta)

        pygame.quit()

    def run(self):
        if self.mode == 'normal' or self.mode == 'deepq-visual':
            self.run_game()
        elif self.mode == 'deepq':
            deepq.run()
        elif self.mode == 'evolution':
            evolution.run()
        elif self.mode == 'deepq2':
            deepq2.run()