import os
from pathlib import Path
import sys
import pygame
from chinese_checkers_ai.v1.controller.q_controller import QController
from chinese_checkers_ai.v1.controller.user_controller import UserController
from chinese_checkers_ai.input import Input
from chinese_checkers_ai.v1.model.color import Color
from chinese_checkers_ai.v1.trainer import neat
from chinese_checkers_ai.v1.view.board import Board
from chinese_checkers_ai.v1.view.node import Node
from chinese_checkers_ai.v1.model.board import Board as BoardModal


class Game(Node):
    BACKGROUND_COLOR = pygame.colordict.THECOLORS['saddlebrown']

    __SINGLETON = None

    @classmethod
    def get(cls) -> "Game":
        if cls.__SINGLETON is None:
            cls.__SINGLETON = Game()
        return cls.__SINGLETON

    def __init__(self):
        pygame.init()
        pygame.font.init() 

        self.running = False
        self.board = BoardModal.new([Color(0), Color(3)])
        self.board_view = Board(self.board, offset=(300, 300))
        self.controller = UserController(self.board_view)
        self.input = Input.get()

        self.option = None
        if len(sys.argv) > 1:
            self.option = sys.argv[1]

        super().__init__([
            self.board_view,
            self.controller,
            QController.stupid_player(self.board, Color(3))
        ])

    def draw(self, surface: pygame.Surface):
        surface.fill(self.BACKGROUND_COLOR)
        super().draw(surface)

    def run(self):
        if self.option == 'neat':
            neat.run()
        else:
            screen = pygame.display.set_mode((600,600))
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