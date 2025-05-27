from typing import List, Optional, Union
import numpy as np
import pygame

from chinese_checkers_ai.input import Input
from chinese_checkers_ai.v2.node import Node
from chinese_checkers_ai.v3.controller.abstract_controller import AbstractController
from chinese_checkers_ai.v3.controller.heuristic_controller import HeuristicController
from chinese_checkers_ai.v3.controller.user_controller import UserController
from chinese_checkers_ai.v3.model.board import Board
from chinese_checkers_ai.v3.view.board_view import BoardView
from chinese_checkers_ai.v3.model.player import Player


class Game(Node):
    def __init__(
        self, 
        dimensions: np.ndarray = np.array([600, 600]),
        players: Optional[Union[List[Player], int]] = None
    ):
        self.running = False
        self.input = Input.get()
        self.dimensions = dimensions

        self.players = Board.to_players(players)
        self.__playing_index = 0
        self.__controller_index = 0
        self.view = BoardView(Board.new(players=self.players))
        self.view.position = (self.dimensions - self.view.dimensions) / 2

        self.controllers = [
            # UserController(self, [self.players[0]]),
            HeuristicController(self, list(self.players[0:]))
        ]
        self.move_counts = {player: 0 for player in self.players}

        super().__init__(children=[self.view])

    @property
    def playing(self) -> Player:
        return self.players[self.__playing_index]
    
    def next_player(self):
        self.__playing_index = (self.__playing_index + 1) % len(self.players)

    def process(self, delta: float):
        player = self.playing
        controller = self.controllers[self.__controller_index]
        self.view.board = controller.process(self.view.board)
        self.__controller_index = (self.__controller_index + 1) % len(self.controllers)
        if self.playing != player:
            self.move_counts[player] += 1
        if self.view.board.has_won(player):
            print(f"Player {player.name} has won! ({self.move_counts[player]})")
            self.view.board = Board.new(players=self.players)
            self.move_counts = {player: 0 for player in self.players}
            self.__playing_index = 0

    def run(self):
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