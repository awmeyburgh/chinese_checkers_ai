from torch import nn
import torch

from chinese_checkers_ai.v3.model.board import Board
from chinese_checkers_ai.v3.model.move import Move
from chinese_checkers_ai.v3.model.player import Player

class DeepQNetwork:
    def __init__(self):
        self.layers = nn.Sequential(
            nn.Conv1d(8, 32, kernel_size=3, padding=1),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 121)
        )

    def forward(self, x):
        return self.layers(x)
    
    def select_action(self, board: Board, player: Player) -> Move:
        moves = board.moves(player, board.focused)
        tensor = board.to_tensor(player, moves=moves)
        tensor = tensor.unsqueeze(0)
        q_values = self.forward(tensor)
        q_values = q_values.masked_fill(~tensor[7].bool(), float('-inf'))
        action = q_values.argmax(dim=1).item()
        for move in moves:
            if Board.get_position_to_idx()[move.to_position] == action:
                return move
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))