"""
AI Controller using trained neural network for Chinese Checkers gameplay.
"""

from pathlib import Path
import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from chinese_checkers_ai.v2.model.board import Board
from chinese_checkers_ai.v2.model.color import Color
from chinese_checkers_ai.v2.trainer.evolution import QNetwork
from chinese_checkers_ai.v2.node import Node
from chinese_checkers_ai.v2.view.board_view import BoardView

class AIController(Node):
    def __init__(self, view: BoardView, players: List[Color], model_path: Optional[str] = None):
        """
        Initialize AI controller with a trained model.
        
        Args:
            view: The board view to control
            players: List of player colors this AI controls
            model_path: Path to the trained model file. If None, will look for 'evolution_best.network'
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = QNetwork().to(self.device)
        self.view = view
        self.players = players
        
        # Load model
        if model_path is None:
            model_path = Path("evolution_best.network")
        else:
            model_path = Path(model_path)
            
        if model_path.exists():
            print(f"Loading model from {model_path}")
            self.network.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            raise FileNotFoundError(f"No model file found at {model_path}")
            
        self.network.eval()  # Set to evaluation mode

    @property
    def board(self) -> Board:
        return self.view.board
        
    def select_move(self, board: Board) -> Tuple[tuple, tuple]:
        """
        Select the best move for the current board state.
        
        Args:
            board: Current game board
            
        Returns:
            Tuple of (from_position, to_position) representing the selected move
        """
        with torch.no_grad():
            # Get all valid moves
            valid_moves = []
            for from_pos, to_positions in board.all_moves():
                for to_pos in to_positions:
                    valid_moves.append((from_pos, to_pos))
            
            if not valid_moves:
                raise ValueError("No valid moves available")
            
            # Create states for each possible move
            next_states = []
            for from_pos, to_pos in valid_moves:
                # Create a copy of the board and make the move
                board_copy = Board(board.state.clone())
                board_copy.move(from_pos, to_pos)
                next_states.append(board_copy.state)
            
            # Stack all states and get Q-values
            states_tensor = torch.stack(next_states).to(self.device)
            q_values = self.network.q(states_tensor)
            
            # Select move with highest Q-value
            best_idx = q_values.squeeze().argmax().item()
            best_move = valid_moves[best_idx]
            
            return best_move
    
    def get_move(self, board: Board) -> Tuple[tuple, tuple]:
        """
        Interface method for the game controller.
        
        Args:
            board: Current game board
            
        Returns:
            Tuple of (from_position, to_position) representing the selected move
        """
        try:
            return self.select_move(board)
        except Exception as e:
            print(f"Error selecting move: {e}")
            # Fallback to first available move if there's an error
            for from_pos, to_positions in board.all_moves():
                if to_positions:
                    return (from_pos, to_positions[0])
            raise ValueError("No valid moves available")
    
    def process(self, delta: float):
        """
        Process the AI's turn in the game.
        Inherited from Node class.
        
        Args:
            delta: Time elapsed since last process call
        """
        if self.board.winner is None:
            if Color(self.board.playing[1]) in self.players:
                try:
                    move = self.get_move(self.board)
                    self.board.move(*move)
                except Exception as e:
                    print(f"Error during AI move: {e}")
                    # Don't raise the error to keep the game running