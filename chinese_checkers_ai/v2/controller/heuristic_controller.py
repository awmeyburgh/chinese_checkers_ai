"""
Heuristic-based controller for Chinese Checkers that uses strategic evaluation
instead of deep learning to select moves.
"""

import math
from typing import List, Tuple, Set, Dict, Union
import torch
import numpy as np
from collections import deque
from chinese_checkers_ai.v2.model.board import Board
from chinese_checkers_ai.v2.model.color import Color
from chinese_checkers_ai.v2.node import Node
from chinese_checkers_ai.v2.view.board_view import BoardView

class HeuristicController(Node):
    def __init__(self, view: BoardView, players: List[Color], lookahead: int = 2):
        """
        Initialize the heuristic controller.
        
        Args:
            view: The board view to control
            players: List of player colors this controller controls
            lookahead: Number of moves to look ahead (default: 2)
        """
        super().__init__()
        self.view = view
        self.players = players
        self.lookahead = lookahead
        # Track move history for each player to detect repetitions
        self.move_history: Dict[int, deque] = {
            player.value: deque(maxlen=10) for player in players
        }
        # Track position history for each piece to detect oscillation
        self.position_history: Dict[tuple, deque] = {}
        
    @property
    def board(self) -> Board:
        return self.view.board
    
    def update_move_history(self, player_idx: int, from_pos: tuple, to_pos: tuple):
        """
        Update the move history for a player.
        
        Args:
            player_idx: The player's index
            from_pos: Starting position of the move
            to_pos: Ending position of the move
        """
        self.move_history[player_idx].append((from_pos, to_pos))
        
        # Initialize position history for the piece if not exists
        if to_pos not in self.position_history:
            self.position_history[to_pos] = deque(maxlen=5)
        
        # Update position history for the piece
        self.position_history[to_pos].append(from_pos)
    
    def get_repetition_penalty(self, from_pos: tuple, to_pos: tuple, player_idx: int) -> float:
        """
        Calculate penalty for repetitive moves.
        
        Args:
            from_pos: Starting position
            to_pos: Ending position
            player_idx: Player's index
            
        Returns:
            float: Penalty value (negative number)
        """
        penalty = 0.0
        
        # Penalize moves that undo the last move
        if len(self.move_history[player_idx]) > 0:
            last_move = self.move_history[player_idx][-1]
            if last_move[1] == from_pos and last_move[0] == to_pos:
                penalty -= 20.0  # Heavy penalty for immediate move reversal
        
        # Penalize moves to recently visited positions
        if to_pos in self.position_history:
            recent_positions = self.position_history[to_pos]
            if from_pos in recent_positions:
                # Penalty increases with frequency of the move
                count = sum(1 for pos in recent_positions if pos == from_pos)
                penalty -= count * 10.0
        
        return penalty
    
    def adjust_position_for_player(self, pos: tuple, player_idx: int) -> tuple:
        """
        Adjust position coordinates based on player's perspective.
        For players 0-5, the board needs to be rotated accordingly.
        
        Args:
            pos: Original position tuple (x, y)
            player_idx: Player index (0-5)
            
        Returns:
            tuple: Adjusted position coordinates
        """
        # Convert to numpy array for easier rotation
        pos_array = np.array([pos[0] - 6, pos[1] - 8])  # Center the coordinates
        
        # Calculate rotation angle based on player index
        # Player 0 (red) is at top, rotate clockwise for others
        angle = (player_idx * np.pi / 3)  # 60 degrees per player
        
        # Rotation matrix
        rotation = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        
        # Apply rotation
        rotated = rotation @ pos_array
        
        # Convert back to board coordinates
        x = np.round(rotated[0] + 6)  # Re-center
        y = np.round(rotated[1] + 8)
        
        return (x, y)
    
    def is_piece_trapped(self, board: Board, pos: Union[tuple, torch.Tensor], player_positions: Set[tuple]) -> bool:
        """
        Check if a piece is trapped (no valid moves).
        
        Args:
            board: The game board
            pos: Position to check (tuple or tensor)
            player_positions: Set of all player positions
            
        Returns:
            bool: True if the piece is trapped
        """
        try:
            # Convert position to tensor properly
            if isinstance(pos, tuple):
                pos_tensor = torch.tensor(list(pos), dtype=torch.float32)
            elif isinstance(pos, torch.Tensor) and len(pos.shape) == 1:
                pos_tensor = pos.clone().detach()
            else:
                raise ValueError(f"Invalid position type: {type(pos)}")
                
            # Get all possible moves for this piece
            moves = board.moves(pos_tensor)
            
            # Handle both list and tensor return types
            if isinstance(moves, list):
                return len(moves) == 0
            else:
                # For tensor type, check if it's empty using numel()
                return moves.numel() == 0 if moves is not None else True
        except Exception as e:
            print(f"Error in is_piece_trapped: {e}")
            return True  # Assume trapped if there's an error
    
    def is_back_row(self, pos: Union[tuple, torch.Tensor], player_positions: Set[tuple], player_idx: int) -> bool:
        """
        Check if a position is in the back row relative to other pieces.
        
        Args:
            pos: Position to check (tuple or tensor)
            player_positions: Set of all player positions
            player_idx: Player index for perspective adjustment
            
        Returns:
            bool: True if the position is in the back row
        """
        try:
            # Convert position to tuple if it's a tensor
            if isinstance(pos, torch.Tensor):
                pos = tuple(map(int, pos.tolist()))
            elif not isinstance(pos, tuple):
                raise ValueError(f"Invalid position type: {type(pos)}")
                
            # Adjust all positions for player perspective
            adjusted_pos = self.adjust_position_for_player(pos, player_idx)
            adjusted_positions = {self.adjust_position_for_player(p, player_idx) 
                                for p in player_positions}
            
            # Count how many pieces are ahead of this one from player's perspective
            pieces_ahead = sum(1 for p in adjusted_positions if p[1] > adjusted_pos[1])
            return pieces_ahead >= len(player_positions) - 2
        except Exception as e:
            print(f"Error in is_back_row: {e}")
            return False  # Assume not in back row if there's an error
    
    def evaluate_position(self, board: Board, player_idx: int) -> float:
        """
        Evaluate the current board position for the given player.
        Higher scores are better.
        
        Args:
            board: The game board to evaluate
            player_idx: The player's index to evaluate for
            
        Returns:
            float: Score for the position
        """
        score = 0.0
        player_positions = set(board.player_keys(player_idx))
        goal_positions = set(board.winning_keys(player_idx))
        
        if not player_positions:
            return float('-inf')  # Lost all pieces
            
        # Check for win
        if player_positions == goal_positions:
            return float('inf')
        
        # Calculate distance-based score
        total_distance = 0
        min_y = float('inf')
        max_y = float('-inf')
        min_x = float('inf')
        max_x = float('-inf')
        
        # Track trapped and back row pieces
        trapped_pieces = 0
        back_row_pieces = 0
        
        # Adjust all positions for player's perspective
        adjusted_positions = {self.adjust_position_for_player(pos, player_idx) 
                            for pos in player_positions}
        adjusted_goals = {self.adjust_position_for_player(pos, player_idx) 
                         for pos in goal_positions}
        
        for pos in player_positions:
            adjusted_pos = self.adjust_position_for_player(pos, player_idx)
            
            # Track piece spread from player's perspective
            min_y = min(min_y, adjusted_pos[1])
            max_y = max(max_y, adjusted_pos[1])
            min_x = min(min_x, adjusted_pos[0])
            max_x = max(max_x, adjusted_pos[0])
            
            # Calculate minimum distance to any goal position from player's perspective
            min_goal_dist = min(
                abs(adjusted_pos[0] - goal[0]) + abs(adjusted_pos[1] - goal[1])
                for goal in adjusted_goals
            )
            total_distance += min_goal_dist
            
            # Check for trapped pieces
            if self.is_piece_trapped(board, pos, player_positions):
                trapped_pieces += 1
                
            # Check for back row pieces from player's perspective
            if self.is_back_row(pos, player_positions, player_idx):
                back_row_pieces += 1
        
        # Penalize spreading out too much horizontally (but less than before)
        spread_penalty = (max_x - min_x) * 0.3
        
        # Reward vertical progress (increased weight)
        progress_bonus = (max_y - min_y) * 3.0
        
        # Pieces in goal bonus
        pieces_in_goal = len(player_positions.intersection(goal_positions))
        goal_bonus = pieces_in_goal * 10.0
        
        # Heavy penalty for trapped pieces
        trapped_penalty = trapped_pieces * -15.0
        
        # Penalty for having pieces in back row
        back_row_penalty = back_row_pieces * -8.0
        
        # Calculate final score
        score = (
            -total_distance * 0.8  # Slightly reduced weight on distance
            - spread_penalty
            + progress_bonus
            + goal_bonus
            + trapped_penalty
            + back_row_penalty
        )
        
        return score
    
    def evaluate_move(self, board: Board, from_pos: Union[tuple, torch.Tensor], to_pos: Union[tuple, torch.Tensor]) -> float:
        """
        Evaluate a single move's quality.
        
        Args:
            board: The game board
            from_pos: Starting position (tuple or tensor)
            to_pos: Ending position (tuple or tensor)
            
        Returns:
            float: Score for the move
        """
        try:
            # Convert positions to tensors if they're tuples
            if isinstance(from_pos, tuple):
                from_pos = torch.tensor(list(from_pos), dtype=torch.float32)
            elif isinstance(from_pos, torch.Tensor) and len(from_pos.shape) == 1:
                from_pos = from_pos.clone().detach()
            else:
                raise ValueError(f"Invalid from_pos type: {type(from_pos)}")
                
            if isinstance(to_pos, tuple):
                to_pos = torch.tensor(list(to_pos), dtype=torch.float32)
            elif isinstance(to_pos, torch.Tensor) and len(to_pos.shape) == 1:
                to_pos = to_pos.clone().detach()
            else:
                raise ValueError(f"Invalid to_pos type: {type(to_pos)}")
                
            # Create a copy of the board and make the move
            new_board = board.move_new(from_pos, to_pos, check=False)
            
            # Get the base position score
            score = self.evaluate_position(new_board, board.playing[1])
            
            # Convert positions to tuples for distance calculations
            from_tuple = tuple(map(int, from_pos.tolist()))
            to_tuple = tuple(map(int, to_pos.tolist()))
            
            # Add bonus for jumps (encourages capturing opportunities)
            if abs(from_tuple[0] - to_tuple[0]) > 1 or abs(from_tuple[1] - to_tuple[1]) > 1:
                score += 5.0
            
            # Adjust positions for player's perspective
            adjusted_from = self.adjust_position_for_player(from_tuple, board.playing[1])
            adjusted_to = self.adjust_position_for_player(to_tuple, board.playing[1])
            
            # Add bonus for forward progress (increased for back pieces)
            y_progress = adjusted_to[1] - adjusted_from[1]
            if y_progress > 0:  # Moving forward from player's perspective
                # Higher bonus for pieces starting from back rows
                if self.is_back_row(from_tuple, set(board.player_keys(board.playing[1])), board.playing[1]):
                    score += y_progress * 4.0  # Extra bonus for moving back pieces forward
                else:
                    score += y_progress * 2.0
            
            # Add bonus for moving out of trapped positions
            if self.is_piece_trapped(board, from_tuple, set(board.player_keys(board.playing[1]))):
                score += 10.0  # High bonus for escaping trapped positions
                
            # Add repetition penalty
            score += self.get_repetition_penalty(from_tuple, to_tuple, board.playing[1])
            
            # Add small random factor to break ties (scaled by score magnitude)
            score += np.random.normal(0, 0.1) * abs(score + 1e-10)
            
            return score
            
        except Exception as e:
            print(f"Error in evaluate_move: {e}")
            return float('-inf')
    
    def select_best_move(self, board: Board) -> Tuple[tuple, tuple]:
        """
        Select the best move for the current position.
        
        Args:
            board: The current game board
            
        Returns:
            Tuple of (from_position, to_position) representing the selected move
        """
        best_move = None
        best_score = float('-inf')
        
        # Get all valid moves
        all_moves = board.all_moves()
        if not all_moves:
            raise ValueError("No valid moves available")
            
        for from_tensor, to_tensors in all_moves:
            # Skip empty positions
            if torch.all(from_tensor == 0):
                continue
                
            # Keep original tensor for board operations
            from_pos = from_tensor.clone().detach()
                
            for to_tensor in to_tensors:
                # Skip empty moves
                if torch.all(to_tensor == 0):
                    continue
                    
                # Keep original tensor for board operations
                to_pos = to_tensor.clone().detach()
                    
                try:
                    score = self.evaluate_move(board, from_pos, to_pos)
                    
                    if score > best_score:
                        best_score = score
                        # Convert to tuples only for storage
                        best_move = (
                            tuple(map(int, from_pos.tolist())),
                            tuple(map(int, to_pos.tolist()))
                        )
                except Exception as e:
                    print(f"Error evaluating move {from_pos.tolist()} -> {to_pos.tolist()}: {e}")
                    continue
        
        if best_move is None:
            raise ValueError("No valid moves available")
            
        # Update move history
        self.update_move_history(board.playing[1], best_move[0], best_move[1])
            
        return best_move
    
    def process(self, delta: float):
        """
        Process the controller's turn in the game.
        
        Args:
            delta: Time elapsed since last process call
        """
        if self.board.winner is None:
            if Color(self.board.playing[1]) in self.players:
                try:
                    move = self.select_best_move(self.board)
                    # Convert tuples to tensors for board.move
                    from_pos = torch.tensor(list(move[0]), dtype=torch.float32)
                    to_pos = torch.tensor(list(move[1]), dtype=torch.float32)
                    self.board.move(from_pos, to_pos)
                except Exception as e:
                    print(f"Error during heuristic move: {e}") 