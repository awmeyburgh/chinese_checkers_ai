import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Tuple, List, Optional
import os

from chinese_checkers_ai.v3.model.board import Board
from chinese_checkers_ai.v3.model.player import Player
from chinese_checkers_ai.v3.model.move import Move
from chinese_checkers_ai.v3.model.position import Position


class QNetwork(nn.Module):
    def __init__(self, num_positions: int):
        super().__init__()
        
        # Number of positions on the board (length of flattened board state)
        self.num_positions = num_positions
        
        # Input: 8 channels (7 for pieces, 1 for valid moves)
        self.conv1 = nn.Conv1d(8, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * num_positions, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_positions)  # Output Q-value for each position
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, 8, num_positions)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: torch.Tensor, action_idx: int, reward: float, 
             next_state: torch.Tensor, done: bool, valid_moves_mask: torch.Tensor):
        self.buffer.append((state, action_idx, reward, next_state, done, valid_moves_mask))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)


class DeepQTrainer:
    def __init__(self, 
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,      # Higher minimum exploration
                 epsilon_decay: float = 0.9995,  # Much slower decay
                 buffer_size: int = 10000,
                 batch_size: int = 32,
                 target_update: int = 100):
        
        # Get number of positions from board's position mapping
        self.num_positions = len(Board.get_position_to_idx())
        
        # Networks
        self.q_network = QNetwork(self.num_positions)
        self.target_network = QNetwork(self.num_positions)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Training parameters
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        
        # Training tracking
        self.steps = 0
        self.episodes = 0
        self._episode_rewards = []
        self._episode_losses = []
        
        # Reward normalization
        self.reward_scale = 0.5  # Increased reward scale
        self.reward_running_mean = 0
        self.reward_running_std = 1
        self.reward_alpha = 0.01  # Slower reward stats updates
        
        # Gradient clipping
        self.max_grad_norm = 1.0
    
    def select_action(self, board: Board, player: Player) -> Move:
        """Select an action using epsilon-greedy policy"""
        # Convert board to tensor and get valid moves mask
        state = board.to_tensor(player)
        valid_moves_mask = state[7]  # Channel 7 contains valid moves
        
        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            with torch.no_grad():
                # Get Q-values and mask invalid moves
                state = state.unsqueeze(0)  # Add batch dimension
                q_values = self.q_network(state).squeeze()
                q_values = q_values.to(torch.float32)  # Ensure float32
                q_values[~valid_moves_mask.bool()] = float('-inf')
                
                # Select best valid move
                action_idx = q_values.argmax().item()
        else:
            # Random valid move
            valid_indices = torch.nonzero(valid_moves_mask).squeeze()
            action_idx = valid_indices[random.randint(0, len(valid_indices)-1)].item()
        
        # Convert action index back to move
        pos_to_idx = Board.get_position_to_idx()
        idx_to_pos = {idx: pos for pos, idx in pos_to_idx.items()}
        to_position = idx_to_pos[action_idx]
        
        # Find the piece that can make this move
        for move in board.all_moves(player):
            if move.to_position == to_position:
                return move
                
        raise ValueError("No valid move found for selected position")
    
    def train_step(self) -> Optional[float]:
        """Perform one training step if enough samples are available"""
        if len(self.buffer) < self.batch_size:
            return None
            
        # Sample from replay buffer
        transitions = self.buffer.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        # Prepare batch with explicit dtypes
        state_batch = torch.stack(batch[0])  # Already float32 from board.to_tensor()
        action_batch = torch.tensor(batch[1], dtype=torch.int64)  # Actions are indices
        reward_batch = torch.tensor(batch[2], dtype=torch.float32)
        next_state_batch = torch.stack(batch[3])  # Already float32 from board.to_tensor()
        done_batch = torch.tensor(batch[4], dtype=torch.bool)
        valid_moves_mask_batch = torch.stack(batch[5])  # Already float32 from board.to_tensor()
        
        # Compute current Q values
        current_q_values = self.q_network(state_batch)
        current_q_values = current_q_values.gather(1, action_batch.unsqueeze(1))
        
        # Compute next Q values using Double Q-learning
        with torch.no_grad():
            # Get actions from current network
            next_q_values_online = self.q_network(next_state_batch)
            next_q_values_online[~valid_moves_mask_batch.bool()] = float('-inf')
            next_actions = next_q_values_online.argmax(1, keepdim=True)
            
            # Get Q-values from target network
            next_q_values_target = self.target_network(next_state_batch)
            next_q_values = next_q_values_target.gather(1, next_actions)
            next_q_values[done_batch] = 0.0
            
        # Compute expected Q values
        expected_q_values = reward_batch.unsqueeze(1) + (self.gamma * next_q_values)
        
        # Compute Huber loss (more robust than MSE)
        loss = torch.nn.functional.smooth_l1_loss(current_q_values, expected_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Update target network if needed
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save_model(self, path: str):
        """Save the Q-network model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes
        }, path)
    
    def load_model(self, path: str):
        """Load a saved Q-network model"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
    
    def train(self, board: Board, current_player: Player, episode_reward: float, moves_without_progress: int) -> Tuple[Board, float, int, bool]:
        """Execute one training step for the AI player.
        
        Args:
            board: Current game board
            current_player: Current player (should be AI player)
            episode_reward: Current episode's accumulated reward
            moves_without_progress: Counter for moves without progress
            
        Returns:
            Tuple of (next_board, episode_reward, moves_without_progress, done)
        """
        # Get current state
        state = board.to_tensor(current_player)
        
        # Select and perform action
        move = self.select_action(board, current_player)
        old_distance = self._distance_to_goal(board, current_player)
        
        # Execute move
        next_board = board.move(move)
        new_distance = self._distance_to_goal(next_board, current_player)
        
        # Calculate reward
        reward = self._calculate_reward(board, next_board, current_player, 
                                      old_distance, new_distance)
        episode_reward += reward
        
        # Check if game is done
        done = next_board.has_won(current_player)
        if done:
            reward += 100  # Bonus for winning
        
        # Store transition
        next_state = next_board.to_tensor(current_player)
        self.buffer.push(
            state, 
            Board.get_position_to_idx()[move.to_position],
            reward,
            next_state,
            done,
            state[7]  # Valid moves mask
        )
        
        # Perform training step
        loss = self.train_step()
        if loss is not None:
            self._episode_losses.append(loss)
        
        # Update move counter
        if new_distance >= old_distance:
            moves_without_progress += 1
        else:
            moves_without_progress = 0
            
        return next_board, episode_reward, moves_without_progress, done

    @classmethod
    def run(cls) -> None:
        """Run the training process with default configuration."""
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Training parameters
        num_episodes = 1000
        save_path = "models/deepq_model"
        update_interval = 10  # Print progress every episode
        initial_max_moves = 10  # Start with few moves
        max_moves_cap = 100  # Maximum moves allowed
        
        # Train against 1 opponent initially
        opponent_players = [Player(3)]  # Opposite player
        
        print("Starting DeepQ Training...")
        print(f"Number of episodes: {num_episodes}")
        print(f"Save path: {save_path}")
        print(f"Training against {len(opponent_players)} opponents")
        print(f"Update interval: {update_interval} episodes")
        print(f"Initial max moves: {initial_max_moves}")
        print(f"Max moves cap: {max_moves_cap}")
        print("------------------------")
        
        # Initialize trainer and run training
        trainer = cls()
        rewards, losses, win_rate = trainer.train_loop(
            num_episodes=num_episodes,
            opponent_players=opponent_players,
            save_path=save_path,
            update_interval=update_interval,
            initial_max_moves=initial_max_moves,
            max_moves_cap=max_moves_cap
        )
        
        print("\nTraining Complete!")
        print(f"Final Win Rate: {win_rate*100:.1f}%")
        print(f"Model saved to: {save_path}_final.pt")

    def train_loop(self, num_episodes: int, opponent_players: List[Player], 
                  save_path: Optional[str] = None, update_interval: int = 10,
                  initial_max_moves: int = 10,
                  max_moves_cap: int = 100):
        """Run training for specified number of episodes."""
        # Initialize statistics tracking
        self._episode_rewards = []
        self._episode_losses = []
        wins = 0
        current_max_moves = initial_max_moves
        
        # Performance tracking for move scaling
        initial_window = 20  # Longer initial window
        max_window = 50
        current_window = initial_window
        win_rate_threshold = 0.4  # Higher win rate required
        reward_improvement_threshold = 0.5  # Much higher improvement required
        min_episodes_between_increases = 50  # Minimum episodes between increases
        last_increase_episode = 0
        last_avg_reward = float('-inf')
        moves_increase_count = 0
        
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Training against players: {[p.value for p in opponent_players]}")
        print(f"Initial max moves: {initial_max_moves}")
        print(f"Max moves cap: {max_moves_cap}")
        
        # Track running statistics
        running_reward = 0
        running_loss = 0
        episode_moves = 0
        
        for episode in range(num_episodes):
            # Create new game board with AI (Player 0) and opponents
            all_players = [Player(0)] + opponent_players
            board = Board.new(all_players, random=True)
            player_idx = 0  # Start with AI player
            episode_reward = 0
            moves_without_progress = 0
            episode_moves = 0
            episode_loss = 0
            loss_count = 0
            
            while episode_moves < current_max_moves:
                current_player = all_players[player_idx]
                
                if current_player == Player(0):  # AI's turn
                    board, episode_reward, moves_without_progress, done = self.train(
                        board, current_player, episode_reward, moves_without_progress
                    )
                    episode_moves += 1
                    
                    # Track loss for this move if available
                    if self._episode_losses and self._episode_losses[-1] is not None:
                        episode_loss += self._episode_losses[-1]
                        loss_count += 1
                    
                    if done:
                        wins += 1
                        break
                        
                    # Break if stuck
                    if moves_without_progress > 50:
                        break
                        
                else:  # Opponent's turn
                    # Simple opponent strategy: random valid move
                    valid_moves = board.all_moves(current_player)
                    if not valid_moves:
                        break
                    move = random.choice(valid_moves)
                    board = board.move(move)
                    episode_moves += 1
                    
                    if board.has_won(current_player):
                        break
                
                # Move to next player
                player_idx = (player_idx + 1) % len(all_players)
            
            # Track statistics
            self._episode_rewards.append(episode_reward)
            running_reward = (running_reward * episode + episode_reward) / (episode + 1)
            if loss_count > 0:
                avg_episode_loss = episode_loss / loss_count
                running_loss = (running_loss * episode + avg_episode_loss) / (episode + 1)
            
            # Check if we should increase max_moves
            if (episode > 0 and 
                episode % current_window == 0 and 
                episode - last_increase_episode >= min_episodes_between_increases):
                
                window_start = max(0, episode - current_window)
                recent_wins = sum(1 for i in range(window_start, episode) 
                                if self._episode_rewards[i] > 0)
                recent_win_rate = recent_wins / current_window
                recent_avg_reward = sum(self._episode_rewards[window_start:episode]) / current_window
                
                # More conservative increases
                early_stage = moves_increase_count < 2
                current_win_threshold = win_rate_threshold * (0.75 if early_stage else 1.0)
                current_reward_threshold = reward_improvement_threshold * (0.75 if early_stage else 1.0)
                
                # Only increase if significant improvement
                if (recent_win_rate >= current_win_threshold and 
                    recent_avg_reward > last_avg_reward + current_reward_threshold and
                    current_max_moves < max_moves_cap):
                    
                    old_max_moves = current_max_moves
                    # Smaller increases
                    increase_amount = 5 if early_stage else 3
                    current_max_moves = min(current_max_moves + increase_amount, max_moves_cap)
                    moves_increase_count += 1
                    last_increase_episode = episode
                    
                    print(f"\nIncreasing max moves from {old_max_moves} to {current_max_moves}")
                    print(f"Recent win rate: {recent_win_rate:.2f}")
                    print(f"Recent avg reward: {recent_avg_reward:.2f}")
                    print(f"Episodes since last increase: {episode - last_increase_episode}")
                    
                    # Gradually increase window size
                    if current_window < max_window:
                        current_window = min(current_window + 10, max_window)
                        print(f"Increasing performance window to {current_window} episodes")
                
                last_avg_reward = recent_avg_reward
            
            self.episodes += 1
            
            # Print progress
            if (episode + 1) % update_interval == 0:
                win_rate = wins / (episode + 1) * 100
                print(f"\nEpisode {episode+1}/{num_episodes}")
                print(f"Episode Reward: {episode_reward:.2f} (Running Avg: {running_reward:.2f})")
                if loss_count > 0:
                    print(f"Episode Loss: {avg_episode_loss:.4f} (Running Avg: {running_loss:.4f})")
                print(f"Moves This Episode: {episode_moves}")
                print(f"Current Max Moves: {current_max_moves}")
                print(f"Win Rate: {win_rate:.1f}%")
                print(f"Epsilon: {self.epsilon:.3f}")
                print("------------------------")
            
            # Save checkpoint
            if save_path and (episode + 1) % 100 == 0:
                self.save_model(f"{save_path}_episode_{episode+1}.pt")
            
        # Save final model
        if save_path:
            self.save_model(f"{save_path}_final.pt")
            
        return self._episode_rewards, self._episode_losses, wins / num_episodes
    
    def _distance_to_goal(self, board: Board, player: Player) -> float:
        """Calculate total manhattan distance of player's pieces to goal positions"""
        total_distance = 0
        winning_positions = {pos for pos in board.rotated_winning_group(board.rotation, player).positions}
        
        for pos in board.players[player].positions:
            # Find minimum distance to any goal position
            min_distance = min(abs(pos.euclid[0] - goal.euclid[0]) + 
                             abs(pos.euclid[1] - goal.euclid[1])
                             for goal in winning_positions)
            total_distance += min_distance
            
        return total_distance
    
    def _calculate_reward(self, old_board: Board, new_board: Board, 
                         player: Player, old_distance: float, new_distance: float) -> float:
        """Calculate reward for a move"""
        reward = 0.0
        
        # Distance improvement reward
        distance_improvement = float(old_distance - new_distance)
        reward += distance_improvement * 3.0  # Increased distance reward
        
        # Winning position reward
        if new_board.has_won(player):
            reward += 200.0  # Increased winning reward
            
        # Smaller move penalty
        reward -= 0.05  # Reduced move penalty
        
        # Add bonus for getting pieces to the goal area
        goal_positions = new_board.rotated_winning_group(new_board.rotation, player).positions
        pieces_in_goal = sum(1 for pos in new_board.players[player].positions if pos in goal_positions)
        reward += pieces_in_goal * 1.0  # Bonus for each piece in goal area
        
        # Normalize reward
        self.reward_running_mean = (1 - self.reward_alpha) * self.reward_running_mean + self.reward_alpha * reward
        self.reward_running_std = (1 - self.reward_alpha) * self.reward_running_std + self.reward_alpha * abs(reward - self.reward_running_mean)
        if self.reward_running_std > 0:
            normalized_reward = (reward - self.reward_running_mean) / (self.reward_running_std + 1e-8)
            return normalized_reward * self.reward_scale
        return reward 