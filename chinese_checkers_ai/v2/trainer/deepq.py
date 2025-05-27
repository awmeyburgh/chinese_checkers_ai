from pathlib import Path
import random
import math
from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
from functools import lru_cache
import time
import traceback

from chinese_checkers_ai.v2.model.board import Board
from chinese_checkers_ai.v2.model.color import Color

@lru_cache(maxsize=1024)
def get_goal_positions(player):
    """Cache goal positions for each player to avoid recomputation"""
    return frozenset(board.winning_keys(player))

class ReplayBuffer:
    """Advanced prioritized experience replay buffer with game progress weighting"""
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = 0.001  # Beta increment per sampling
        
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.game_progress = deque(maxlen=capacity)  # Store game progress for each transition
        self.rewards = deque(maxlen=capacity)  # Store rewards for magnitude-based sampling
        self.position = 0
        self.eps = 1e-6  # Small constant to prevent zero probabilities
        
        # Statistics for reward normalization
        self.reward_mean = 0
        self.reward_std = 1
        self.reward_count = 0
    
    def push(self, state: torch.Tensor, action: Tuple, reward: float, 
            next_state: torch.Tensor, done: bool, turn: int = 0, max_turns: int = 100):
        """Store a transition with priority based on game progress and reward magnitude"""
        # Update reward statistics
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        if self.reward_count > 1:
            self.reward_std = ((self.reward_count - 2) * self.reward_std ** 2 + delta * (reward - self.reward_mean)) / (self.reward_count - 1)
            self.reward_std = max(math.sqrt(self.reward_std), 1.0)  # Prevent zero division
        
        # Calculate game progress (0 to 1)
        progress = min(turn / max_turns, 1.0)
        
        # Store transition
        self.buffer.append((state, action, reward, next_state, done))
        self.game_progress.append(progress)
        self.rewards.append(reward)
        
        # Calculate initial priority based on reward magnitude and game progress
        normalized_reward = abs(reward - self.reward_mean) / self.reward_std
        progress_weight = 1.0 + 2.0 * progress  # Later game states get higher weight
        priority = (normalized_reward + 1.0) * progress_weight
        
        # Add to priorities
        self.priorities.append(max(priority, max(self.priorities) if self.priorities else 1.0))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of transitions with importance sampling and game progress weighting"""
        total = len(self.buffer)
        
        # Calculate sampling probabilities
        raw_priorities = torch.tensor(list(self.priorities), dtype=torch.float32)
        progress_weights = torch.tensor(list(self.game_progress), dtype=torch.float32)
        reward_weights = torch.tensor(list(self.rewards), dtype=torch.float32)
        
        # Normalize reward weights
        normalized_rewards = (reward_weights - self.reward_mean) / (self.reward_std + self.eps)
        reward_priorities = torch.exp(normalized_rewards / 2.0)  # Softmax temperature of 2.0
        
        # Combine priorities with progress and reward weights
        combined_priorities = (
            raw_priorities.pow(self.alpha) * 
            (1.0 + progress_weights) * 
            reward_priorities
        )
        
        probs = combined_priorities / (combined_priorities.sum() + self.eps)
        
        # Sample indices based on combined priorities
        try:
            indices = torch.multinomial(probs, batch_size, replacement=True)
        except RuntimeError:
            # Fallback to uniform sampling if numerical issues
            indices = torch.randint(0, len(self.buffer), (batch_size,))
        
        # Calculate importance sampling weights
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get samples
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (
            torch.stack(states),
            actions,
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.bool).unsqueeze(1),
            weights.unsqueeze(1),
            indices.tolist()
        )
    
    def update_priorities(self, indices: List[int], td_errors: torch.Tensor):
        """Update priorities based on TD errors and game progress"""
        for idx, error in zip(indices, td_errors):
            if idx < len(self.priorities):  # Safety check
                progress = self.game_progress[idx]
                reward = self.rewards[idx]
                
                # Scale priority by game progress and reward magnitude
                normalized_reward = abs(reward - self.reward_mean) / (self.reward_std + self.eps)
                progress_weight = 1.0 + 2.0 * progress
                priority = ((abs(error.item()) + self.eps) ** self.alpha) * progress_weight * (normalized_reward + 1.0)
                
                self.priorities[idx] = priority
    
    def __len__(self) -> int:
        return len(self.buffer)

class ChineseCheckersNet(nn.Module):
    """Neural network for Chinese Checkers with attention and residual connections"""
    def __init__(self, board_size: int = 132):  # 6x11x2 = 132 elements when flattened
        super().__init__()
        self.board_size = board_size
        self.hidden_dim = 256  # Make hidden_dim an instance variable
        
        # Input processing
        self.input_norm = nn.LayerNorm(board_size)
        self.input_proj = nn.Sequential(
            nn.Linear(board_size, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(self.hidden_dim) for _ in range(3)
        ])
        
        # Self-attention for board state
        self.attention = nn.MultiheadAttention(self.hidden_dim, num_heads=4, batch_first=True)
        self.post_attention_norm = nn.LayerNorm(self.hidden_dim)
        
        # Move encoder
        self.move_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Value stream with residual connections
        self.value_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        # Advantage stream with residual connections
        self.advantage_net = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        # Initialize with smaller weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor, valid_moves: List[Tuple]) -> torch.Tensor:
        batch_size = state.size(0)
        num_moves = len(valid_moves)
        
        if num_moves == 0:
            return torch.zeros((batch_size, 1), dtype=torch.float32, requires_grad=True)
        
        # Ensure state is properly shaped and flattened
        if len(state.shape) == 4:
            state = state.view(batch_size, -1)
        elif len(state.shape) == 2 and state.size(1) != self.board_size:
            raise ValueError(f"Expected flattened state size of {self.board_size}, got {state.size(1)}")
        
        # Process input
        state = self.input_norm(state)
        x = self.input_proj(state)
        
        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Apply self-attention
        x_attn = x.unsqueeze(1)  # Add sequence length dimension
        attn_out, _ = self.attention(x_attn, x_attn, x_attn)
        x = x + attn_out.squeeze(1)  # Residual connection
        x = self.post_attention_norm(x)
        
        # Get state value
        state_value = self.value_net(x)
        
        # Process moves
        move_tensors = torch.stack([torch.tensor([
            from_pos[0] / 5.0,
            from_pos[1] / 5.0,
            to_pos[0] / 5.0,
            to_pos[1] / 5.0
        ], dtype=torch.float32) for from_pos, to_pos in valid_moves])
        
        move_encodings = self.move_encoder(move_tensors)  # [num_moves, hidden_dim]
        
        # Combine state with moves
        state_expanded = x.unsqueeze(1).expand(-1, num_moves, -1)
        moves_expanded = move_encodings.unsqueeze(0).expand(batch_size, -1, -1)
        combined = torch.cat([state_expanded, moves_expanded], dim=2)
        combined = combined.view(-1, self.hidden_dim * 2)
        
        # Calculate advantages
        advantages = self.advantage_net(combined).view(batch_size, num_moves)
        
        # Combine value and advantages (dueling architecture)
        q_values = state_value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(0.1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)  # Residual connection

def calculate_loss(current_q_values: torch.Tensor, target_q_values: torch.Tensor) -> torch.Tensor:
    """Optimized and numerically stable loss calculation"""
    # Ensure inputs have the same shape
    if current_q_values.shape != target_q_values.shape:
        current_q_values = current_q_values.view_as(target_q_values)
    
    # Clip values to prevent extreme differences
    current_q_values = torch.clamp(current_q_values, -100, 100)
    target_q_values = torch.clamp(target_q_values, -100, 100)
    
    # Calculate TD errors with clipping
    td_errors = torch.clamp(current_q_values - target_q_values, -10, 10)
    
    # Huber loss with safe handling
    loss = torch.where(
        torch.abs(td_errors) < 1.0,
        0.5 * td_errors.pow(2),
        torch.abs(td_errors) - 0.5
    )
    
    # Handle any remaining NaN values
    loss = torch.nan_to_num(loss, 0.0)
    
    # Return mean if we have valid losses, otherwise return zero
    mean_loss = loss.mean()
    return torch.where(torch.isnan(mean_loss), torch.tensor(0.0), mean_loss)

def calculate_reward(board: Board, player: int, done: bool) -> float:
    """Calculate reward for the current state with emphasis on exploration and progress"""
    # Get piece positions
    goal_positions = set(board.winning_keys(player))
    current_positions = set(board.player_keys(player))
    
    if not current_positions:
        return -5.0  # Severe penalty for losing pieces
    
    # Calculate base metrics
    pieces_in_goal = len(goal_positions.intersection(current_positions))
    total_pieces = len(current_positions)
    
    # Calculate position-based metrics
    y_coords = [pos[1] for pos in current_positions]
    x_coords = [pos[0] for pos in current_positions]
    
    # Progress metrics
    max_y = max(y_coords)
    min_y = min(y_coords)
    avg_y = sum(y_coords) / len(y_coords)
    
    # Spread metrics
    x_spread = max(x_coords) - min(x_coords)
    y_spread = max_y - min_y
    
    # Distance to goal for each piece
    distances_to_goal = []
    for pos in current_positions:
        min_goal_dist = min(abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]) for goal in goal_positions)
        distances_to_goal.append(min_goal_dist)
    avg_distance = sum(distances_to_goal) / len(distances_to_goal)
    
    # Normalize metrics
    normalized_progress = avg_y / 10.0
    normalized_x_spread = x_spread / 5.0
    normalized_y_spread = y_spread / 10.0
    normalized_distance = avg_distance / 15.0  # Normalize by max possible distance
    
    # Calculate rewards
    progress_reward = (1.0 - normalized_distance) * 2.0  # Higher weight on progress
    formation_bonus = 1.0 - (normalized_x_spread + normalized_y_spread) / 3.0
    goal_occupation = (pieces_in_goal / total_pieces) * 3.0  # Higher weight on reaching goals
    
    # Movement diversity bonus (encourage exploring different board areas)
    diversity_bonus = 0.0
    if hasattr(board, 'last_positions'):
        new_positions = current_positions - board.last_positions
        diversity_bonus = len(new_positions) * 0.5
    board.last_positions = current_positions
    
    # Combine all rewards
    reward = (
        progress_reward +
        formation_bonus +
        goal_occupation +
        diversity_bonus
    )
    
    # Terminal state rewards
    if done:
        if board.winner and board.winner.value == player:
            return 10.0  # Significant win reward
        elif board.winner:
            return -5.0  # Significant loss penalty
    
    # Clip reward to reasonable range
    return max(min(reward, 5.0), -5.0)

def get_valid_actions(board: Board) -> List[Tuple[tuple, tuple]]:
    """Get list of valid moves in (from, to) format"""
    actions = []
    for fro, tos in board.all_moves():
        for to in tos:
            actions.append((fro, to))
    return actions

def find_move_index(moves: List[Tuple[torch.Tensor, torch.Tensor]], target_move: Tuple[torch.Tensor, torch.Tensor]) -> int:
    """Find the index of a move in a list of moves using tensor comparison"""
    target_from, target_to = target_move
    for idx, (from_pos, to_pos) in enumerate(moves):
        if torch.all(from_pos == target_from) and torch.all(to_pos == target_to):
            return idx
    return -1

def run_match(
    versus: int,
    policy: ChineseCheckersNet,
    target: ChineseCheckersNet,
    replay_buffer: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    max_turns: int = 100,
    batch_size: int = 32,
    gamma: float = 0.99,
    epsilon: float = 0.1
) -> Tuple[float, float]:
    # Initialize board with specified number of players
    players = []
    for i in range(versus//2):
        players.extend([Color(i), Color(i+3)])
    board = Board.new(players, rotate=True)
    
    total_loss = 0.0
    num_steps = 0
    total_reward = 0.0
    
    # Initialize temperature for Boltzmann exploration
    temperature = 1.0
    min_temperature = 0.1
    temperature_decay = 0.995

    for turn in range(1, max_turns+1):
        for player_idx in range(versus):
            # Ensure it's the correct player's turn
            if board.playing[1] != player_idx:
                board.rotate(1)
                continue
                
            # Get valid actions
            valid_actions = get_valid_actions(board)
            if not valid_actions:
                continue
            
            # Get current state
            current_state = board.state.clone()
            if not isinstance(current_state, torch.Tensor):
                current_state = torch.tensor(current_state, dtype=torch.float32)
            current_state = current_state.unsqueeze(0)
            
            # Get Q-values and apply exploration
            with torch.no_grad():
                q_values = policy(current_state, valid_actions).squeeze()
                q_values = torch.clamp(q_values, -100, 100)
                
                # Dynamic noise scaling based on game progress
                progress_factor = turn / max_turns
                noise_scale = 1.0 - 0.5 * progress_factor
                noise = torch.randn_like(q_values) * noise_scale
                q_values = q_values + noise
                
                # Multi-strategy exploration with dynamic probabilities
                if random.random() < epsilon:
                    strategy_choice = random.random()
                    
                    if strategy_choice < 0.3:  # 30% pure random
                        action_idx = random.randrange(len(valid_actions))
                    
                    elif strategy_choice < 0.6:  # 30% goal-directed random
                        # Find moves that generally progress toward goal
                        progress_scores = []
                        for from_pos, to_pos in valid_actions:
                            y_progress = to_pos[1] - from_pos[1]
                            progress_scores.append(max(y_progress, 0))
                        
                        # Convert to probabilities
                        scores = torch.tensor(progress_scores, dtype=torch.float32)
                        if scores.sum() > 0:
                            probs = scores / scores.sum()
                            action_idx = torch.multinomial(probs, 1).item()
                        else:
                            action_idx = random.randrange(len(valid_actions))
                    
                    elif strategy_choice < 0.8:  # 20% temperature-based
                        # Adaptive temperature based on game progress
                        adaptive_temp = temperature * (1 + progress_factor)
                        q_values = q_values - q_values.max()
                        scaled_values = q_values / adaptive_temp
                        exp_values = torch.exp(scaled_values)
                        probs = exp_values / (exp_values.sum() + 1e-10)
                        try:
                            action_idx = torch.multinomial(probs, 1).item()
                        except RuntimeError:
                            action_idx = random.randrange(len(valid_actions))
                    
                    else:  # 20% novelty-based
                        # Prefer moves to positions not recently visited
                        novelty_scores = []
                        for _, to_pos in valid_actions:
                            if hasattr(board, 'position_history'):
                                score = 1.0 / (1.0 + board.position_history.count(to_pos))
                            else:
                                score = 1.0
                            novelty_scores.append(score)
                        
                        scores = torch.tensor(novelty_scores, dtype=torch.float32)
                        probs = scores / (scores.sum() + 1e-10)
                        action_idx = torch.multinomial(probs, 1).item()
                
                else:
                    # Epsilon-greedy with adaptive exploration
                    if random.random() < 0.2:  # 20% chance for alternative actions
                        num_alternatives = min(3, len(valid_actions))
                        _, indices = torch.topk(q_values, num_alternatives)
                        action_idx = indices[random.randrange(num_alternatives)].item()
                    else:
                        action_idx = torch.argmax(q_values).item()
                
                # Update position history
                if not hasattr(board, 'position_history'):
                    board.position_history = []
                board.position_history = (board.position_history + [valid_actions[action_idx][1]])[-10:]
            
            # Take action
            action = valid_actions[action_idx]
            old_state = board.state.clone()
            if not isinstance(old_state, torch.Tensor):
                old_state = torch.tensor(old_state, dtype=torch.float32)
            
            # Store pre-move score
            pre_move_score = calculate_board_score(board, player_idx)
            
            # Make move and rotate board
            board.move(*action)
            board.rotate(1)
            
            # Calculate immediate reward
            post_move_score = calculate_board_score(board, player_idx)
            score_diff = (post_move_score - pre_move_score) / 25.0
            
            # Calculate reward with additional components
            is_done = board.winner is not None
            reward = calculate_reward(board, player_idx, is_done)
            
            # Add move quality bonus/penalty
            if score_diff > 0:
                reward += 1.0
            elif score_diff < 0:
                reward -= 1.0
            
            # Get next state
            next_state = board.state.clone()
            if not isinstance(next_state, torch.Tensor):
                next_state = torch.tensor(next_state, dtype=torch.float32)
            
            # Store transition with game progress information
            replay_buffer.push(
                old_state,
                action,
                torch.tensor(reward, dtype=torch.float32),
                next_state,
                torch.tensor(is_done, dtype=torch.bool),
                turn=turn,
                max_turns=max_turns
            )
            
            # Train on batch if enough samples
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones, weights, indices = replay_buffer.sample(batch_size)
                
                # Calculate current Q-values first
                current_q_values = torch.zeros((batch_size, 1), dtype=torch.float32, requires_grad=True)
                for j in range(batch_size):
                    current_board = Board(states[j].detach().clone())
                    current_valid_moves = get_valid_actions(current_board)
                    if current_valid_moves:
                        action_idx = find_move_index(current_valid_moves, actions[j])
                        if action_idx != -1:
                            current_q = policy(states[j].unsqueeze(0), current_valid_moves)
                            current_q_values[j] = current_q[0, action_idx]
                
                # Calculate target Q-values
                with torch.no_grad():
                    next_q_values = torch.zeros((batch_size, 1), dtype=torch.float32)
                    for j in range(batch_size):
                        next_board = Board(next_states[j].detach().clone())
                        next_valid_moves = get_valid_actions(next_board)
                        if next_valid_moves:
                            next_state_q = policy(next_states[j].unsqueeze(0), next_valid_moves)
                            best_action_idx = torch.argmax(next_state_q)
                            target_next_q = target(next_states[j].unsqueeze(0), next_valid_moves)
                            next_q_values[j] = target_next_q[0, best_action_idx]
                    
                    target_q_values = rewards + gamma * next_q_values * (~dones)
                
                # Calculate TD errors for priority update
                td_errors = torch.abs(current_q_values - target_q_values.detach())
                
                # Update priorities in replay buffer
                replay_buffer.update_priorities(indices, td_errors)
                
                # Calculate loss with importance sampling weights
                elementwise_loss = F.smooth_l1_loss(current_q_values, target_q_values.detach(), reduction='none')
                loss = (weights * elementwise_loss).mean()
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                num_steps += 1
                
                # Print debugging info occasionally
                if num_steps % 10 == 0:
                    with torch.no_grad():
                        grad_norms = [p.grad.norm().item() for p in policy.parameters() if p.grad is not None]
                        print(f"Debug - Q-values: {current_q_values.mean().item():.4f}, "
                              f"Target: {target_q_values.mean().item():.4f}, "
                              f"Loss: {loss.item():.4f}, "
                              f"Grad norms: min={min(grad_norms):.4f}, max={max(grad_norms):.4f}")
            
            # Update temperature
            temperature = max(min_temperature, temperature * temperature_decay)
            
            # Accumulate rewards
            total_reward += reward
            
            if board.winner is not None:
                break
        if board.winner is not None:
            break
    
    # Calculate final score and average loss
    final_score = total_reward / max(num_steps, 1)
    avg_loss = total_loss / max(num_steps, 1)
    
    return final_score, avg_loss

def run_epoch(
    matches: int = 20,
    network: Optional[ChineseCheckersNet]=None,
    max_turns: int = 100,
    update: int = 20,
    tau: float = 0.005,
    lr: float = 1e-4,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    batch_size: int = 32,
    target_network: Optional[ChineseCheckersNet]=None,
    optimizer: Optional[torch.optim.Optimizer]=None,
    use_compile: bool = False
) -> Tuple[float, float]:
    policy = network or ChineseCheckersNet()
    target = target_network or ChineseCheckersNet()
    if not target_network:
        target.load_state_dict(policy.state_dict())
    
    if not optimizer:
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    
    # Create and pre-fill replay buffer
    replay_buffer = ReplayBuffer(capacity=50000)
    min_buffer_size = batch_size * 4  # Minimum samples before training starts
    
    print("Pre-filling replay buffer...")
    while len(replay_buffer) < min_buffer_size:
        # Run a quick match to gather samples
        board = Board.new([Color(0), Color(3)], rotate=True)
        
        for _ in range(10):  # Collect 10 moves per pre-fill match
            valid_actions = get_valid_actions(board)
            if not valid_actions:
                break
                
            # Random action for pre-filling
            action = random.choice(valid_actions)
            old_state = board.state.clone()
            if not isinstance(old_state, torch.Tensor):
                old_state = torch.tensor(old_state, dtype=torch.float32)
            
            # Make move
            board.move(*action)
            board.rotate(1)
            
            # Calculate reward
            is_done = board.winner is not None
            reward = calculate_reward(board, board.playing[1], is_done)
            
            # Store transition
            next_state = board.state.clone()
            if not isinstance(next_state, torch.Tensor):
                next_state = torch.tensor(next_state, dtype=torch.float32)
            
            replay_buffer.push(
                old_state,
                action,
                torch.tensor(reward, dtype=torch.float32),
                next_state,
                torch.tensor(is_done, dtype=torch.bool)
            )
            
            if is_done:
                break
    
    print(f"Replay buffer pre-filled with {len(replay_buffer)} samples")
    
    total_score = 0
    total_loss = 0
    training_steps = 0

    for match in range(1, matches + 1):
        score, loss = run_match(
            2,  # Start with 2 players
            policy,
            target,
            replay_buffer,
            optimizer,
            max_turns=max_turns,
            batch_size=batch_size,
            gamma=gamma,
            epsilon=epsilon
        )
        
        total_score += score
        total_loss += loss
        training_steps += 1

        # Update target network
        if use_compile:
            source_state_dict = policy._orig_mod.state_dict()
            target_state_dict = target._orig_mod.state_dict()
        else:
            source_state_dict = policy.state_dict()
            target_state_dict = target.state_dict()
            
        for key in source_state_dict:
            target_state_dict[key] = source_state_dict[key]*tau + target_state_dict[key]*(1-tau)
            
        if use_compile:
            target._orig_mod.load_state_dict(target_state_dict)
        else:
            target.load_state_dict(target_state_dict)

        if (match-1)%update == update-1:
            avg_score = total_score/match
            avg_loss = total_loss/max(training_steps, 1)
            print(f"Match {match}: Score={avg_score:.2f}, Loss={avg_loss:.4f}")

    return total_score/matches, total_loss/max(training_steps, 1)

def run(
    epochs: int = 100,
    matches: int = 40,
    max_turns: int = 10,
    update: int = 2,
    tau: float = 0.005,
    lr: float = 2e-4,  # Reduced learning rate
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,  # Lower minimum epsilon
    epsilon_decay: float = 0.9995,  # Slower decay
    num_workers: int = 4
):
    print("Initializing training environment...")
    
    # Load or create network
    filename = Path('deepq.network')
    if filename.exists():
        print("Loading existing model...")
        network = ChineseCheckersNet()
        network.load_state_dict(torch.load(filename))
        print("Model loaded successfully")
    else:
        print("Creating new model...")
        network = ChineseCheckersNet()
        print("Model created successfully")
    
    # Create target network
    print("Creating target network...")
    target_network = ChineseCheckersNet()
    target_network.load_state_dict(network.state_dict())
    print("Target network created")
    
    # Create optimizer with learning rate schedule
    print("Setting up optimizer...")
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    print(f"Initial learning rate: {lr}")
    
    print("Setup complete, starting training...")
    epsilon = epsilon_start
    min_epsilon = epsilon_end
    
    # Initialize curriculum learning parameters
    curriculum_stage = 1
    stage_performance = []
    stage_threshold = 0.7  # Performance threshold to advance curriculum
    
    # Training loop with curriculum learning
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}:")
        print("=" * 20)
        
        # Adjust max_turns based on curriculum stage
        current_max_turns = max_turns * curriculum_stage
        
        # Periodically reset exploration to encourage new strategies
        if epoch % 20 == 0:
            epsilon = min(epsilon_start, epsilon * 2.0)
            print(f"Resetting epsilon to {epsilon:.3f} to encourage exploration")
        
        score, loss = run_epoch(
            matches=matches,
            network=network,
            max_turns=current_max_turns,
            update=update,
            tau=tau,
            lr=optimizer.param_groups[0]['lr'],
            gamma=gamma,
            epsilon=epsilon,
            batch_size=32,
            target_network=target_network,
            optimizer=optimizer,
            use_compile=False
        )
        
        # Update learning rate based on loss
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"Learning rate adjusted: {old_lr:.2e} -> {new_lr:.2e}")
        
        # Decay epsilon with a minimum value
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        # Track performance for curriculum advancement
        stage_performance.append(score)
        if len(stage_performance) >= 5:  # Check last 5 episodes
            avg_performance = sum(stage_performance[-5:]) / 5
            if avg_performance > stage_threshold:
                curriculum_stage += 1
                stage_performance = []
                print(f"Advancing to curriculum stage {curriculum_stage}")
                # Increase exploration temporarily when advancing stages
                epsilon = min(epsilon * 1.5, 0.5)
        
        print(f'Epoch {epoch}: Score={score:.2f}, Loss={loss:.4f}, Epsilon={epsilon:.3f}, Stage={curriculum_stage}')
        
        # Save model periodically
        if epoch % 10 == 0:
            print("Saving model checkpoint...")
            torch.save(network.state_dict(), filename)
            print("Model saved")

def calculate_board_score(board: Board, player=None) -> float:
    """Calculate score for a player's position"""
    total_score = 0
    player = board.playing[1] if player is None else player
    steps = board.rotate(player=player)
    
    for key in board.player_keys(player):
        y = max(key[1]/2+5, 0)
        total_score += y*(y+1)
    
    if board.players_keys == board.winning_keys(player):
        total_score += 500
    board.rotate(-steps)
    
    return total_score

class ChineseCheckersTrainer:
    """Trainer class for Chinese Checkers AI"""
    def __init__(self, 
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 batch_size: int = 32,
                 target_update_freq: int = 10):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = ChineseCheckersNet().to(self.device)
        self.target_net = ChineseCheckersNet().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer()
        
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps = 0
    
    def select_action(self, state: torch.Tensor, valid_moves: List[Tuple]) -> Tuple:
        """Select an action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        
        with torch.no_grad():
            q_values = self.policy_net(state.unsqueeze(0), valid_moves)
            action_idx = q_values.argmax().item()
            return valid_moves[action_idx]
    
    def train_step(self, board: Board) -> Tuple[float, bool]:
        """Perform one step of training"""
        state = torch.tensor(board.state.flatten(), dtype=torch.float32)
        valid_moves = get_valid_actions(board)
        
        if not valid_moves:
            return 0.0, True
        
        # Select and perform action
        action = self.select_action(state, valid_moves)
        board.move(*action)
        
        # Observe new state
        next_state = torch.tensor(board.state.flatten(), dtype=torch.float32)
        done = board.winner is not None
        reward = calculate_reward(board, board.playing[1], done)
        
        # Store transition
        self.memory.push(state, action, reward, next_state, done)
        
        # Perform optimization
        loss = self.optimize_model()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss, done
    
    def optimize_model(self) -> float:
        """Perform one step of optimization"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample transitions
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        total_loss = 0.0
        num_valid = 0
        
        for i in range(self.batch_size):
            state = states[i].unsqueeze(0)
            next_state = next_states[i].unsqueeze(0)
            action = actions[i]
            
            # Get valid moves for current and next states
            current_board = Board(state.squeeze().numpy())
            next_board = Board(next_state.squeeze().numpy())
            current_valid_moves = get_valid_actions(current_board)
            next_valid_moves = get_valid_actions(next_board)
            
            if not current_valid_moves:
                continue
            
            # Find index of the taken action using tensor comparison
            action_idx = find_move_index(current_valid_moves, action)
            if action_idx == -1:
                continue
            
            # Get Q-values for current state
            current_q_values = self.policy_net(state, current_valid_moves)
            current_q_value = current_q_values[0, action_idx]
            
            # Calculate target Q-value
            with torch.no_grad():
                if next_valid_moves:
                    next_q_values = self.target_net(next_state, next_valid_moves)
                    next_q = next_q_values.max()
                else:
                    next_q = torch.tensor(0.0)
                target_q = rewards[i] + self.gamma * next_q * (1 - dones[i].float())
            
            # Compute loss
            loss = F.smooth_l1_loss(current_q_value.unsqueeze(0), target_q.unsqueeze(0))
            total_loss += loss
            num_valid += 1
        
        if num_valid > 0:
            # Average loss
            loss = total_loss / num_valid
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            return loss.item()
        
        return 0.0
    
    def save(self, path: str):
        """Save the policy network"""
        torch.save(self.policy_net.state_dict(), path)
    
    def load(self, path: str):
        """Load the policy network"""
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Maintain backward compatibility
DeepQNetwork = ChineseCheckersNet


        
