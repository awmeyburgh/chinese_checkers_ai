"""
Advanced Deep Q-Learning implementation for Chinese Checkers with dynamic action spaces.
This implementation uses a hybrid architecture combining transformers for board state processing
and action encoding, with a dynamic action masking mechanism.
"""

from pathlib import Path
import random
import math
from typing import List, Optional, Tuple, Dict, Set
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, defaultdict
import numpy as np
from dataclasses import dataclass
import time

from chinese_checkers_ai.v2.model.board import Board
from chinese_checkers_ai.v2.model.color import Color

@dataclass
class Experience:
    state: torch.Tensor
    action: Tuple[tuple, tuple]  # (from_pos, to_pos)
    reward: float
    next_state: torch.Tensor
    done: bool
    valid_actions: List[Tuple[tuple, tuple]]
    next_valid_actions: List[Tuple[tuple, tuple]]

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = 0.001
        
        self.buffer: deque[Experience] = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.position_counts = defaultdict(int)  # Track position frequency
        
        # Statistics for normalization
        self.reward_stats = RunningStats()
        
    def push(self, experience: Experience):
        # Update position frequency
        for action in experience.valid_actions:
            self.position_counts[action[1]] += 1
        
        # Update reward statistics
        self.reward_stats.update(experience.reward)
        
        # Calculate initial priority
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(max_priority)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple:
        total = len(self.buffer)
        priorities = np.array(self.priorities, dtype=np.float32)
        
        # Calculate sampling probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices and calculate importance weights
        indices = np.random.choice(total, batch_size, p=probs)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Retrieve experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        # Prepare batch
        states = torch.stack([e.state for e in experiences])
        actions = [e.action for e in experiences]
        rewards = torch.tensor([self.reward_stats.normalize(e.reward) for e in experiences], dtype=torch.float32)
        next_states = torch.stack([e.next_state for e in experiences])
        dones = torch.tensor([e.done for e in experiences], dtype=torch.bool)
        valid_actions_list = [e.valid_actions for e in experiences]
        next_valid_actions_list = [e.next_valid_actions for e in experiences]
        
        return (states, actions, rewards, next_states, dones, 
                valid_actions_list, next_valid_actions_list, 
                weights, indices)
    
    def update_priorities(self, indices: List[int], td_errors: torch.Tensor):
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = (abs(error.item()) + 1e-6) ** self.alpha
    
    def get_position_weights(self) -> Dict[tuple, float]:
        """Calculate inverse frequency weights for positions"""
        total_count = sum(self.position_counts.values()) + len(self.position_counts)
        weights = {pos: math.log(total_count / (count + 1)) 
                  for pos, count in self.position_counts.items()}
        return weights
        
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)

class RunningStats:
    """Track running statistics for normalization"""
    def __init__(self):
        self.mean = 0
        self.std = 1
        self.count = 0
        
    def update(self, value: float):
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        if self.count > 1:
            self.std = math.sqrt(
                ((self.count - 2) * (self.std ** 2) + delta * (value - self.mean)) 
                / (self.count - 1)
            )
    
    def normalize(self, value: float) -> float:
        return (value - self.mean) / (self.std + 1e-8)

class ChineseCheckersTransformer(nn.Module):
    def __init__(self, 
                 board_size: int = 132,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.board_size = board_size
        self.hidden_dim = hidden_dim
        
        # Board state embedding
        self.board_embedding = nn.Sequential(
            nn.Linear(board_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, board_size, hidden_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Action encoding
        self.action_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),  # 4 = from_x, from_y, to_x, to_y
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Q-value prediction
        self.q_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1/math.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def encode_board(self, state: torch.Tensor) -> torch.Tensor:
        # Add positional encoding
        x = self.board_embedding(state) + self.pos_encoding
        
        # Apply transformer
        return self.transformer(x)
    
    def encode_action(self, action: Tuple[tuple, tuple]) -> torch.Tensor:
        from_pos, to_pos = action
        action_vec = torch.tensor([
            from_pos[0] / 11.0,  # Normalize coordinates
            from_pos[1] / 12.0,
            to_pos[0] / 11.0,
            to_pos[1] / 12.0
        ], dtype=torch.float32)
        
        return self.action_encoder(action_vec)
    
    def forward(self, state: torch.Tensor, valid_actions: List[Tuple[tuple, tuple]]) -> torch.Tensor:
        batch_size = state.shape[0]
        
        # Encode board state
        board_encoding = self.encode_board(state)
        board_features = board_encoding.mean(dim=1)  # Global board representation
        
        # Encode all valid actions
        action_encodings = torch.stack([
            self.encode_action(action) for action in valid_actions
        ])
        
        # Combine board and action features
        board_features = board_features.unsqueeze(1).expand(-1, len(valid_actions), -1)
        combined = torch.cat([board_features, action_encodings.unsqueeze(0).expand(batch_size, -1, -1)], dim=-1)
        
        # Calculate Q-values for all actions
        q_values = self.q_net(combined).squeeze(-1)
        
        return q_values

class DynamicActionAgent:
    def __init__(self,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 target_update_freq: int = 1000,
                 batch_size: int = 32,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        self.device = torch.device(device)
        self.policy_net = ChineseCheckersTransformer().to(self.device)
        self.target_net = ChineseCheckersTransformer().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=10000,
            eta_min=learning_rate * 0.1
        )
        
        self.memory = PrioritizedReplayBuffer()
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.steps = 0
        
        # Training statistics
        self.episode_rewards = []
        self.avg_q_values = []
        
    def select_action(self, state: torch.Tensor, valid_actions: List[Tuple[tuple, tuple]]) -> Tuple[tuple, tuple]:
        if not valid_actions:
            raise ValueError("No valid actions available")
            
        if random.random() < self.epsilon:
            # Epsilon-greedy exploration
            if random.random() < 0.7:  # 70% pure random
                return random.choice(valid_actions)
            else:  # 30% goal-directed random
                # Prefer actions that move pieces forward
                weights = []
                for from_pos, to_pos in valid_actions:
                    progress = to_pos[1] - from_pos[1]  # Vertical progress
                    weights.append(math.exp(progress))  # Exponential weighting
                
                total = sum(weights)
                if total > 0:
                    weights = [w/total for w in weights]
                    return random.choices(valid_actions, weights=weights)[0]
                return random.choice(valid_actions)
        
        # Greedy action selection
        with torch.no_grad():
            state = state.to(self.device)
            q_values = self.policy_net(state.unsqueeze(0), valid_actions)
            action_idx = q_values.argmax().item()
            return valid_actions[action_idx]
    
    def update_model(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None
            
        # Sample experiences
        (states, actions, rewards, next_states, dones,
         valid_actions_list, next_valid_actions_list,
         weights, indices) = self.memory.sample(self.batch_size)
        
        states = states.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        
        # Calculate current Q-values
        current_q_values = torch.zeros(self.batch_size, device=self.device)
        for i in range(self.batch_size):
            q_values = self.policy_net(states[i].unsqueeze(0), valid_actions_list[i])
            # Find the index of the action by comparing tuples
            action_idx = next(idx for idx, act in enumerate(valid_actions_list[i]) 
                             if act[0] == actions[i][0] and act[1] == actions[i][1])
            current_q_values[i] = q_values[0, action_idx]
        
        # Calculate target Q-values
        with torch.no_grad():
            target_q_values = torch.zeros(self.batch_size, device=self.device)
            for i in range(self.batch_size):
                if not dones[i] and next_valid_actions_list[i]:
                    next_q_values = self.target_net(next_states[i].unsqueeze(0), 
                                                  next_valid_actions_list[i])
                    target_q_values[i] = rewards[i] + self.gamma * next_q_values.max()
                else:
                    target_q_values[i] = rewards[i]
        
        # Calculate loss with importance sampling weights
        td_errors = target_q_values - current_q_values
        loss = (weights * F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.scheduler.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Update priorities
        self.memory.update_priorities(indices, td_errors.abs().cpu())
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, path: str):
        save_path = Path(path)
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, save_path)
    
    def load(self, path: str):
        save_path = Path(path)
        if save_path.exists():
            checkpoint = torch.load(save_path)
            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.steps = checkpoint['steps']

def calculate_reward(board: Board, player: int, done: bool) -> float:
    """Calculate reward with multiple components"""
    base_reward = 0.0
    
    # Get piece positions
    goal_positions = set(board.winning_keys(player))
    current_positions = set(board.player_keys(player))
    
    if not current_positions:
        return -10.0  # Severe penalty for losing pieces
    
    # Terminal state rewards
    if done:
        if board.winner and board.winner.value == player:
            return 10.0  # Win reward
        return -5.0  # Loss penalty
    
    # Progress rewards
    pieces_in_goal = len(goal_positions.intersection(current_positions))
    total_pieces = len(current_positions)
    goal_progress = pieces_in_goal / total_pieces
    base_reward += goal_progress * 2.0
    
    # Position-based rewards
    y_coords = [pos[1] for pos in current_positions]
    x_coords = [pos[0] for pos in current_positions]
    
    # Vertical progress
    avg_y = sum(y_coords) / len(y_coords)
    y_progress = avg_y / 12.0  # Normalize by board height
    base_reward += y_progress
    
    # Formation rewards
    x_spread = max(x_coords) - min(x_coords)
    y_spread = max(y_coords) - min(y_coords)
    spread_penalty = -(x_spread + y_spread) / 20.0  # Penalize spreading out too much
    base_reward += spread_penalty
    
    # Distance-based rewards
    total_distance = 0
    for pos in current_positions:
        min_goal_dist = min(abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]) 
                          for goal in goal_positions)
        total_distance += min_goal_dist
    avg_distance = total_distance / total_pieces
    distance_reward = -avg_distance / 10.0  # Negative reward for distance
    base_reward += distance_reward
    
    return base_reward

def train_agent(
    agent: DynamicActionAgent,
    num_episodes: int = 1000,
    max_steps: int = 100,
    eval_freq: int = 50,
    save_path: Optional[str] = None,
    load_path: Optional[str] = None,
    update: int = 10  # Add update parameter with default value
) -> None:
    """Train the agent
    
    Args:
        agent: The DynamicActionAgent to train
        num_episodes: Total number of episodes to train for
        max_steps: Maximum steps per episode
        eval_freq: How often to run evaluation
        save_path: Where to save model checkpoints
        load_path: Path to load a pre-trained model from
        update: How often to print episode updates (every N episodes)
    """
    
    # Load existing model if specified
    if load_path:
        print(f"Loading model from {load_path}")
        agent.load(load_path)
    
    print(f"Starting training for {num_episodes} episodes with updates every {update} episodes")
    best_reward = float('-inf')
    
    for episode in range(num_episodes):
        # Initialize game
        board = Board.new([Color(0), Color(3)])
        episode_reward = 0
        steps_taken = 0
        
        for step in range(max_steps):
            steps_taken = step + 1
            # Get current state and valid actions
            state = board.state.detach().clone().reshape(-1)  # Flatten to 132 elements
            valid_actions = [(tuple(from_pos.tolist()), tuple(to_pos.tolist())) 
                           for from_pos, to_positions in board.all_moves()
                           for to_pos in to_positions]
            
            if not valid_actions:
                break
                
            # Select and perform action
            action = agent.select_action(state, valid_actions)
            old_state = state.clone()
            
            # Make move
            from_pos = torch.tensor(action[0], dtype=torch.float32)
            to_pos = torch.tensor(action[1], dtype=torch.float32)
            board.move(from_pos, to_pos)
            done = board.winner is not None
            
            # Calculate reward
            reward = calculate_reward(board, board.playing[1], done)
            
            # Get next state and valid actions
            next_state = board.state.detach().clone().reshape(-1)  # Flatten to 132 elements
            next_valid_actions = []
            if not done:
                next_valid_actions = [(tuple(from_pos.tolist()), tuple(to_pos.tolist())) 
                                    for from_pos, to_positions in board.all_moves()
                                    for to_pos in to_positions]
            
            # Store experience
            experience = Experience(
                old_state,
                action,
                reward,
                next_state,
                done,
                valid_actions,
                next_valid_actions
            )
            agent.memory.push(experience)
            
            # Update model
            if len(agent.memory) > agent.batch_size:
                loss = agent.update_model()
            
            episode_reward += reward
            
            if done:
                break
            
            # Rotate board for next player
            board.rotate(1)
        
        # Logging - use update parameter to control frequency
        if (episode + 1) % update == 0:
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward={episode_reward:.2f}, "
                  f"Epsilon={agent.epsilon:.3f}, "
                  f"Steps={steps_taken}, "
                  f"Memory={len(agent.memory)}")
        
        # Save best model
        if save_path and episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(f"{save_path}_best.pt")
            print(f"New best model saved with reward {best_reward:.2f}")
        
        # Regular checkpoints
        if save_path and (episode + 1) % 100 == 0:
            agent.save(f"{save_path}_episode_{episode + 1}.pt")
            print(f"Saved checkpoint at episode {episode + 1}")
            
        # Evaluation
        if (episode + 1) % eval_freq == 0:
            eval_reward = evaluate_agent(agent, num_episodes=5)
            print(f"\nEvaluation after {episode + 1} episodes: {eval_reward:.2f}\n")

def evaluate_agent(
    agent: DynamicActionAgent,
    num_episodes: int = 10,
    max_steps: int = 100
) -> float:
    """Evaluate the agent's performance"""
    agent.policy_net.eval()
    total_reward = 0
    
    with torch.no_grad():
        for episode in range(num_episodes):
            board = Board.new([Color(0), Color(3)])
            episode_reward = 0
            
            for step in range(max_steps):
                state = board.state.detach().clone().reshape(-1)  # Flatten to 132 elements
                valid_actions = [(tuple(from_pos.tolist()), tuple(to_pos.tolist())) 
                               for from_pos, to_positions in board.all_moves()
                               for to_pos in to_positions]
                
                if not valid_actions:
                    break
                
                # Use greedy policy (epsilon = 0)
                old_epsilon = agent.epsilon
                agent.epsilon = 0
                action = agent.select_action(state, valid_actions)
                agent.epsilon = old_epsilon
                
                # Make move
                from_pos = torch.tensor(action[0], dtype=torch.float32)
                to_pos = torch.tensor(action[1], dtype=torch.float32)
                board.move(from_pos, to_pos)
                done = board.winner is not None
                
                # Calculate reward
                reward = calculate_reward(board, board.playing[1], done)
                episode_reward += reward
                
                if done:
                    break
                    
                board.rotate(1)
            
            total_reward += episode_reward
    
    agent.policy_net.train()
    return total_reward / num_episodes

def run(
    matches_count: int = 20,
    max_turns: int = 50,
    update: int = 2,
    load_path: Optional[str] = None,
    save_path: Optional[str] = None,
    save_frequency: int = 10,
    learning_rate: float = 1e-4,
    batch_size: int = 32,
    gamma: float = 0.99,
    initial_epsilon: float = 1.0,
    final_epsilon: float = 0.01,
    epsilon_decay: float = 0.995,
    memory_size: int = 100000,
    target_update_freq: int = 10,
    num_episodes: int = 1000
):
    """Run the DQN training process.
    
    Args:
        matches_count: Number of matches per evaluation
        max_turns: Maximum number of turns per match
        update: How often to print updates
        load_path: Path to load a pre-trained model from
        save_path: Path to save checkpoints to
        save_frequency: How often to save checkpoints
        learning_rate: Learning rate for the optimizer
        batch_size: Batch size for training
        gamma: Discount factor
        initial_epsilon: Initial exploration rate
        final_epsilon: Final exploration rate
        epsilon_decay: Rate at which epsilon decays
        memory_size: Size of the replay buffer
        target_update_freq: How often to update the target network
        num_episodes: Total number of episodes to train for
    """
    # Set default save path if not provided
    if save_path is None:
        save_path = 'deepq2'
    
    # Initialize the agent
    agent = DynamicActionAgent(
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=initial_epsilon,
        epsilon_end=final_epsilon,
        epsilon_decay=epsilon_decay,
        target_update_freq=target_update_freq,
        batch_size=batch_size
    )
    
    # Train the agent using the existing train_agent function
    train_agent(
        agent=agent,
        num_episodes=num_episodes,
        max_steps=max_turns,
        eval_freq=matches_count,
        save_path=save_path,
        load_path=load_path,
        update=update
    )
