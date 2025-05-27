"""
so this is a simpler trainer for neat with non structural evolution
this a first attempt, before attempting deep q learning, we'll still be
evolving the q function, but via neat instead of back propgation
"""

from concurrent.futures import Future, ThreadPoolExecutor
from random import Random
import time
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from pathlib import Path

from chinese_checkers_ai.v2.model.board import Board
from chinese_checkers_ai.v2.model.color import Color
from chinese_checkers_ai.v2.trainer.deepq import calculate_board_score as score

random = Random()

class QNetwork(nn.Module):
    DOMINATE_PREVALANCE = 1.2
    MUTATE_MAG = 0.01

    def __init__(self):
        super().__init__()
        # First path: Process board state with attention
        self.board_path = nn.Sequential(
            nn.Flatten(),
            nn.Linear(132, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        
        # Self-attention mechanism
        self.attention = nn.MultiheadAttention(128, num_heads=4, batch_first=True)
        self.attention_norm = nn.LayerNorm(128)
        
        # Positional features path
        self.positional_features = nn.Sequential(
            nn.Linear(6, 32),  # Extended features
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        
        # Combine paths with residual connections
        self.combined = nn.Sequential(
            nn.Linear(192, 128),  # 128 from board + 64 from positional
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            ResidualBlock(128),
            ResidualBlock(128),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def extract_positional_features(self, state: torch.Tensor) -> torch.Tensor:
        """Extract enhanced positional features from the board state"""
        batch_size = state.shape[0] if len(state.shape) > 2 else 1
        state_2d = state.view(batch_size, -1)
        
        features = []
        
        # 1. Distance to goal area
        piece_positions = (state_2d > 0).float()
        row_indices = torch.arange(12, dtype=torch.float32).repeat(11)[:132]
        avg_distance = (piece_positions * row_indices.to(piece_positions.device)).sum(dim=1, keepdim=True)
        avg_distance = avg_distance / (piece_positions.sum(dim=1, keepdim=True) + 1e-6)
        features.append(avg_distance)
        
        # 2. Piece density in different regions
        for i in range(3):
            start_idx = i * (state_2d.shape[1] // 3)
            end_idx = start_idx + (state_2d.shape[1] // 3)
            region = state_2d[:, start_idx:end_idx]
            density = (region > 0).float().mean(dim=1, keepdim=True)
            features.append(density)
        
        # 3. Formation metrics
        x_coords = torch.arange(11, dtype=torch.float32).repeat(12)[:132]
        piece_x_positions = (state_2d > 0).float() * x_coords.to(state_2d.device)
        x_spread = (piece_x_positions.max(dim=1)[0] - piece_x_positions.min(dim=1)[0]).unsqueeze(1)
        features.append(x_spread / 11.0)  # Normalized spread
        
        # 4. Progress metric
        progress = (piece_positions * row_indices.to(piece_positions.device)).mean(dim=1, keepdim=True) / 12.0
        features.append(progress)
        
        return torch.cat(features, dim=1)  # Returns (batch_size, 6)

    def q(self, state: torch.Tensor) -> float:
        with torch.no_grad():
            if len(state.shape) == 2:
                state = state.view(1, *state.shape)
            
            # Process board state
            board_features = self.board_path(state)
            
            # Apply self-attention
            board_features_attn = board_features.unsqueeze(1)
            attn_out, _ = self.attention(board_features_attn, board_features_attn, board_features_attn)
            board_features = board_features + attn_out.squeeze(1)  # Residual connection
            board_features = self.attention_norm(board_features)
            
            # Extract and process positional features
            pos_features = self.extract_positional_features(state)
            pos_features = self.positional_features(pos_features)
            
            # Combine features
            combined = torch.cat([board_features, pos_features], dim=1)
            return self.combined(combined)
    
    @classmethod
    def crossover(cls, dominant: "QNetwork", submissive: "QNetwork"):
        offspring = QNetwork()
        
        d_parameters = list(dominant.parameters())
        s_parameters = list(submissive.parameters())
        
        # Enhanced crossover with adaptive mixing
        for i, parameter in enumerate(offspring.parameters()):
            # Calculate parameter importance based on gradients
            d_grad_mag = torch.abs(d_parameters[i].grad).mean() if d_parameters[i].grad is not None else torch.tensor(0.0)
            s_grad_mag = torch.abs(s_parameters[i].grad).mean() if s_parameters[i].grad is not None else torch.tensor(0.0)
            
            # Adjust dominance based on gradient magnitude
            local_dominance = cls.DOMINATE_PREVALANCE * (1 + d_grad_mag) / (1 + s_grad_mag)
            
            # Crossover with adaptive probability
            mask = torch.rand_like(parameter.data) < local_dominance
            parameter.data = torch.where(mask, d_parameters[i].data, s_parameters[i].data)
            
            # Add small noise for exploration
            noise = torch.randn_like(parameter.data) * cls.MUTATE_MAG * (1 - mask.float())
            parameter.data += noise
        
        return offspring
    
    @classmethod
    def mutate(cls, dominant: "QNetwork"):
        for parameter in dominant.parameters():
            if random.random() < 0.3:  # 30% mutation rate
                mutation_type = random.random()
                
                if mutation_type < 0.7:  # Small adjustments
                    delta = torch.randn_like(parameter.data) * cls.MUTATE_MAG
                    parameter.data += delta
                elif mutation_type < 0.9:  # Larger jumps
                    delta = torch.randn_like(parameter.data) * cls.MUTATE_MAG * 5
                    parameter.data += delta
                else:  # Selective randomization
                    mask = torch.rand_like(parameter.data) < 0.1
                    parameter.data[mask] = torch.randn_like(parameter.data[mask])

class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)  # Residual connection

class Match:
    def __init__(self, players: Dict[Color, QNetwork]):
        self.players = players
        self.board = Board.new(list(players.keys()))
        self.moves = {player: 0 for player in self.players}
        self.history = []  # Track game history for better scoring

    def step_player(self):
        if self.board.winner is None:
            player = self.board.playing[1]
            self.board.rotate(player=player)

            all_moves = self.board.all_moves()
            if not all_moves:
                return

            batch = torch.zeros((sum(len(item[1]) for item in all_moves), *self.board.state.shape))
            batch_moves = []

            i = 0
            for from_position, to_positions in all_moves:
                for to_position in to_positions:
                    batch_moves.append((from_position, to_position))
                    new_state = self.board.move_new(from_position, to_position, check=False)
                    batch[i, :, :] = new_state.state
                    i += 1
            
            # Add exploration to move selection
            results = self.players[player].q(batch)
            if random.random() < 0.1:  # 10% random exploration
                best_index = random.randint(0, len(batch_moves) - 1)
            else:
                best_index = torch.max(results, dim=0).indices[0]

            move = batch_moves[best_index]
            self.board.move(*move)
            self.moves[player] += 1
            
            # Store move in history
            self.history.append((player, move, self.board.state.clone()))

    def step(self):
        for _ in range(len(self.players)):
            self.step_player()

    def score(self, player):
        base_score = score(self.board, player.value)
        
        # Additional scoring factors
        if len(self.history) > 0:
            player_moves = [h for h in self.history if h[0] == player]
            
            # Reward for efficient movement (less moves is better)
            move_efficiency = max(0, 1 - (len(player_moves) / 50))  # 50 moves as benchmark
            
            # Reward for forward progress
            progress = 0
            if len(player_moves) > 0:
                start_pos = player_moves[0][2]  # Initial board state
                end_pos = player_moves[-1][2]   # Final board state
                progress = ((start_pos - end_pos) ** 2).mean().item()  # Measure of position change
            
            # Combine scores
            return base_score * (1 + 0.2 * move_efficiency + 0.2 * progress)
        
        return base_score
    
    def fitest(self) -> Tuple[float, QNetwork]:
        fitest = (-1, None)

        for player in self.players:
            score = self.score(player)
            if fitest[0] < score:
                fitest = (score, self.players[player])

        return fitest
    
    @property
    def done(self) -> bool:
        return self.board.winner is not None
    
    @classmethod
    def new(cls, count:Optional[int]=None, pool:Optional[List[QNetwork]]=None, weights:Optional[List[float]] = None) -> "Match":
        if count is None:
            count = random.randint(2, 6)
        
        players = {}
        colors = list(range(6))

        for _ in range(count):
            color = random.choice(colors)
            colors.remove(color)
            
            if pool is not None:
                parents = random.choices(
                    pool,
                    weights,
                    k = 2
                )

                player = QNetwork.crossover(*parents)
                QNetwork.mutate(player)
            else:
                player = QNetwork()

            players[Color(color)] = player

        return Match(players)

executor = ThreadPoolExecutor(4)

def run_generation(
    player_count: Optional[int] = None,
    matches_count: int = 100,
    pool:Optional[List[QNetwork]]=None, 
    weights:Optional[List[float]] = None,
    max_turns: int = 100,
    survival: int = 10,
    update: int = 10,
    generation: int = 1,
    num_workers: int = 4,
    batch_size: int = 16  # Batch processing of matches
) -> Tuple[List[QNetwork], List[float]]:
    matches: List[Match] = [
        Match.new(
            count=player_count,
            pool=pool,
            weights=weights
        )
        for _ in range(matches_count)
    ]

    fitests: List[Tuple[float, QNetwork]] = []
    active_matches = matches.copy()  # Keep a separate list of active matches
    
    # Process matches in batches for better GPU utilization
    for t_i in range(max_turns):
        if not active_matches:
            break

        # Process matches in parallel batches
        batch_matches = [active_matches[i:i + batch_size] for i in range(0, len(active_matches), batch_size)]
        completed_matches = []  # Track matches to remove after processing
        
        for batch in batch_matches:
            # Create futures for each match in the batch
            futures = [
                executor.submit(match.step)
                for match in batch
            ]

            try:
                # Process completed matches in the batch
                for i, future in enumerate(futures):
                    future.result()
                    match = batch[i]
                    
                    if match.done:
                        # Get fitness score and network
                        fitest = match.fitest()
                        
                        # Add position history tracking for better exploration
                        if hasattr(match.board, 'position_history'):
                            history_bonus = len(set(match.board.position_history)) * 0.01
                            fitest = (fitest[0] + history_bonus, fitest[1])
                        
                        fitests.append(fitest)
                        completed_matches.append(match)
                        
            except Exception as e:
                executor.shutdown(False, cancel_futures=True)
                raise e

        # Remove completed matches from active matches
        for match in completed_matches:
            active_matches.remove(match)

        if t_i%update == update-1:
            _fitestes = list(map(lambda f: f[0], fitests))
            for match in active_matches:
                _fitestes.append(match.fitest()[0])
            print(f"Turn {generation}-{t_i+1}: {max(_fitestes)}")

    # Process any remaining active matches
    for match in active_matches:
        fitests.append(match.fitest())

    # Sort by fitness score
    fitests.sort(reverse=True, key=lambda f: f[0])
    
    # Apply elitism - keep best performers
    elite_count = max(1, survival // 4)
    elite = fitests[:elite_count]
    
    # Tournament selection for remaining slots
    tournament_size = 3
    selected = []
    
    while len(selected) < survival - elite_count:
        tournament = random.sample(fitests, tournament_size)
        winner = max(tournament, key=lambda x: x[0])
        selected.append(winner)
    
    # Combine elite and selected
    result = [], []
    
    # Add elite networks first
    for score, network in elite:
        result[0].append(network)
        result[1].append(score)
    
    # Add selected networks
    for score, network in selected:
        result[0].append(network)
        result[1].append(score)

    return result

def run(
    player_count: Optional[int] = None,
    matches_count: int = 20,
    max_turns: int = 50,
    survival: int = 5,
    generations: int = 100,
    update: int = 10,
    load_path: Optional[str] = None,
    save_path: Optional[str] = None,
    save_frequency: int = 10
):
    # Set default save path if not provided
    if save_path is None:
        save_path = 'evolution'
    save_path = Path(save_path)
    
    # Initialize pool and weights
    pool, weights = None, None
    
    # Load existing networks if path provided
    if load_path is not None:
        load_path = Path(load_path)
        pool_file = load_path.with_suffix('.pool.pt')
        weights_file = load_path.with_suffix('.weights.pt')
        generation_file = load_path.with_suffix('.generation.json')
        
        if pool_file.exists() and weights_file.exists():
            print(f"Loading existing pool from {load_path}...")
            pool_state = torch.load(pool_file)
            weights_state = torch.load(weights_file)
            
            pool = [QNetwork() for _ in range(len(pool_state))]
            for i, state_dict in enumerate(pool_state):
                pool[i].load_state_dict(state_dict)
            weights = weights_state
            print(f"Loaded pool of {len(pool)} networks with weights")
    
    for g_i in range(generations):
        pool, weights = run_generation(
            player_count=player_count,
            matches_count=matches_count,
            max_turns=max_turns,
            survival=survival,
            pool=pool,
            weights=weights,
            update=update,
            generation=g_i+1,
        )

        print(f"Generation {g_i + 1}: {weights[0]}")
        
        # Save networks periodically
        if (g_i + 1) % save_frequency == 0:
            print(f"Saving generation {g_i + 1} checkpoint...")
            
            # Save pool state dictionaries
            pool_state = [network.state_dict() for network in pool]
            pool_path = save_path.with_suffix('.pool.pt')
            torch.save(pool_state, pool_path)
            
            # Save weights
            weights_path = save_path.with_suffix('.weights.pt')
            torch.save(weights, weights_path)
            
            print(f"Checkpoint saved to {save_path}.*")
        
        # Always save the best network
        best_network = pool[0]  # First network is always the best after sorting
        best_path = save_path.with_name(f"{save_path.stem}_best").with_suffix('.network')
        torch.save(best_network.state_dict(), best_path)

    # Save final state
    print("Saving final state...")
    final_pool_state = [network.state_dict() for network in pool]
    final_pool_path = save_path.with_suffix('.final.pool.pt')
    final_weights_path = save_path.with_suffix('.final.weights.pt')
    torch.save(final_pool_state, final_pool_path)
    torch.save(weights, final_weights_path)
    print(f"Final state saved to {save_path}.final.*")