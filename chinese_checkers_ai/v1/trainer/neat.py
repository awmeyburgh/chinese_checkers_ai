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
from chinese_checkers_ai.v1.model.board import Board
from chinese_checkers_ai.v1.model.color import Color

random = Random()

class QNetwork(nn.Module):
    DOMINATE_PREVALANCE = 1.2
    MUTATE_MAG = 0.01

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(732, 1),
            nn.Sigmoid()
        )

    def q(self, state: torch.Tensor) -> float:
        with torch.no_grad():
            if len(state.shape) == 2:
                state = state.view(1, *state.shape)
            return self.layers(state)
    
    @classmethod
    def crossover(cls, dominant: "QNetwork", submissive: "QNetwork"):
        offspring = QNetwork()
        
        d_paramters = list(dominant.parameters())
        s_paramters = list(submissive.parameters())

        for i, parameter in enumerate(offspring.parameters()):
            distibution = torch.clamp(torch.randn(parameter.data.shape)*cls.DOMINATE_PREVALANCE, 0, 1)

            parameter.data = distibution*d_paramters[i].data + (1-distibution)*s_paramters[i].data
        
        return offspring
    
    @classmethod
    def mutate(cls, dominant: "QNetwork"):
        for parameter in dominant.parameters():
            delta = (torch.randn(parameter.data.shape)-0.5)*cls.MUTATE_MAG

            parameter.data += delta


class Match:
    def __init__(self, players: Dict[Color, QNetwork]):
        self.players = players
        self.board = Board.new(list(players.keys()))
        self.moves = {player: 0 for player in self.players}

    def step_player(self):
        if self.board.winner is None:
            player = self.board.player
            self.board.rotate(player)

            all_moves = self.board.all_moves().items()

            batch = torch.zeros((sum(len(item[1]) for item in all_moves), *self.board.state.shape))
            batch_moves = []

            i = 0
            for from_position, to_positions in all_moves:
                for to_position in to_positions:
                    batch_moves.append((from_position, to_position))
                    batch[i, :, :] = self.board.move_new(from_position, to_position, check=False).state
                    i += 1
            results = self.players[player].q(batch)
            best_index = torch.max(results, dim=0).indices[0]

            self.board.move(*batch_moves[best_index])
            self.moves[player] += 1

    def step(self):
        start = time.time()
        
        for _ in range(len(self.players)):
            self.step_player()

        print(time.time()-start)

    def score(self, player: Color) -> float:
        self.board.rotate(player)

        score = 0
        all_in = True

        for position in self.board.players[player]:
            score += max(position[1]-3, 0)
            slot = self.board.slots[position]
            if slot.floor_color is None or slot.floor_color.value != (slot.floor_color+3)%6:
                all_in = False

        if all_in:
            score += 1000

        return score
    
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
    def new(self, count:Optional[int]=None, pool:Optional[List[QNetwork]]=None, weights:Optional[List[float]] = None) -> List["Match"]:
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

    for t_i in range(max_turns):
        if len(matches) == 0:
            break

        futures = [
            executor.submit(match.step)
            for match in matches
        ]

        try:
            offset = 0
            for _i in range(len(matches)):
                i = _i - offset
                futures[i].result()
                if matches[i].done:
                    fitests.append(matches[i].fitest())
                    matches.pop(i)
                    offset += 1
        except Exception as e:
            executor.shutdown(False, cancel_futures=True)
            raise e

        if t_i%update == update-1:
            _fitestes = list(map(lambda f: f[0], fitests))
            for match in matches:
                _fitestes.append(match.fitest()[0])
            print(f"Turn {generation}-{t_i+1}: {max(_fitestes)}")

    for match in matches:
        fitests.append(match.fitest())

    fitests.sort(reverse=True, key=lambda f: f[0])
    
    result = [], []

    for i in range(survival):
        result[0].append(fitests[i][1])
        result[1].append(fitests[i][0])

    return result

def run(
    player_count: Optional[int] = None,
    matches_count: int = 20,
    max_turns: int = 50,
    survival: int = 5,
    generations: int = 100,
    update = 10,
):
    pool, weights = None, None
    
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

        torch.save(pool[0], 'neat.network')