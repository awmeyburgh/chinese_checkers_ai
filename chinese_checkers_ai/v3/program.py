import sys
from chinese_checkers_ai.v3.view.game import Game
from chinese_checkers_ai.v3.trainer.cursor_deepq_trainer import DeepQTrainer


class Program:
    __SINGLETON = None

    @classmethod
    def get(cls) -> "Program":
        if cls.__SINGLETON is None:
            cls.__SINGLETON = Program()
        return cls.__SINGLETON
    
    def __init__(self):
        self.mode = self.get_mode()
        if self.mode == "game":
            self.game = Game(players=6)

    def get_mode(self) -> str:
        if len(sys.argv) == 1:
            return "game"
        else:
            return sys.argv[1]
        
    def run(self):
        if self.mode == "game":
            self.game.run()
        elif self.mode == "train":
            DeepQTrainer.run()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")