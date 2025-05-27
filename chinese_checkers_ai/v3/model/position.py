from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Position:
    euclid: np.ndarray
    key: Tuple[int, int]

    @staticmethod
    def rotation_matrix(steps: int) -> np.ndarray:
        angle_radians = steps * np.pi / 3
        return np.array([
            [np.cos(angle_radians), -np.sin(angle_radians)], 
            [np.sin(angle_radians), np.cos(angle_radians)]
        ])
    
    @staticmethod
    def euclid_to_key(euclid: np.ndarray) -> Tuple[int, int]:
        return tuple(map(int, np.round(np.array([euclid[0], euclid[1]/np.sin(np.pi/3)]))))

    @staticmethod
    def key_to_euclid(key: Tuple[int, int]) -> np.ndarray:
        return np.array([key[0], key[1]*np.sin(np.pi/3)])

    def rotate(self, steps: int) -> "Position":
        euclid = self.euclid @ self.rotation_matrix(steps)
        key = self.euclid_to_key(euclid)
        return Position(euclid, key)
    
    def translate(self, x_steps: int, y_steps: int) -> "Position":
        euclid = self.euclid + np.array([x_steps, y_steps*2*np.sin(np.pi/3)])
        key = self.euclid_to_key(euclid)
        return Position(euclid, key)
    
    def scale(self, factor: float) -> "Position":
        euclid = self.euclid * factor
        return Position(euclid, self.key)

    def siblings(self, magnitude: int = 1) -> List["Position"]:
        right = Position.new_center().translate(2*magnitude, 0)
        return [
            self + right.rotate(i)
            for i in range(6)
        ]
    
    def __add__(self, other: "Position") -> "Position":
        return Position.new(euclid=self.euclid + other.euclid)
    
    def __eq__(self, other: "Position") -> bool:
        return self.key == other.key
    
    def __hash__(self) -> int:
        return hash(self.key)
    
    def __str__(self) -> str:
        return f"{self.key}"
    
    def __repr__(self) -> str:
        return f"{self.key}"

    @classmethod
    def new(cls, euclid: Optional[np.ndarray] = None, key: Optional[Tuple[int, int]] = None) -> "Position":
        if euclid is None and key is None:
            raise ValueError("Either euclid or key must be provided")
        if euclid is not None and key is not None:
            return cls(euclid, key)
        if euclid is not None:
            return cls(euclid, cls.euclid_to_key(euclid))
        return cls(cls.key_to_euclid(key), key)
    
    @classmethod
    def new_center(cls) -> "Position":
        return cls(np.array([0, 0]), (0, 0))