import numpy as np
from abc import ABC, abstractmethod

class BaseIndexer(ABC):
    @abstractmethod
    def add(self, vectors: np.ndarray) -> None:
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int = 5):
        """Returns (indices, distances)"""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass
