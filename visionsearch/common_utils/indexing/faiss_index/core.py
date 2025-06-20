import faiss
import numpy as np
from common_utils.indexing.base import BaseIndexer

class FaissIndexer(BaseIndexer):
    def __init__(self, dim=512):
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray) -> None:
        faiss.normalize_L2(vectors)
        self.index.add(vectors)

    def search(self, query_vector: np.ndarray, k=30):
        faiss.normalize_L2(query_vector)
        D, I = self.index.search(query_vector, k)
        return I[0], D[0]

    def save(self, path: str) -> None:
        faiss.write_index(self.index, path)

    def load(self, path: str) -> None:
        self.index = faiss.read_index(path)
