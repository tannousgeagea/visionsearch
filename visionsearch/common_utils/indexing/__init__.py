from .faiss_index.core import FaissIndexer

def create(backend: str = "faiss", **kwargs):
    backend = backend.lower()
    if backend == "faiss":
        return FaissIndexer(**kwargs)
    else:
        raise ValueError(f"Unsupported index backend: {backend}")
