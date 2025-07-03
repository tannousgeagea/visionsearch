from .clip.core import ClipEmbedding
from .perception_encoder.core import PerceptionEmbedding
from .base import BaseEmbedder

def create(model_name: str) -> BaseEmbedder:
    if model_name.lower() == "clip":
        return ClipEmbedding()
    elif model_name.lower() == "perception":
        return PerceptionEmbedding()
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
