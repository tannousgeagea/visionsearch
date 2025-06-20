import numpy as np
from common_utils.embedding.base import BaseEmbedder

class PerceptionEmbedding(BaseEmbedder):
    def __init__(self):
        # TODO: load Meta Perception model
        pass

    def extract_text_feature(self, text: str) -> np.ndarray:
        # TODO: convert text to embedding
        raise NotImplementedError("PerceptionEmbedder.embed_text not implemented")

    def extract_image_feature(self, image) -> np.ndarray:
        # TODO: convert image to embedding
        raise NotImplementedError("PerceptionEmbedder.embed_image not implemented")