
import clip
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from common_utils.embedding.base import BaseEmbedder

class ClipEmbedding(BaseEmbedder):
    def __init__(self):
        super().__init__()
        self.model_name = "ViT-B/32"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)

    def extract_image_feature(self, image) -> np.ndarray:
        """Extract CLIP image embedding from the given image path."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
            
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model.encode_image(tensor).cpu().numpy()

    def extract_text_feature(self, text: str) -> np.ndarray:
        """Extract CLIP text embedding from the given text query."""
        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            return self.model.encode_text(tokens).cpu().numpy()