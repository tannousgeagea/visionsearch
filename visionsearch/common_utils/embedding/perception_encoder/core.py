import numpy as np
import torch

import common_utils.perception_models.core.vision_encoder.pe as p_encoder 
import common_utils.perception_models.core.vision_encoder.transforms as transforms
from PIL import Image

from common_utils.embedding.base import BaseEmbedder

class PerceptionEmbedding(BaseEmbedder):
    def __init__(self):
        "Use smallest weights for model init"
        self.weights = "PE-Core-B16-224"
        self.model = p_encoder.CLIP.from_config(self.weights, pretrained=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.preprocess = transforms.get_image_transform(self.model.image_size)
        self.tokenizer = transforms.get_text_tokenizer(self.model.context_length)
       
    def extract_text_feature(self, text: str) -> np.ndarray:
        "Extract normalized feature embeddings in evaluation mode"
        tokens = self.tokenizer(text).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy()
    
    def extract_image_feature(self, image) -> np.ndarray:
        "Extract normalized img embeddings in evaluation mode"
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    def get_embed_dim(self):
        return self.model.output_dim