from abc import ABC, abstractmethod
from typing import Union
import numpy as np
from PIL import Image

class BaseEmbedder(ABC):
    @abstractmethod
    def extract_image_feature(self, image: Union[str, Image.Image]) -> np.ndarray:
        pass

    @abstractmethod
    def extract_text_feature(self, text: str) -> np.ndarray:
        pass
