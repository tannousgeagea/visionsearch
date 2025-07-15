from .base import VLMBase, VLMConfig, VLMConfigurationError
from .gemini.core import GeminiVLM

class ModelTypeFactory:
    @staticmethod
    def create_vlm(provider:str, config: VLMConfig) -> VLMBase:
        if provider == "gemini":
            return GeminiVLM(config)
        else:
            raise VLMConfigurationError(f"VLM Provider: {provider} is not supported !")