from .base import VLMBase, VLMConfig, VLMConfigurationError
from .gemini.core import GeminiVLM
from .gemma.core import GemmaVLM

class ModelTypeFactory:
    @staticmethod
    def create_vlm(provider:str, config: VLMConfig) -> VLMBase:
        if provider == "gemini":
            return GeminiVLM(config)
        if provider == "gemma":
            return GemmaVLM(config, system_prompt="")
        else:
            raise VLMConfigurationError(f"VLM Provider: {provider} is not supported !")