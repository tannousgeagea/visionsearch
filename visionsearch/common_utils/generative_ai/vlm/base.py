from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, AsyncIterator
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime
import base64
import io
from PIL import Image

# Response and Request Types
class ImageFormat(Enum):
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    GIF = "gif"
    BMP = "bmp"

class AnalysisType(Enum):
    GENERAL_DESCRIPTION = "general_description"
    OBJECT_DETECTION = "object_detection"
    TEXT_EXTRACTION = "text_extraction"
    VISUAL_QA = "visual_qa"
    SCENE_UNDERSTANDING = "scene_understanding"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    STYLE_ANALYSIS = "style_analysis"
    CONTENT_MODERATION = "content_moderation"

class ConfidenceLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class ImageInput:
    """Standardized image input format"""
    data: bytes
    format: ImageFormat
    width: Optional[int] = None
    height: Optional[int] = None
    size_bytes: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class DetectedObject:
    """Represents a detected object in an image"""
    label: str
    confidence: float
    bounding_box: Optional[Dict[str, float]] = None  # {x, y, width, height}
    attributes: Optional[Dict[str, Any]] = None

@dataclass
class ExtractedText:
    """Represents extracted text from an image"""
    text: str
    confidence: float
    bounding_box: Optional[Dict[str, float]] = None
    language: Optional[str] = None

@dataclass
class VLMResponse:
    """Standardized response format for VLM operations"""
    success: bool
    response_text: str
    confidence: ConfidenceLevel
    analysis_type: AnalysisType
    detected_objects: Optional[List[DetectedObject]] = None
    extracted_text: Optional[List[ExtractedText]] = None
    metadata: Optional[Dict[str, Any]] = None
    processing_time_ms: Optional[int] = None
    model_info: Optional[Dict[str, str | int]] = None
    error_message: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None

@dataclass
class BatchVLMResponse:
    """Response format for batch operations"""
    success: bool
    responses: List[VLMResponse]
    total_processed: int
    failed_count: int
    processing_time_ms: Optional[int] = None
    errors: Optional[List[str]] = None

@dataclass
class StreamingVLMResponse:
    """Response format for streaming operations"""
    chunk: str
    is_complete: bool
    chunk_index: int
    total_chunks: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class VLMConfig:
    """Configuration for VLM models"""
    api_key: str
    model_name: str
    endpoint: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.7
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    max_image_size_mb: int = 20
    supported_formats: List[ImageFormat] = None
    rate_limit_per_minute: int = 60
    enable_streaming: bool = False
    enable_caching: bool = True
    enable_json: bool = False
    custom_headers: Optional[Dict[str, str]] = None

class VLMException(Exception):
    """Base exception for VLM operations"""
    pass

class VLMConfigurationError(VLMException):
    """Configuration related errors"""
    pass

class VLMProcessingError(VLMException):
    """Processing related errors"""
    pass

class VLMRateLimitError(VLMException):
    """Rate limit related errors"""
    pass

class VLMBase(ABC):
    """
    Abstract base class for Vision Language Models (VLMs)
    
    This class defines the standard interface that all VLM implementations must follow.
    It provides common functionality and enforces consistent behavior across different
    VLM providers (Claude, Gemini, GPT-4V, etc.).
    """
    
    def __init__(self, config: VLMConfig):
        self.config = config
        self._validate_config()
        self._setup_client()
    
    def _validate_config(self) -> None:
        """Validate the configuration"""
        if not self.config.api_key:
            raise VLMConfigurationError("API key is required")
        if not self.config.model_name:
            raise VLMConfigurationError("Model name is required")
        if self.config.max_tokens <= 0:
            raise VLMConfigurationError("max_tokens must be positive")
        if not 0 <= self.config.temperature <= 2:
            raise VLMConfigurationError("temperature must be between 0 and 2")
    
    @abstractmethod
    def _setup_client(self) -> None:
        """Setup the client connection (model-specific implementation)"""
        pass
    
    def _preprocess_image(self, image_data: bytes) -> ImageInput:
        """
        Preprocess image data into standardized format
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            ImageInput: Standardized image input
        """
        try:
            # Open image to get metadata
            image = Image.open(io.BytesIO(image_data))
            
            # Determine format
            format_map = {
                'JPEG': ImageFormat.JPEG,
                'PNG': ImageFormat.PNG,
                'WEBP': ImageFormat.WEBP,
                'GIF': ImageFormat.GIF,
                'BMP': ImageFormat.BMP
            }
            
            image_format = format_map.get(image.format, ImageFormat.JPEG)
            
            # Check size constraints
            size_mb = len(image_data) / (1024 * 1024)
            if size_mb > self.config.max_image_size_mb:
                raise VLMProcessingError(
                    f"Image size {size_mb:.2f}MB exceeds limit of {self.config.max_image_size_mb}MB"
                )
            
            return ImageInput(
                data=image_data,
                format=image_format,
                width=image.width,
                height=image.height,
                size_bytes=len(image_data),
                metadata={
                    'mode': image.mode,
                    'has_transparency': image.mode in ('RGBA', 'LA', 'P')
                }
            )
            
        except Exception as e:
            raise VLMProcessingError(f"Failed to preprocess image: {str(e)}")
    
    def _encode_image_base64(self, image_data: bytes) -> str:
        """Encode image data to base64 string"""
        return base64.b64encode(image_data).decode('utf-8')
    
    @abstractmethod
    def analyze_image(self, image_data: bytes, prompt: str, 
                     analysis_type: AnalysisType = AnalysisType.GENERAL_DESCRIPTION) -> VLMResponse:
        """
        Analyze a single image with the given prompt
        
        Args:
            image_data: Raw image bytes
            prompt: Analysis prompt/question
            analysis_type: Type of analysis to perform
            
        Returns:
            VLMResponse: Analysis results
        """
        pass
    
    @abstractmethod
    def analyze_multiple_images(self, images: List[bytes], prompt: str,
                               analysis_type: AnalysisType = AnalysisType.COMPARATIVE_ANALYSIS) -> VLMResponse:
        """
        Analyze multiple images with the given prompt
        
        Args:
            images: List of raw image bytes
            prompt: Analysis prompt/question
            analysis_type: Type of analysis to perform
            
        Returns:
            VLMResponse: Analysis results
        """
        pass
    
    @abstractmethod
    def batch_analyze_images(self, images: List[bytes], prompts: List[str],
                            analysis_type: AnalysisType = AnalysisType.GENERAL_DESCRIPTION) -> BatchVLMResponse:
        """
        Analyze multiple images with corresponding prompts in batch
        
        Args:
            images: List of raw image bytes
            prompts: List of prompts (must match length of images)
            analysis_type: Type of analysis to perform
            
        Returns:
            BatchVLMResponse: Batch analysis results
        """
        pass
    
    @abstractmethod
    def extract_text_from_image(self, image_data: bytes, 
                               language: Optional[str] = None) -> VLMResponse:
        """
        Extract text from image (OCR functionality)
        
        Args:
            image_data: Raw image bytes
            language: Optional language hint for OCR
            
        Returns:
            VLMResponse: Extracted text results
        """
        pass
    
    @abstractmethod
    def visual_question_answering(self, image_data: bytes, question: str) -> VLMResponse:
        """
        Answer questions about image content
        
        Args:
            image_data: Raw image bytes
            question: Question about the image
            
        Returns:
            VLMResponse: Answer to the question
        """
        pass
    
    @abstractmethod
    def detect_objects(self, image_data: bytes, 
                      object_classes: Optional[List[str]] = None) -> VLMResponse:
        """
        Detect objects in the image
        
        Args:
            image_data: Raw image bytes
            object_classes: Optional list of specific object classes to detect
            
        Returns:
            VLMResponse: Object detection results
        """
        pass
    
    @abstractmethod
    def describe_scene(self, image_data: bytes, 
                      detail_level: str = "medium") -> VLMResponse:
        """
        Provide detailed scene description
        
        Args:
            image_data: Raw image bytes
            detail_level: Level of detail ("low", "medium", "high")
            
        Returns:
            VLMResponse: Scene description
        """
        pass
    
    def analyze_image_stream(self, image_data: bytes, prompt: str) -> AsyncIterator[StreamingVLMResponse]:
        """
        Analyze image with streaming response (if supported)
        
        Args:
            image_data: Raw image bytes
            prompt: Analysis prompt
            
        Yields:
            StreamingVLMResponse: Streaming response chunks
        """
        if not self.config.enable_streaming:
            raise VLMConfigurationError("Streaming is not enabled for this configuration")
        
        # Default implementation - override in subclasses that support streaming
        response = self.analyze_image(image_data, prompt)
        yield StreamingVLMResponse(
            chunk=response.response_text,
            is_complete=True,
            chunk_index=0,
            total_chunks=1
        )
    
    def compare_images(self, image1_data: bytes, image2_data: bytes, 
                      comparison_aspects: List[str] = None) -> VLMResponse:
        """
        Compare two images
        
        Args:
            image1_data: First image bytes
            image2_data: Second image bytes
            comparison_aspects: Specific aspects to compare
            
        Returns:
            VLMResponse: Comparison results
        """
        aspects = comparison_aspects or ["similarity", "differences", "style", "content"]
        prompt = f"Compare these two images focusing on: {', '.join(aspects)}"
        
        return self.analyze_multiple_images(
            [image1_data, image2_data], 
            prompt, 
            AnalysisType.COMPARATIVE_ANALYSIS
        )
    
    def moderate_content(self, image_data: bytes) -> VLMResponse:
        """
        Moderate image content for safety
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            VLMResponse: Content moderation results
        """
        prompt = """Analyze this image for potentially inappropriate content including:
        - Violence or gore
        - Sexual content
        - Hate symbols or offensive material
        - Dangerous activities
        
        Provide a safety rating and explain any concerns."""
        
        return self.analyze_image(image_data, prompt, AnalysisType.CONTENT_MODERATION)
    
    def analyze_style(self, image_data: bytes) -> VLMResponse:
        """
        Analyze artistic style of the image
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            VLMResponse: Style analysis results
        """
        prompt = """Analyze the artistic style of this image including:
        - Art movement or style (e.g., impressionism, modernism)
        - Color palette and composition
        - Technical qualities
        - Mood and atmosphere"""
        
        return self.analyze_image(image_data, prompt, AnalysisType.STYLE_ANALYSIS)
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model
        
        Returns:
            Dict containing model information
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if the model is healthy and accessible
        
        Returns:
            bool: True if healthy, False otherwise
        """
        pass

    def _enhance_prompt_for_analysis_type(self, prompt: str, analysis_type: AnalysisType) -> str:
        """Enhance prompt based on analysis type"""
        enhancements = {
            AnalysisType.GENERAL_DESCRIPTION: "Provide a comprehensive description of this image. ",
            AnalysisType.OBJECT_DETECTION: "Focus on identifying and listing all objects visible in this image. ",
            AnalysisType.TEXT_EXTRACTION: "Extract and transcribe all text visible in this image. ",
            AnalysisType.VISUAL_QA: "Answer the question based on what you can observe in this image. ",
            AnalysisType.SCENE_UNDERSTANDING: "Analyze and describe the overall scene, setting, and context. ",
            AnalysisType.COMPARATIVE_ANALYSIS: "Compare and analyze the provided images, noting similarities and differences. ",
            AnalysisType.STYLE_ANALYSIS: "Analyze the artistic style, composition, and visual elements. ",
            AnalysisType.CONTENT_MODERATION: "Analyze this image for any potentially inappropriate or harmful content. "
        }
        
        enhancement = enhancements.get(analysis_type, "")
        return enhancement + prompt
    
    def _enhance_prompt_for_json(self, prompt: str, kv_pair: dict, **kwargs) -> str:
        """
        Enhances a base prompt by generating JSON-like attribute instructions recursively from a nested dict.
        """
        obj = kwargs.get("obj", "object")

        def format_attributes(d: dict, parent_key=""):
            items = []
            for k, v in d.items():
                full_key = k
                if isinstance(v, dict):
                    items.extend(format_attributes(v, full_key))
                else:
                    example = f" (e.g., {v})" if isinstance(v, str) else ""
                    items.append(f"'{full_key}': <{full_key}>{example}")
            return items

        if kv_pair:
            attributes = format_attributes(kv_pair)
            attribute_description = ", ".join(attributes)
            enhancement = (
                f"For each detected {obj}, provide the following information in JSON format: "
                f"{{'object': {{{attribute_description}}}}}. "
            )
            return f"{prompt.strip()} {enhancement}"
        else:
            return prompt


    def get_supported_formats(self) -> List[ImageFormat]:
        """Get list of supported image formats"""
        return self.config.supported_formats or [
            ImageFormat.JPEG, 
            ImageFormat.PNG, 
            ImageFormat.WEBP
        ]
    
    def estimate_cost(self, image_data: bytes, prompt: str) -> Optional[float]:
        """
        Estimate the cost of processing (if cost info is available)
        
        Args:
            image_data: Raw image bytes
            prompt: Analysis prompt
            
        Returns:
            Optional[float]: Estimated cost in USD, None if not available
        """
        # Base implementation - override in subclasses with cost info
        return None
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model_name})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model_name}, provider={self.__class__.__module__})"