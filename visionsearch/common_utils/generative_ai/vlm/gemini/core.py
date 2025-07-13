
import re
import time
import logging
import asyncio
import io
from PIL import Image
from google.genai import types
from google import genai
from typing import List, Dict, Any, Optional, AsyncIterator
from common_utils.generative_ai.vlm.base import  (
    VLMBase, VLMConfig, VLMResponse, BatchVLMResponse, StreamingVLMResponse,
    ImageInput, DetectedObject, ExtractedText, AnalysisType, ConfidenceLevel,
    ImageFormat, VLMException, VLMProcessingError, VLMRateLimitError, VLMConfigurationError
)

class GeminiVLM(VLMBase):
    def __init__(self, config:VLMConfig):
        self.logger = logging.getLogger(__name__)
        
        # Track rate limiting
        super().__init__(config)
        self._last_request_time = 0
        self._request_count = 0
        self._request_window_start = time.time()

    def _setup_client(self):
        self.client = genai.Client(api_key=self.config.api_key)


    def _prepare_image_for_gemini(self, image_data: bytes):
        """Prepare image data for Gemini API"""
        image_input = self._preprocess_image(image_data)
        encoded_image = self._encode_image_base64(image_data)
        
        # Gemini expects image data in specific format
        return types.Part.from_bytes(
            mime_type=f"image/{image_input.format.value}",
            data=encoded_image
        )
    
    def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting"""
        current_time = time.time()
        
        # Reset window if needed (per minute)
        if current_time - self._request_window_start >= 60:
            self._request_count = 0
            self._request_window_start = current_time
        
        # Check if we're hitting rate limits
        if self._request_count >= self.config.rate_limit_per_minute:
            wait_time = 60 - (current_time - self._request_window_start)
            if wait_time > 0:
                raise VLMRateLimitError(f"Rate limit exceeded. Wait {wait_time:.1f} seconds.")
        
        # Minimum delay between requests
        time_since_last = current_time - self._last_request_time
        if time_since_last < 1.0:  # At least 1 second between requests
            time.sleep(1.0 - time_since_last)
        
        self._request_count += 1
        self._last_request_time = time.time()

    def _extract_confidence_level(self, response_text: str) -> ConfidenceLevel:
        """Extract confidence level from response text"""
        text_lower = response_text.lower()
        
        # Look for confidence indicators
        high_confidence_words = ["confident", "certain", "clearly", "definitely", "obviously"]
        low_confidence_words = ["might", "possibly", "perhaps", "unclear", "uncertain"]
        
        high_count = sum(1 for word in high_confidence_words if word in text_lower)
        low_count = sum(1 for word in low_confidence_words if word in text_lower)
        
        if high_count > low_count:
            return ConfidenceLevel.HIGH
        elif low_count > high_count:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.MEDIUM
    
    def _extract_objects_from_response(self, response_text: str) -> List[DetectedObject]:
        """Extract detected objects from response text"""
        objects = []
        
        # Pattern to match object descriptions
        # This is a simplified approach - in production, you might want more sophisticated parsing
        object_patterns = [
            r"(?:I can see|There (?:is|are)|I notice|I observe|I detect)\s+(?:a|an|some|several|many)?\s*([^.]+?)(?:\s+(?:in|on|at|near))?",
            r"Objects?:\s*([^.]+)",
            r"Items?:\s*([^.]+)",
        ]
        
        for pattern in object_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            for match in matches:
                # Clean up the match
                cleaned = re.sub(r'\s+', ' ', match).strip()
                if cleaned and len(cleaned) > 2:
                    objects.append(DetectedObject(
                        label=cleaned,
                        confidence=0.8,  # Default confidence
                        attributes={"source": "gemini_text_extraction"}
                    ))
        
        return objects[:10]  # Limit to top 10 objects
    
    def _extract_text_from_response(self, response_text: str) -> List[ExtractedText]:
        """Extract text mentions from response"""
        text_items = []
        
        # Pattern to match text extraction
        text_patterns = [
            r"(?:text|writing|words?|says?|reads?|written):\s*[\"']([^\"']+)[\"']",
            r"[\"']([^\"']+)[\"']",
            r"(?:The text|Text|It says|Writing):\s*([^.]+)",
        ]
        
        for pattern in text_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            for match in matches:
                cleaned = match.strip()
                if cleaned and len(cleaned) > 2:
                    text_items.append(ExtractedText(
                        text=cleaned,
                        confidence=0.7,  # Default confidence
                        language="en"  # Default to English
                    ))
        
        return text_items[:5]  # Limit to top 5 text items

    def _make_request_with_retry(self, prompt: str, image_data: Optional[bytes] = None, 
                                images: Optional[List[bytes]] = None) -> genai.types.GenerateContentResponse:
        """Make request with retry logic"""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                self._check_rate_limit()
                
                # Prepare content
                content = [prompt]
                if image_data:
                    content.append(self._prepare_image_for_gemini(image_data))
                elif images:
                    for img in images:
                        content.append(self._prepare_image_for_gemini(img))
                
                # Make the request
                response = self.client.models.generate_content(
                    model=self.config.model_name,
                    contents=content
                )
                
                # Check if response was blocked
                if response.candidates and response.candidates[0].finish_reason.name == "SAFETY":
                    raise VLMProcessingError("Content was blocked by safety filters")
                
                return response
                
            except VLMRateLimitError:
                raise  # Don't retry rate limit errors
            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    wait_time = self.config.retry_delay_seconds * (2 ** attempt)
                    self.logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {str(e)}")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Request failed after {self.config.max_retries + 1} attempts: {str(e)}")
        
        raise VLMProcessingError(f"Request failed after retries: {str(last_exception)}")

    def analyze_image(self, image_data: bytes, prompt: str, 
                     analysis_type: AnalysisType = AnalysisType.GENERAL_DESCRIPTION) -> VLMResponse:
        """Analyze a single image with the given prompt"""
        start_time = time.time()
        
        try:

            enhanced_prompt = self._enhance_prompt_for_analysis_type(prompt, analysis_type)

            print(enhanced_prompt)
            response = self._make_request_with_retry(prompt=enhanced_prompt, image_data=image_data)

            response_text = response.text if response.text else ""
            
            # Extract structured data based on analysis type
            detected_objects = None
            extracted_text = None
            
            if analysis_type == AnalysisType.OBJECT_DETECTION:
                detected_objects = self._extract_objects_from_response(response_text)
            elif analysis_type == AnalysisType.TEXT_EXTRACTION:
                extracted_text = self._extract_text_from_response(response_text)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            print(processing_time)
            return VLMResponse(
                success=True,
                response_text=response.text,
                confidence=ConfidenceLevel.HIGH,
                analysis_type=analysis_type,
                detected_objects=detected_objects,
                extracted_text=extracted_text,
                processing_time_ms=processing_time,
                model_info={"provider": "gemini", "model": self.config.model_name},
                # raw_response={
                #     "candidates": len(response.candidates) if response.candidates else 0,
                #     "finish_reason": response.candidates[0].finish_reason.name if response.candidates else None,
                #     "safety_ratings": [
                #         {
                #             "category": rating.category.name,
                #             "probability": rating.probability.name
                #         }
                #         for rating in (response.candidates[0].safety_ratings if response.candidates else [])
                #     ]
                # }
            )
        except Exception as e:
            return VLMResponse(
                success=False,
                response_text="",
                confidence=ConfidenceLevel.LOW,
                analysis_type=analysis_type,
                error_message=str(e),
                processing_time_ms=int((time.time() - start_time) * 1000)
            )

    def describe_scene(self, image_data: bytes, 
                      detail_level: str = "medium") -> VLMResponse:
        """Provide detailed scene description"""
        detail_instructions = {
            "low": "Provide a brief, general description of the scene.",
            "medium": "Provide a detailed description including main objects, setting, and activities.",
            "high": "Provide a comprehensive description including all visible details, colors, textures, lighting, mood, and any subtle elements."
        }
        
        instruction = detail_instructions.get(detail_level, detail_instructions["medium"])
        
        prompt = f"""
        {instruction}
        
        Please structure your response to include:
        1. Overall scene setting and context
        2. Main subjects and objects
        3. Visual details (colors, lighting, composition)
        4. Mood and atmosphere
        5. Any notable or interesting elements
        """
        
        return self.analyze_image(image_data, prompt, AnalysisType.SCENE_UNDERSTANDING)
    
    def analyze_multiple_images(self, images: List[bytes], prompt: str,
                               analysis_type: AnalysisType = AnalysisType.COMPARATIVE_ANALYSIS) -> VLMResponse:
        """Analyze multiple images with the given prompt"""
        start_time = time.time()
        
        try:
            # Enhance prompt for multiple images
            enhanced_prompt = f"""
            Analyze these {len(images)} images: {prompt}
            
            Please provide a comprehensive analysis comparing and contrasting the images.
            """
            
            # Make the request
            response = self._make_request_with_retry(enhanced_prompt, images=images)
            
            response_text = response.text if response.text else ""
            processing_time = int((time.time() - start_time) * 1000)
            
            return VLMResponse(
                success=True,
                response_text=response_text,
                confidence=self._extract_confidence_level(response_text),
                analysis_type=analysis_type,
                processing_time_ms=processing_time,
                model_info={
                    "provider": "gemini",
                    "model": self.config.model_name,
                    "images_processed": len(images)
                }
            )
            
        except Exception as e:
            return VLMResponse(
                success=False,
                response_text="",
                confidence=ConfidenceLevel.LOW,
                analysis_type=analysis_type,
                error_message=str(e),
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
    
    def batch_analyze_images(self, images: List[bytes], prompts: List[str],
                            analysis_type: AnalysisType = AnalysisType.GENERAL_DESCRIPTION) -> BatchVLMResponse:
        """Analyze multiple images with corresponding prompts in batch"""
        start_time = time.time()
        
        if len(images) != len(prompts):
            raise VLMProcessingError("Number of images must match number of prompts")
        
        responses = []
        failed_count = 0
        errors = []
        
        for i, (image_data, prompt) in enumerate(zip(images, prompts)):
            try:
                response = self.analyze_image(image_data, prompt, analysis_type)
                responses.append(response)
                
                if not response.success:
                    failed_count += 1
                    errors.append(f"Image {i+1}: {response.error_message}")
                    
            except Exception as e:
                failed_count += 1
                error_msg = f"Image {i+1}: {str(e)}"
                errors.append(error_msg)
                
                # Create error response
                responses.append(VLMResponse(
                    success=False,
                    response_text="",
                    confidence=ConfidenceLevel.LOW,
                    analysis_type=analysis_type,
                    error_message=str(e)
                ))
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return BatchVLMResponse(
            success=failed_count == 0,
            responses=responses,
            total_processed=len(images),
            failed_count=failed_count,
            processing_time_ms=processing_time,
            errors=errors if errors else None
        )
    
    def extract_text_from_image(self, image_data: bytes, 
                               language: Optional[str] = None) -> VLMResponse:
        """Extract text from image (OCR functionality)"""
        lang_hint = f" (text is in {language})" if language else ""
        prompt = f"""
        Extract all text visible in this image{lang_hint}. 
        
        Please provide:
        1. The exact text content
        2. Any formatting or structure you observe
        3. Confidence level for each piece of text
        
        If no text is visible, please state that clearly.
        """
        
        response = self.analyze_image(image_data, prompt, AnalysisType.TEXT_EXTRACTION)
        
        # If successful, try to extract more structured text data
        if response.success and not response.extracted_text:
            response.extracted_text = self._extract_text_from_response(response.response_text)
        
        return response
    
    def visual_question_answering(self, image_data: bytes, question: str) -> VLMResponse:
        """Answer questions about image content"""
        prompt = f"""
        Please answer the following question about this image: {question}
        
        Provide a clear, detailed answer based on what you can observe in the image.
        If you cannot determine the answer from the image, please say so.
        """
        
        return self.analyze_image(image_data, prompt, AnalysisType.VISUAL_QA)
    
    def detect_objects(self, image_data: bytes, 
                      object_classes: Optional[List[str]] = None) -> VLMResponse:
        """Detect objects in the image"""
        if object_classes:
            prompt = f"""
            Look for these specific objects in the image: {', '.join(object_classes)}
            
            For each object you find, please provide:
            1. The object name
            2. Your confidence level
            3. A brief description of where it appears
            4. Any notable characteristics
            """
        else:
            prompt = """
            Identify and describe all objects, people, animals, and items visible in this image.
            
            Please provide:
            1. A comprehensive list of everything you can see
            2. Their approximate locations
            3. Any relationships between objects
            4. Notable characteristics or details
            """
        
        response = self.analyze_image(image_data, prompt, AnalysisType.OBJECT_DETECTION)
        
        # Extract structured object data
        if response.success and not response.detected_objects:
            response.detected_objects = self._extract_objects_from_response(response.response_text)
        
        return response
    
    async def analyze_image_stream(self, image_data: bytes, prompt: str) -> AsyncIterator[StreamingVLMResponse]:
        """Analyze image with streaming response"""
        if not self.config.enable_streaming:
            raise VLMConfigurationError("Streaming is not enabled for this configuration")
        
        # Gemini supports streaming, but we'll implement a simple version
        # In practice, you'd use the actual streaming API
        response = self.analyze_image(image_data, prompt)
        
        # Simulate streaming by chunking the response
        text = response.response_text
        chunk_size = 50  # Characters per chunk
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        for i, chunk in enumerate(chunks):
            yield StreamingVLMResponse(
                chunk=chunk,
                is_complete=(i == len(chunks) - 1),
                chunk_index=i,
                total_chunks=len(chunks),
                metadata={"analysis_type": response.analysis_type.value}
            )
            await asyncio.sleep(0.5)  # Simulate streaming delay
            
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        return {
            "provider": "google",
            "model_name": self.config.model_name,
            "model_type": "vision_language_model",
            "capabilities": [
                "image_analysis",
                "text_extraction",
                "object_detection",
                "visual_question_answering",
                "scene_understanding",
                "multi_image_analysis"
            ],
            "supported_formats": [fmt.value for fmt in self.get_supported_formats()],
            "max_image_size_mb": self.config.max_image_size_mb,
            "max_tokens": self.config.max_tokens,
            "supports_streaming": self.config.enable_streaming,
            "rate_limits": {
                "requests_per_minute": self.config.rate_limit_per_minute,
                "timeout_seconds": self.config.timeout_seconds
            }
        }
    
    def health_check(self) -> bool:
        """Check if the model is healthy and accessible"""
        try:
            # Create a simple test image (1x1 pixel)
            test_image = Image.new('RGB', (1, 1), color='red')
            img_byte_arr = io.BytesIO()
            test_image.save(img_byte_arr, format='PNG')
            test_image_data = img_byte_arr.getvalue()
            
            # Simple test prompt
            test_prompt = "What color is this image?"
            
            # Make test request
            response = self.analyze_image(test_image_data, test_prompt)
            
            return response.success
            
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return False
    
    def estimate_cost(self, image_data: bytes, prompt: str) -> Optional[float]:
        """Estimate the cost of processing"""
        # Gemini Pro Vision pricing (approximate)
        # Note: Actual pricing should be fetched from Google Cloud Pricing API
        
        # Base cost per image
        base_cost = 0.0025  # $0.0025 per image
        
        # Text token cost (rough estimate)
        estimated_output_tokens = min(self.config.max_tokens, len(prompt) * 1.5)
        token_cost = estimated_output_tokens * 0.000005  # $0.000005 per token
        
        total_cost = base_cost + token_cost
        
        return round(total_cost, 6)
    
    def get_supported_formats(self) -> List[ImageFormat]:
        """Get list of supported image formats"""
        return [
            ImageFormat.JPEG,
            ImageFormat.PNG,
            ImageFormat.WEBP,
            ImageFormat.GIF
        ]


if __name__ == "__main__":
    prompt = (
        'Caption this image, '
        'note that this a gate of a waste bunker, '
        'can you please specify why kind of waste objects and material do you see. '
        'highlight if you see any solid pipes, metal chunks, and other waste objects that might be harmful to the plant boiler'
    )

    with open('/home/appuser/src/archive/AGR_gate03_Gate_3_2025-05-27_07-06-52_f6905251-1323-4638-b682-e7bc4251e82a.jpg', 'rb') as f:
        image_bytes = f.read()

    import os
    from common_utils.generative_ai.vlm.base import VLMConfig
    config = VLMConfig(
        api_key=os.getenv('GOOGLE_API_KEY', ''),
        model_name="gemini-2.5-flash",
    )
    vlm = GeminiVLM(config=config)
    response = vlm.analyze_image(image_data=image_bytes, analysis_type=AnalysisType.SCENE_UNDERSTANDING, prompt=prompt)
    print(response.response_text)