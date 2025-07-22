import ast
import asyncio
import io
import logging
import re
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from common_utils.generative_ai.vlm.base import (
    AnalysisType,
    BatchVLMResponse,
    ConfidenceLevel,
    DetectedObject,
    ExtractedText,
    ImageFormat,
    ImageInput,
    StreamingVLMResponse,
    VLMBase,
    VLMConfig,
    VLMConfigurationError,
    VLMException,
    VLMProcessingError,
    VLMRateLimitError,
    VLMResponse,
)


class GemmaVLM(VLMBase):
    def __init__(self, config:VLMConfig, system_prompt):
        self.logger = logging.getLogger(__name__)
        
        # Track rate limiting
        super().__init__(config)
        self._last_request_time = 0
        self._request_count = 0
        self._request_window_start = time.time()
        self.system_prompt = system_prompt

    def _setup_client(self):
        model_id = "google/gemma-3-4b-it"
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, device_map="auto"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    
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
    
    def _extract_json_from_response(self, response_text: str) -> dict:
        json_string = response_text.split('```json')[1]
        json_string = json_string.split('```')[0]
        return ast.literal_eval(json_string)
     
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
    
    def _infer_from_model(self, prompt: str, image_data: Union[bytes, List[bytes]]):
        try:
            if isinstance(image_data, bytes):
                image_data = [image_data]

            pil_images = [Image.open(io.BytesIO(im_data)) for im_data in image_data]
            content = [
                        {"type": "image", "image": pil_image} for pil_image in pil_images
                    ]
            content.append({"type": "text", "text": prompt})

            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": self.system_prompt}
                    ]
                },
                {
                    "role": "user",
                    "content": content,
                        
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = self.model.generate(**inputs, max_new_tokens=self.config.max_tokens, do_sample=False)
                generation = generation[0][input_len:]

            answer = self.processor.decode(generation, skip_special_tokens=True)    
            return answer

        except Exception as e:
            raise VLMProcessingError(f"Encountered error during inference: {e}")

    def analyze_image(self, image_data: bytes, prompt: str, 
                     analysis_type: AnalysisType = AnalysisType.GENERAL_DESCRIPTION) -> VLMResponse:
        """Analyze a single image with the given prompt"""
        start_time = time.time()
        
        try:

            enhanced_prompt = self._enhance_prompt_for_analysis_type(prompt, analysis_type)

            response_text = self._infer_from_model(prompt=enhanced_prompt, image_data=image_data)
        
            response_text = response_text if response_text else ""
            
            # Extract structured data based on analysis type
            extracted_text = None
            
            if analysis_type == AnalysisType.TEXT_EXTRACTION:
                extracted_text = self._extract_text_from_response(response_text)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            print(f"Processing time {processing_time}")
            return VLMResponse(
                success=True,
                response_text=response_text,
                confidence=ConfidenceLevel.HIGH,
                analysis_type=analysis_type,
                extracted_text=extracted_text,
                processing_time_ms=processing_time,
                model_info={"provider": "google", "model": self.config.model_name}
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
                               analysis_type: AnalysisType = AnalysisType.COMPARATIVE_ANALYSIS,
                               **kwargs) -> VLMResponse:
        """Analyze multiple images with the given prompt"""
        start_time = time.time()
        
        try:
            # Enhance prompt for multiple images
            if self.config.enable_json:
               prompt = self._enhance_prompt_for_json(prompt, kv_pair=kwargs.get("kv_pair"))
               print(prompt)
            enhanced_prompt = f"""
            Analyze these {len(images)} images: {prompt}
            
            Please provide a comprehensive analysis comparing and contrasting the images.
            """

            print(enhanced_prompt)
            
            # Make the request
            response_text = self._infer_from_model(prompt=enhanced_prompt, image_data=images)
            if self.config.enable_json:
                json_dict = self._extract_json_from_response(response_text)

            processing_time = int((time.time() - start_time) * 1000)
            
            return VLMResponse(
                success=True,
                response_text=response_text,
                confidence=self._extract_confidence_level(response_text),
                analysis_type=analysis_type,
                processing_time_ms=processing_time,
                model_info={
                    "provider": "google",
                    "model": self.config.model_name,
                    "images_processed": len(images)
                },
                raw_response=json_dict,
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
    
    def analyze_images_with_reference_image(self, 
                                            reference_image: bytes, 
                                            images: List[bytes], 
                                            reference_desription: str, 
                                            prompt:str,
                                            analysis_type=AnalysisType.COMPARATIVE_ANALYSIS,
                                            **kwargs):
        start_time = time.time()
        try:
            if self.config.enable_json:
                prompt = self._enhance_prompt_for_json(prompt, kv_pair=kwargs.get("kv_pair"))
            

            enhanced_prompt = reference_desription + prompt
            print(enhanced_prompt) 

            image_data = [reference_image] + images
            response_text = self._infer_from_model(prompt=enhanced_prompt, image_data=image_data) 
            if self.config.enable_json:
                json_dict = self._extract_json_from_response(response_text)
            processing_time = int((time.time() - start_time) * 1000)
            
            return VLMResponse(
                success=True,
                response_text=response_text,
                confidence=self._extract_confidence_level(response_text),
                analysis_type=analysis_type,
                processing_time_ms=processing_time,
                raw_response=json_dict,
                model_info={
                    "provider": "google",
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
    def detect_objects(self, image_data: bytes, 
                    object_classes: Optional[List[str]] = None):
        raise NotImplemented("Gemma VLM doesn't support object detection")
        
        
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
      
    async def analyze_image_stream(self, image_data: bytes, prompt: str) -> AsyncIterator[StreamingVLMResponse]:
       raise NotImplementedError(f"{self.__class__.__name__} class doesn't support image streams yet")
            
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        return {
            "provider": "google",
            "model_name": self.config.model_name,
            "model_type": "vision_language_model",
            "capabilities": [
                "image_analysis",
                "text_extraction",
                "visual_question_answering",
                "scene_understanding",
                "multi_image_analysis"
            ],
            "supported_formats": [fmt.value for fmt in self.get_supported_formats()],
            "max_image_size_mb": self.config.max_image_size_mb,
            "max_tokens": self.config.max_tokens,
            "supports_streaming": self.config.enable_streaming,
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
    
   
    def get_supported_formats(self) -> List[ImageFormat]:
        """Get list of supported image formats"""
        return [
            ImageFormat.JPEG,
            ImageFormat.PNG
        ]
