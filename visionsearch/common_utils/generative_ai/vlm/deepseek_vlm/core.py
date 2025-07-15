import asyncio
import io
import logging
import re
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import AutoModelForCausalLM

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

from common_utils.generative_ai.vlm.deepseek_vlm.deepseek_vl.deepseek_vl.models import (
    VLChatProcessor,
)

from common_utils.generative_ai.vlm.deepseek_vlm.deepseek_vl.deepseek_vl.utils.io import (
    load_pil_images,
)


class DeepseekVLM(VLMBase):
    def __init__(self, config:VLMConfig):
        self.logger = logging.getLogger(__name__)
        
        # Track rate limiting
        super().__init__(config)
        self._last_request_time = 0
        self._request_count = 0
        self._request_window_start = time.time()

    def _setup_client(self):
        model_path = "deepseek-ai/deepseek-vl-7b-chat"
        self.vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        self.vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.vl_gpt.to(torch.bfloat16).cuda().eval()
 
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
    
    def _infer_from_deepseek_vl(self, prompt: str, image_data: Union[bytes, List[bytes]]):
        try:
            data = image_data if isinstance(image_data, list) else [image_data]
            conversation = [
                {
                    "role": "User",
                    "content": [f"<image_placeholder>{prompt}" for _ in range(len(data))],
                    "images": data
                },
                {
                    "role": "Assistant",
                    "content": ""
                }
            ]

            pil_images = load_pil_images(conversation)
        
            prepare_inputs = self.vl_chat_processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True
            ).to(self.vl_gpt.device)

            # run image encoder to get the image embeddings
            inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

            # run the model to get the response
            outputs = self.vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True
            )

            answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            return answer

        except Exception as e:
            raise VLMProcessingError(f"Encountered error during inference: {e}")

    def analyze_image(self, image_data: bytes, prompt: str, 
                     analysis_type: AnalysisType = AnalysisType.GENERAL_DESCRIPTION) -> VLMResponse:
        """Analyze a single image with the given prompt"""
        start_time = time.time()
        
        try:

            enhanced_prompt = self._enhance_prompt_for_analysis_type(prompt, analysis_type)

            response_text = self._infer_from_deepseek_vl(prompt=enhanced_prompt, image_data=image_data)
        
            response_text = response_text if response_text else ""
            
            # Extract structured data based on analysis type
            detected_objects = None
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
                detected_objects=detected_objects,
                extracted_text=extracted_text,
                processing_time_ms=processing_time,
                model_info={"provider": "deepseek", "model": self.config.model_name}
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
            response_text = self._infer_from_deepseek_vl(prompt=enhanced_prompt, image_data=images)
            print(response_text)
            processing_time = int((time.time() - start_time) * 1000)
            
            return VLMResponse(
                success=True,
                response_text=response_text,
                confidence=self._extract_confidence_level(response_text),
                analysis_type=analysis_type,
                processing_time_ms=processing_time,
                model_info={
                    "provider": "deepseek",
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
        raise NotImplemented("Deepseek VLM doesn't support object detection")
        
        
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
       raise NotImplementedError("DeepseekVLM class doesn't support image streams yet")
            
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

    prompts = [prompt, prompt]
    
    image_data = []
    with open('/home/appuser/src/archive/AGR_gate03_Gate_3_2025-05-27_07-06-52_f6905251-1323-4638-b682-e7bc4251e82a.jpg', 'rb') as f:
        image_data.append(f.read())

    with open('/home/appuser/src/archive/AGR_gate03_Gate_3_2025-05-27_07-06-52_f6905251-1323-4638-b682-e7bc4251e82a.jpg', 'rb') as f:
        image_data.append(f.read())
    import os
    from common_utils.generative_ai.vlm.base import VLMConfig
    config = VLMConfig(
        api_key="xddxdxddxdd",
        model_name="deepseek-vl-7b-chat",
    )
    vlm = DeepseekVLM(config=config)
    response = vlm.analyze_multiple_images(images=image_data, analysis_type=AnalysisType.SCENE_UNDERSTANDING, prompt=prompt)
    print(response)