import ast
import gc
import io
import logging
import os
import re
import shutil
import tempfile
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import cv2
import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from langchain.prompts import ChatPromptTemplate

from common_utils.adapters.ollama import OllamaAdapter
from common_utils.generative_ai.vlm.base import (
    AnalysisType,
    BatchVLMResponse,
    ConfidenceLevel,
    ExtractedText,
    ImageFormat,
    ModelSource,
    StreamingVLMResponse,
    VideoAnalysisArguments,
    VideoBatchResponse,
    VideoFormat,
    VideoVLMResponse,
    VLMBase,
    VLMConfig,
    VLMProcessingError,
    VLMResponse,
)


class GemmaVLM(VLMBase):
    def __init__(self, config:VLMConfig, system_prompt):
        self.logger = logging.getLogger(__name__)
        
        super().__init__(config)
        self._last_request_time = 0
        self._request_count = 0
        self._request_window_start = time.time()
        self.system_prompt = system_prompt

    def _setup_client(self):
        if self.config.model_source == ModelSource.LOCAL:
            token = os.getenv("HF_TOKEN")
            model_id = "google/gemma-3-4b-it"
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_id, token=token
            ).eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            self.processor = AutoProcessor.from_pretrained(model_id, token=token)

        elif self.config.model_source == ModelSource.OLLAMA:
            print("Hi")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
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
        pil_images = []
        try:
            if isinstance(image_data, bytes):
                image_data = [image_data]
            for img_bytes in image_data:
                pil_image = Image.open(io.BytesIO(img_bytes))
                pil_image.verify()  # Will raise error if the image is not complete or corrupt

                # Reopen if needed after verify
                pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                pil_images.append(pil_image)
           
        except UnidentifiedImageError:
            raise VLMProcessingError(f"Cannot convert image into PIL Image, might be corrupted")
        
        except Exception as e:
            raise VLMProcessingError(f"Unexpected error: {str(e)}")
        
        try:
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
                     analysis_type: AnalysisType = AnalysisType.GENERAL_DESCRIPTION, **kwargs) -> VLMResponse:
        """Analyze a single image with the given prompt"""
        start_time = time.time()
        
        try:
            if self.config.enable_json:
               json_obj_attributes = kwargs.get("json_obj_attributes", [])
               json_img_attributes = kwargs.get("json_img_attributes", [])
               prompt = self._enhance_prompt_for_json(prompt, json_obj_attributes, json_img_attributes)
            else: 
                json_dict = {}
            
            enhanced_prompt = self._enhance_prompt_for_analysis_type(prompt, analysis_type)

            response_text = self._infer_from_model(prompt=enhanced_prompt, image_data=image_data)
        
            response_text = response_text if response_text else ""
            
            # Extract structured data based on analysis type
            extracted_text = None
            
            if analysis_type == AnalysisType.TEXT_EXTRACTION:
                extracted_text = self._extract_text_from_response(response_text)
            if self.config.enable_json:
                json_dict = self._extract_json_from_response(response_text)
            processing_time = int((time.time() - start_time) * 1000)
            
            return VLMResponse(
                success=True,
                response_text=response_text,
                confidence=ConfidenceLevel.HIGH,
                analysis_type=analysis_type,
                extracted_text=extracted_text,
                processing_time_ms=processing_time,
                model_info={"provider": "google", "model": self.config.model_name},
                raw_response=json_dict
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
               json_obj_attributes = kwargs.get("json_obj_attributes", [])
               json_img_attributes = kwargs.get("json_img_attributes", [])
               prompt = self._enhance_prompt_for_json(prompt, json_obj_attributes, json_img_attributes)
            else: 
                json_dict = {}

            enhanced_prompt = f"""
            Analyze these {len(images)} images: {prompt}
            
            Please provide a comprehensive analysis comparing and contrasting the images.
            """
            enhanced_prompt = prompt
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
                json_obj_attributes = kwargs.get("json_obj_attributes", [])
                json_img_attributes = kwargs.get("json_img_attributes", [])
                prompt = self._enhance_prompt_for_json(prompt, json_obj_attributes, json_img_attributes)
            else: 
                json_dict = {}
                
            enhanced_prompt = reference_desription + prompt

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
    
    def analyze_video(
            self, 
            video_data: bytes,  
            prompt: str, 
            args: VideoAnalysisArguments
        ):
        start_time = time.time()
        if isinstance(prompt, ChatPromptTemplate):
            prompt_template = prompt
            messages = prompt_template.format_prompt(input="None").to_messages()
            self.system_prompt = messages[0].content
            prompt = messages[1].content

        try:
            tmp_dir = "/tmp/gemma_video_analysis/"
            os.makedirs(tmp_dir, exist_ok=True)
            if args.video_format == VideoFormat.MP4:
                suffix = ".mp4"
            elif args.video_format == VideoFormat.AVI:
                suffix = ".avi"
            elif args.video_format == VideoFormat.MOV:
                suffix = ".mov"
            else:
                raise VLMProcessingError(f"Video Format: {args.video_format} not supported for Gemma")
        except Exception as e:
            return VideoVLMResponse(
                success=False,
                response_per_frames=[],
                processing_time_ms=0,
                errors=[e]
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(video_data)
                tmp_path = tmp_file.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise VLMProcessingError("There is some error with the video")
        
        frames = []
        last_response = None
        video_batch_responses = []
        errors = []

        original_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        if original_fps <= 0:
            original_fps = 30

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        saved_count = 0
        start_second = 0

        # Calculate frame indices at given interval
        interval = max(int(original_fps * args.target_seconds), 1)
        frame_indices = list(range(0, total_frames, interval))

        # Ensure last frame is included
        if frame_indices[-1] != (total_frames - 1):
            frame_indices.append(total_frames - 1)

        for x in frame_indices:
            if start_second == -1:
                start_second = x / original_fps
            cap.set(cv2.CAP_PROP_POS_FRAMES, x)
            ret, frame = cap.read()
            if not ret:
                continue

            # Save every 'frame_interval'-th frame
            frame_filename = os.path.join(tmp_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            with open(frame_filename, "rb") as f:
                frames.append(f.read())

            # Generate prompt with previous analysis
            if last_response:
                messages = prompt_template.format_prompt(input=last_response).to_messages()
                prompt = messages[1].content

            # Generate prompt without previous analysis
            else:
                messages = prompt_template.format_prompt(input="not available").to_messages()
                prompt = messages[1].content

            # Analyze batch of frames
            if len(frames) == args.batch_per_frames or x == total_frames - 1:
                try:
                    end_second = x / original_fps
                    response = self.analyze_multiple_images(frames, prompt, args.analysis_type)
                    batch_response = VideoBatchResponse(
                        response=response.response_text,
                        start_second=start_second,
                        end_second=end_second
                    )
                    video_batch_responses.append(batch_response)

                    start_second = -1

                except Exception as e:
                    errors.append(e)
                finally:
                    # Clear resources
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    gc.collect()

                    frames = []
        
        cap.release()
        os.remove(tmp_path)
        shutil.rmtree(tmp_dir)

        report = ""
        if args.generate_report:
            summaries = "\n".join([batch_response.response for batch_response in video_batch_responses])
            self.system_prompt = "You're an AI Assistant processing video summaries from different timeframes for one delivery."
            prompt = (
                f"You are instructed to use the following {summaries} to generate an informative and concise report "
                "summarizing what happened in the video. The report should highlight the most important events during the delivery."
            )

            report = self._infer_from_model(prompt=prompt, image_data=[])

        return VideoVLMResponse(
            success=True,
            responses_per_batch=video_batch_responses,
            processing_time_ms=int((time.time() - start_time) * 1000),
            errors=errors,
            video_meta_information={
                "duration": total_frames / original_fps,
                "fps": original_fps
            },
            report=report
        )
        
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
