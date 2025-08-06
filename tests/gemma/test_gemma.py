import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, Gemma3ForConditionalGeneration
from PIL import Image, ImageDraw, ImageFont
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WasteAnalyzer:
    """Efficient waste detection and analysis class"""
    
    def __init__(self, model_id: str = "gemma-3-4b-it", device: Optional[str] = None):
        """
        Initialize the waste analyzer
        
        Args:
            model_id: Model identifier
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.model_id = f"google/{model_id}"
        
        # Auto-detect best available device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info("Using CUDA GPU")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon
                logger.info("Using Apple MPS")
            else:
                self.device = "cpu"
                logger.info("Using CPU")
        else:
            self.device = device
        
        # Load model and processor once during initialization
        self._load_model()
        
        # Predefined waste categories for better consistency
        self.waste_categories = [
            'pipe', 'mattress', 'furniture', 'metal object', 'fabric', 
            'gas canister', 'bottle', 'rug', 'duvet', 'bed sheet', 
            'plastic bag', 'other'
        ]
    
    def _load_model(self):
        """Load model and processor with optimized settings"""
        try:
            logger.info(f"Loading model: {self.model_id}")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            
            # Optimized model loading
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                "low_cpu_mem_usage": True,
                "device_map": "auto" if self.device == "cuda" else None,
            }
            
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_id, device_map="auto"
            ).eval()
            
            # Move to device if not using device_map
            if model_kwargs["device_map"] is None:
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    # SOLUTION 1: Visual Bounding Box Overlay
    def create_visual_bbox_image(self, image: Image.Image, bbox: list, 
                                highlight_color: str = "red", 
                                line_width: int = 5) -> Image.Image:
        """
        Create image with visible bounding box overlay
        This helps the model visually identify the region of interest
        """
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        # Convert normalized coordinates to pixel coordinates
        width, height = image.size
        x_min = int(bbox[0] * width)
        y_min = int(bbox[1] * height)
        x_max = int(bbox[2] * width)
        y_max = int(bbox[3] * height)
        
        # Draw bounding box
        draw.rectangle(
            [(x_min, y_min), (x_max, y_max)], 
            outline=highlight_color, 
            width=line_width
        )
        
        # Add corner markers for better visibility
        corner_size = 10
        corners = [
            (x_min, y_min), (x_max, y_min), 
            (x_min, y_max), (x_max, y_max)
        ]
        
        for corner in corners:
            draw.rectangle([
                (corner[0] - corner_size, corner[1] - corner_size),
                (corner[0] + corner_size, corner[1] + corner_size)
            ], fill=highlight_color)
        
        # Add label
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        draw.text((x_min, y_min - 25), "TARGET OBJECT", 
                 fill=highlight_color, font=font)
        

        img_copy.save('test_visual.png')
        return img_copy
    
    # SOLUTION 2: Cropped Region Analysis
    def extract_bbox_region(self, image: Image.Image, bbox: list, 
                           padding: float = 0.1) -> Tuple[Image.Image, dict]:
        """
        Extract and analyze only the bounding box region
        More accurate but loses context
        """
        width, height = image.size
        
        # Convert to pixel coordinates with optional padding
        x_min = max(0, int((bbox[0] - padding) * width))
        y_min = max(0, int((bbox[1] - padding) * height))
        x_max = min(width, int((bbox[2] + padding) * width))
        y_max = min(height, int((bbox[3] + padding) * height))
        
        # Crop the region
        cropped = image.crop((x_min, y_min, x_max, y_max))
        
        # Return cropped image and metadata
        crop_info = {
            "original_size": (width, height),
            "crop_coords": (x_min, y_min, x_max, y_max),
            "crop_size": cropped.size,
            "padding_used": padding
        }
        

        cropped.save("test_crop.png")
        return cropped, crop_info
    
    # SOLUTION 3: Grid-based Spatial Reference
    def create_grid_reference_image(self, image: Image.Image, grid_size: int = 8) -> Image.Image:
        """
        Add grid overlay to help model understand spatial coordinates
        """
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        width, height = image.size
        
        # Draw vertical lines
        for i in range(1, grid_size):
            x = int(i * width / grid_size)
            draw.line([(x, 0), (x, height)], fill="lightgray", width=1)
        
        # Draw horizontal lines
        for i in range(1, grid_size):
            y = int(i * height / grid_size)
            draw.line([(0, y), (width, y)], fill="lightgray", width=1)
        
        # Add coordinate labels
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        for i in range(grid_size + 1):
            for j in range(grid_size + 1):
                x = int(i * width / grid_size)
                y = int(j * height / grid_size)
                coord_text = f"({i/grid_size:.1f},{j/grid_size:.1f})"
                draw.text((x + 2, y + 2), coord_text, fill="gray", font=font)
        

        img_copy.save('test_grid.png')
        return img_copy
    
    def _create_prompt(self, bbox: list, prefix:str="") -> str:
        """Create optimized prompt template"""
        categories_str = "', '".join(self.waste_categories[:-1])  # Exclude 'other'
        
        return f"""
        You are a vision-language AI trained to analyze waste material in industrial settings.
        {prefix}
        Original coordinates: {bbox} (x_min, y_min, x_max, y_max)

        This region likely contains a relevant waste object. Your task is to analyze only this object and return detailed semantic and visual properties for it.
        Describe the object with the following properties:

            "object_type": The most appropriate label from ['pipe', 'mattress', 'furniture', 'metal object', 'fabric', 'gas canister', 'bottle', 'rug', 'duvet', 'bed sheet', 'plastic bag'".

            "color": The dominant color or meaningful color pattern (e.g., 'blue and grey', 'rusted metal').

            "material": The inferred material type (e.g., plastic, metal, foam, wood, fabric, rubber, composite).

            "visibility": One of 'fully visible', 'partially occluded', or 'heavily occluded'.

            "size": Relative to the image — 'small', 'medium', or 'large'.

            "location": Spatial location in the image using terms like 'bottom right', 'center', etc.

            "confidence": Your certainty (float between 0.0 and 1.0) that the interpretation is correct.

        Output format:

        {{
        "object": {{
            "object_type": "<object_type>",
            "color": "<dominant_color>",
            "material": "<inferred_material>",
            "visibility": "<visibility_level>",
            "size": "<object_size>",
            "location": "<spatial_position>",
            "confidence": <confidence_score>
        }}
        }}

        Focus only on the content inside the bounding box. If you are unsure, use "object_type": "other" and reduce confidence accordingly.
        Only analyze the specified region. Do not describe other parts of the image.
        """

    # SOLUTION 4: Multi-stage Analysis
    def analyze_with_spatial_verification(self, image: Image.Image, bbox: list, 
                                        method: str = "visual_bbox") -> Dict[str, Any]:
        """
        Analyze waste with enhanced spatial grounding
        
        Methods:
        - 'visual_bbox': Add visible bounding box to image
        - 'cropped': Analyze only cropped region
        - 'grid_reference': Add coordinate grid
        - 'multi_stage': Combine multiple approaches
        """
        
        if method == "visual_bbox":
            return self._analyze_with_visual_bbox(image, bbox)
        elif method == "cropped":
            return self._analyze_cropped_region(image, bbox)
        elif method == "grid_reference":
            return self._analyze_with_grid(image, bbox)
        elif method == "multi_stage":
            return self._multi_stage_analysis(image, bbox)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _analyze_with_visual_bbox(self, image: Image.Image, bbox: list) -> Dict[str, Any]:
        """Analyze using visual bounding box overlay"""
        # Create image with visible bounding box
        bbox_image = self.create_visual_bbox_image(image, bbox)
        
        prefix = f"""
        You are analyzing waste in a bunker. The RED BOUNDING BOX shows the exact object to analyze.
        Focus ONLY on the object inside the red box. Ignore everything outside."""

        prompt = self._create_prompt(bbox, prefix)
        return self._generate_response(bbox_image, prompt)
    
    def _analyze_cropped_region(self, image: Image.Image, bbox: list) -> Dict[str, Any]:
        """Analyze cropped region only"""
        cropped_image, crop_info = self.extract_bbox_region(image, bbox, padding=0.05)
        
        prefix = f"""
            You are analyzing a cropped waste object from a bunker image.
            This image shows ONLY the object of interest (with small padding)."""
        
        prompt = self._create_prompt(bbox, prefix)
        result = self._generate_response(cropped_image, prompt)
        result["crop_info"] = crop_info
        return result
    
    def _analyze_with_grid(self, image: Image.Image, bbox: list) -> Dict[str, Any]:
        """Analyze using grid coordinate system"""
        grid_image = self.create_grid_reference_image(image)
        
        # Add bounding box to grid image
        bbox_grid_image = self.create_visual_bbox_image(grid_image, bbox, "red", 3)
        
        prefix = f"""
            You are analyzing waste using a coordinate grid system.
            The grid shows normalized coordinates (0.0 to 1.0).
            Target object is in RED BOX at coordinates: {bbox}
            Use the grid to verify you're looking at the correct location.
        """

        prompt = self._create_prompt(bbox, prefix)
        
        return self._generate_response(bbox_grid_image, prompt)
    
    def _multi_stage_analysis(self, image: Image.Image, bbox: list) -> Dict[str, Any]:
        """Combine multiple spatial grounding approaches"""
        
        # Stage 1: Visual bbox analysis
        visual_result = self._analyze_with_visual_bbox(image, bbox)
        
        # Stage 2: Cropped analysis for detail
        cropped_result = self._analyze_cropped_region(image, bbox)
        
        # Stage 3: Confidence-weighted combination
        visual_conf = visual_result.get("object", {}).get("confidence", 0.0)
        cropped_conf = cropped_result.get("object", {}).get("confidence", 0.0)
        
        # Use result with higher confidence
        primary_result = visual_result if visual_conf >= cropped_conf else cropped_result
        secondary_result = cropped_result if visual_conf >= cropped_conf else visual_result
        
        # Combine insights
        combined_result = primary_result.copy()
        combined_result["multi_stage_analysis"] = {
            "primary_method": "visual_bbox" if visual_conf >= cropped_conf else "cropped",
            "visual_confidence": visual_conf,
            "cropped_confidence": cropped_conf,
            "agreement_check": self._check_result_agreement(visual_result, cropped_result)
        }
        
        return combined_result

    def load_image(self, image_path: str) -> Image.Image:
        """Load and validate image"""
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            image = Image.open(image_path).convert("RGB")
            logger.info(f"Loaded image: {image_path.name} ({image.size})")
            return image
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise


    def _generate_response(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        """Generate model response"""
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }]
            

            print(prompt)
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            response = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[-1]:], 
                skip_special_tokens=True
            ).strip()
            
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return {"raw_response": response, "parsing_error": True}
                
        except Exception as e:
            return {"error": str(e)}

    def analyze_waste(self, image: Image.Image, bbox: list, max_tokens: int = 150) -> Dict[str, Any]:
        """
        Analyze waste object in the given bounding box
        
        Args:
            image: PIL Image object
            bbox: Bounding box coordinates [x_min, y_min, x_max, y_max] (normalized)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Create prompt
            prompt = self._create_prompt(bbox)
            
            # Prepare messages
            messages = [
                {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            # Process inputs
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response with optimized settings
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[-1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Try to parse JSON response
            try:
                result = json.loads(response)
                logger.info("Successfully parsed JSON response")
                return result
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON, returning raw response")
                return {"raw_response": response, "error": "Invalid JSON format"}
                
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {"error": str(e)}
    
    def batch_analyze(self, image_bbox_pairs: list, max_tokens: int = 150) -> list:
        """
        Analyze multiple waste objects efficiently
        
        Args:
            image_bbox_pairs: List of (image_path, bbox) tuples
            max_tokens: Maximum tokens per generation
            
        Returns:
            List of analysis results
        """
        results = []
        
        for i, (image_path, bbox) in enumerate(image_bbox_pairs):
            logger.info(f"Processing image {i+1}/{len(image_bbox_pairs)}: {Path(image_path).name}")
            
            try:
                image = self.load_image(image_path)
                result = self.analyze_waste(image, bbox, max_tokens)
                results.append({
                    "image_path": str(image_path),
                    "bbox": bbox,
                    "analysis": result
                })
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results.append({
                    "image_path": str(image_path),
                    "bbox": bbox,
                    "error": str(e)
                })
        
        return results

def main():
    """Example usage"""
    # Initialize analyzer
    analyzer = WasteAnalyzer()
    
    # Configuration
    # image_path = "/media/WasteAnt_gate01_top_2025_08_05_11_26_15_0b2e231d-981a-466a-b19e-29c3edf0d80a.jpg"
    # bbox = [0.5365500450134277, 0.8005996346473694, 0.7194855213165283, 1.0]
    
    # image_path = "/media/WasteAnt_gate01_top_2025_08_05_04_10_55_d60c6521-58f4-45f4-adfc-83cdbbb9956d.jpg"
    # bbox = [0.3042489290237427, 0.11954745650291443, 0.33218657970428467, 0.3184957504272461]

    # image_path = "/media/WasteAnt_gate01_top_2025_08_05_12_55_32_71470d40-fa8b-48a9-8a07-53401774eeaa.jpg"
    # bbox = [0.0006971120601519942, 0.3759176731109619, 0.11406988650560379, 0.493928462266922]

    # image_path = "/media/WasteAnt_gate01_top_2025_08_05_15_15_03_35c03688-83b1-440d-98dc-7426f58b2390.jpg"
    # bbox = [0.6246758699417114, 0.20973001420497894, 0.7435525059700012, 0.3615691065788269]

    # image_path = "/media/WasteAnt_gate01_top_2025_08_05_12_15_42_dbfde8b6-bb42-4b5a-b1b0-1769a4488366.jpg"
    # bbox = [0.02586503140628338, 0.5508846640586853, 0.11629045009613037, 0.6896491646766663]

    # image_path = "/media/WasteAnt_gate01_top_2025_08_06_08_55_15_f0ae0f64-d824-4d01-b4cd-629b81b85d49.jpg"
    # bbox = [0.33583348989486694, 0.8297451138496399, 0.4280712306499481, 0.9993402361869812]

    # image_path = "/media/WasteAnt_gate01_top_2025_08_06_06_40_29_404998f8-8daa-45f4-8fda-1ececf7130fa.jpg"
    # bbox = [0.5022584199905396, 0.4768979549407959, 0.6689766645431519, 0.7036324739456177] 

    image_path = "/media/WasteAnt_gate01_top_2025_08_06_09_45_12_f4383b0a-22eb-48be-b3d2-089e21d2f80e.jpg"
    bbox = [0.4895365536212921, 0.42953336238861084, 0.6118486523628235, 0.5532657504081726]

    # Single analysis
    try:
        image = analyzer.load_image(image_path)
        # result = analyzer.analyze_waste(image, bbox)

        result = analyzer.analyze_with_spatial_verification(image, bbox, 'visual_bbox')        
        print("Analysis Result:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
    
    # Example batch processing
    # batch_pairs = [
    #     (image_path, bbox),
    #     # Add more (image_path, bbox) pairs here
    # ]
    # batch_results = analyzer.batch_analyze(batch_pairs)

if __name__ == "__main__":
    main()