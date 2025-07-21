import argparse
import sys
import time
import logging
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Moondream2VLM:
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.client = None
        self.img = None
        self.model = None
        self.tokenizer = None
        self.categories = []

    def _initialize_client(self):
        """Initialize the Moondream2 VL client and model"""
        try:
            self._load_model_and_tokenizer()
        except Exception as e:
            self.logger.error(f"❌ Error initializing VL model: {e}")
            sys.exit(1)

    def _load_model_and_tokenizer(self):
        """Load the Moondream2 model and tokenizer"""
        model_name = "vikhyatk/moondream2"
        revision = "2025-06-21"
        device = "cuda"  # Or 'mps' for Apple Silicon
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, trust_remote_code=True, device_map={"": device}
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)

    def _load_image(self, image_path: str):
        """Load image from disk"""
        try:
            self.img = Image.open(image_path).convert("RGB")
        except Exception as e:
            self.logger.error(f"❌ Error loading image: {e}")
            sys.exit(1)

    def _detect_objects(self):
        """Detect objects in the image using the model"""
        width, height = self.img.size
        detections = {}

        def compute_location(cx, cy):
            x_norm, y_norm = cx / width, cy / height
            horiz = 'left' if x_norm < 0.33 else 'right' if x_norm > 0.66 else 'center'
            vert  = 'top'  if y_norm < 0.33 else 'bottom' if y_norm > 0.66 else 'middle'
            if vert == 'middle':
                return horiz
            if horiz == 'center':
                return vert
            return f"{vert} {horiz}"

        for cat in self.categories:
            objs = []
            # Run object detection (simplified for example)
            result = self.model.detect(self.img, cat)
            for o in result.get('objects', []):
                raw_bbox = o.get('bbox', None) or [o.get('x_min'), o.get('y_min'), o.get('x_max'), o.get('y_max')]
                loc = None
                if raw_bbox and len(raw_bbox) == 4:
                    try:
                        x1, y1, x2, y2 = map(float, raw_bbox)
                        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        loc = compute_location(cx, cy)
                    except Exception:
                        loc = None
                objs.append({
                    'bbox': raw_bbox,
                    'score': o.get('score', 0.8),  # Default confidence
                    'label': o.get('label', cat),
                    'location': loc
                })
            detections[cat] = objs
        return detections

    def _draw_detections(self, detections, out_path="output_with_boxes.jpg"):
        """Draw bounding boxes on detected objects and save the image"""
        from PIL import ImageDraw, ImageFont

        img = self.img.convert("RGB").copy()
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        width, height = img.size
        for category, objs in detections.items():
            for obj in objs:
                bbox = obj.get("bbox")
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    # scale normalized coords
                    if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:
                        x1, y1 = x1 * width, y1 * height
                        x2, y2 = x2 * width, y2 * height
                    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
                    label = f"{obj.get('label', category)}:{(obj.get('score') or 0):.2f}"

                    # measure text size
                    if font:
                        text_width, text_height = font.getsize(label)
                    else:
                        bbox_txt = draw.textbbox((0, 0), label)
                        text_width = bbox_txt[2] - bbox_txt[0]
                        text_height = bbox_txt[3] - bbox_txt[1]
                    text_bg = [x1, y1 - text_height, x1 + text_width, y1]
                    draw.rectangle(text_bg, fill=(255, 0, 0))
                    draw.text((x1, y1 - text_height), label, fill=(255, 255, 255), font=font)

        img.save(out_path)
        self.logger.info(f"✅ Annotated image saved to {out_path}")

    def _generate_assessment(self, prompt: str):
        """Send prompt to the model and get the response"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs)
            caption = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            self.logger.error(f"❌ Assessment generation failed: {e}")
            sys.exit(1)

    def _generate_caption(self, prompt: str):
        """Generate a caption for the image using the client"""
        caption = self._generate_assessment(prompt)
        self.logger.info(f"📝 Generated caption:\n{caption.strip()}")

    def analyze_image(self, image_path, objects_list):
        """Process the entire image analysis and generate results"""
        self._load_image(image_path)
        self.categories = [o.strip() for o in objects_list.split(',') if o.strip()]

        # Object detection
        detections = self._detect_objects()

        # Annotate detections and generate caption
        self._draw_detections(detections)

        # Generate the caption
        self._generate_caption(f"Generate a detailed caption for this image with detected objects.")

def main():
    parser = argparse.ArgumentParser(description="Detect objects, perform JSON analysis, and generate captions.")
    parser.add_argument("--image", "-i", default="/media/Dataset/AGR_gate02_right_2025-05-28_07-06-59_04f0270b-9725-4023-b145-10a9b2e3c679.jpg", help="Path to the waste bunker image file")
    parser.add_argument("--objects", "-o", default="mattress,mattress, mattress", help="Comma-separated list of object types to detect")
    args = parser.parse_args()

    config = {}

    # Initialize Moondream2VLM client and process image
    moondream2 = Moondream2VLM(config)
    moondream2._initialize_client()
    moondream2.analyze_image(args.image, args.objects)

if __name__ == "__main__":
    main()
