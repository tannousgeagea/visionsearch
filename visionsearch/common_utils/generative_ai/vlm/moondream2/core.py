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
        logging.basicConfig(level=logging.INFO)  # Ensure that INFO logs are shown
        self.config = config
        self.model = None
        self.tokenizer = None
        self.img = None
        self.categories = []

    def _initialize_client(self):
        """Initialize the Moondream2 VL model and tokenizer"""
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
            print(f"Image loaded successfully: {image_path}")  # Debugging output
        except Exception as e:
            self.logger.error(f"❌ Error loading image: {e}")
            sys.exit(1)


    def caption_image(self, length="normal"):
        """Generate a caption for the image using the model's caption method"""
        try:
            caption_result = self.model.caption(self.img, length=length)
            return caption_result["caption"]
        except Exception as e:
            self.logger.error(f"❌ Caption generation failed: {e}")
            sys.exit(1)

    def query_image(self, query):
        """Query the image with a question"""
        try:
            answer = self.model.query(self.img, query)["answer"]
            return answer
        except Exception as e:
            self.logger.error(f"❌ Query failed: {e}")
            sys.exit(1)

    def detect_objects(self, category):
        """Perform object detection on the image"""
        try:
            objects = self.model.detect(self.img, category)["objects"]
            return objects
        except Exception as e:
            self.logger.error(f"❌ Object detection failed: {e}")
            sys.exit(1)

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

                    # Use textbbox to calculate text size
                    if font:
                        try:
                            bbox_txt = draw.textbbox((0, 0), label, font=font)
                            text_width = bbox_txt[2] - bbox_txt[0]
                            text_height = bbox_txt[3] - bbox_txt[1]
                        except AttributeError:
                            text_width = text_height = 0
                    else:
                        bbox_txt = draw.textbbox((0, 0), label)
                        text_width = bbox_txt[2] - bbox_txt[0]
                        text_height = bbox_txt[3] - bbox_txt[1]

                    text_bg = [x1, y1 - text_height, x1 + text_width, y1]
                    draw.rectangle(text_bg, fill=(255, 0, 0))
                    draw.text((x1, y1 - text_height), label, fill=(255, 255, 255), font=font)

        img.save(out_path)
        self.logger.info(f"✅ Annotated image saved to {out_path}")

    def analyze_image(self, image_path, objects_list):
        """Process the entire image analysis and generate results"""
        self._load_image(image_path)
        self.categories = [o.strip() for o in objects_list.split(',') if o.strip()]

        # Generate caption for the image
        caption = self.caption_image(length="normal")
        self.logger.info(f"📜 Generated Caption: {caption}")

        # Query example (You can change this as needed)
        query_result = self.query_image("How many people are in the image?")
        self.logger.info(f"📋 Query Result: {query_result}")

        # Object detection for a specific category
        for category in self.categories:
            objects = self.detect_objects(category)
            self.logger.info(f"🛠 Found {len(objects)} '{category}' in the image.")

def main():
    parser = argparse.ArgumentParser(description="Analyze images, perform object detection, and generate captions.")
    parser.add_argument("--image", "-i", default="/media/Dataset/AGR_gate02_right_2025-05-28_07-06-59_04f0270b-9725-4023-b145-10a9b2e3c679.jpg", help="Path to the image file")
    parser.add_argument("--objects", "-o", default="mattress,mattress,mattress", help="Comma-separated list of object categories to detect")
    args = parser.parse_args()

    config = {}

    # Initialize Moondream2VLM client and process image
    moondream2 = Moondream2VLM(config)
    moondream2._initialize_client()
    moondream2.analyze_image(args.image, args.objects)

    # Call the method on the Moondream2VLM instance and print the output
    caption = moondream2.caption_image(length="normal")
    moondream2.logger.info(f"📜 Generated Caption: {caption}")

    print(f"Generated Caption: {caption}")  # Also printing to the console directly

if __name__ == "__main__":
    main()

