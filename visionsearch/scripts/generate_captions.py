import os
import re
import json
from glob import glob

from common_utils.generative_ai.vlm.deepseek_vlm.core import DeepseekVLM
from common_utils.generative_ai.vlm.base import VLMConfig, AnalysisType
from common_utils.indexing.types import ImageData


DATASET_PATH = "/home/appuser/src/TiledDataset/WasteRigidDetection_tiled.v13.yolo/valid/images"
OUTPUT_DIR = "/home/appuser/src/captions"
CAPTION_FILE = "captions4.json"
IMAGE_FILTER_PREFIXES = ("AGR_gate01", "AGR_gate02")
PROMPT = """Analyze the scene and identify objects within it. Use the following categories to classify the objects:

- Plastics
- Cardboard
- Metal
- Cables / Cable Piles
- Wood
- Pipes
- Electronic Waste
- Mattresses
- Rigid Objects

For each category that contains identifiable objects, provide a detailed description of the objects, including:
- Appearance
- Condition
- Material
- Variations

Instructions:
- If you are not confident about the presence of objects in a specific category, omit that category entirely and focus on the remaining ones.
- Do not describe the scene, location, or environment.
- Ignore any sluices or barriers that restrict access.
- If a crane is visible, mention and describe it explicitly.
- If an object appears to be a fan, assume it is part of a crane and describe it accordingly.
- The response must not exceed 250 characters.
- Focus only on harmful objects that are clearly visible with a clear overview of the image content.
- Be too specific.
"""



def load_existing_captions(path: str) -> dict:
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_captions(captions: dict, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(captions, f, ensure_ascii=False, indent=4)


def get_filtered_images(dataset_path: str, prefixes: tuple) -> list:
    image_paths = sorted(glob(os.path.join(dataset_path, "*.jpg")))
    return [
        ImageData(id=i, file_path=path)
        for i, path in enumerate(image_paths)
        if os.path.basename(path).startswith(prefixes)
    ]

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    caption_path = os.path.join(OUTPUT_DIR, CAPTION_FILE)
    img_captions = load_existing_captions(caption_path)

    images = get_filtered_images(DATASET_PATH, IMAGE_FILTER_PREFIXES)

    vlm = DeepseekVLM(
        config=VLMConfig(
            api_key=" ",  # Set your API key here if needed
            model_name="deepseek-vl-7b-chat"
        )
    )

    try:
        for i, img_data in enumerate(images):
            if img_data.file_path in img_captions:
                continue

            print(f"[{i}] Generating caption for: {os.path.basename(img_data.file_path)}")
            with open(img_data.file_path, 'rb') as f:
                image_bytes = f.read()

            response = vlm.analyze_image(
                image_data=image_bytes,
                analysis_type=AnalysisType.SCENE_UNDERSTANDING,
                prompt=PROMPT
            )

            caption_sentences = re.split(r'(?<=[.!?])\s+', response.response_text.strip())
            img_captions[img_data.file_path] = caption_sentences

            save_captions(img_captions, caption_path)

    except KeyboardInterrupt:
        print("\nInterrupted. Saving progress...")
        save_captions(img_captions, caption_path)
