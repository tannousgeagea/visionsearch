from transformers import AutoProcessor, AutoModelForImageTextToText, Gemma3ForConditionalGeneration
from PIL import Image
import torch

# Load model and processor
processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
model = AutoModelForImageTextToText.from_pretrained(
    "google/gemma-3-4b-it", device_map="auto"
)
model.eval()

# Load a local image

src = "/media"
image_name = "WasteAnt_gate01_top_2025_08_06_06_40_29_404998f8-8daa-45f4-8fda-1ececf7130fa.jpg"
image = Image.open(f"{src}/{image_name}").convert("RGB")

# Define your custom prompt
custom_prompt = """

You are given an image containing waste material. Focus on the object located within the following bounding box, specified in normalized XYXY format relative to the image dimensions:

Bounding box:
    bbox = [0.5022584199905396, 0.4768979549407959, 0.6689766645431519, 0.7036324739456177] (x_min, y_min, x_max, y_max)

This region likely contains a relevant waste object. Your task is to analyze only this object and return detailed semantic and visual properties for it.
Describe the object with the following properties:

    "object_type": The most appropriate label from ['pipe', 'mattress', 'furniture', 'metal object', 'fabric', 'gas canister', 'bottle', 'rug', 'duvet', 'bed sheet', 'plastic bag'] — if none apply, return "other".

    "color": The dominant color or meaningful color pattern (e.g., 'blue and grey', 'rusted metal').

    "material": The inferred material type (e.g., plastic, metal, foam, wood, fabric, rubber, composite).

    "visibility": One of 'fully visible', 'partially occluded', or 'heavily occluded'.

    "size": Relative to the image — 'small', 'medium', or 'large'.

    "location": Spatial location in the image using terms like 'bottom right', 'center', etc.

    "confidence": Your certainty (float between 0.0 and 1.0) that the interpretation is correct.

Output format:

{
  "object": {
    "object_type": "<object_type>",
    "color": "<dominant_color>",
    "material": "<inferred_material>",
    "visibility": "<visibility_level>",
    "size": "<object_size>",
    "location": "<spatial_position>",
    "confidence": <confidence_score>
  }
}

Focus only on the content inside the bounding box. If you are unsure, use "object_type": "other" and reduce confidence accordingly.
"""

# Build message structure
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": custom_prompt}
        ]
    }
]

# Prepare model input
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

# Generate response
with torch.inference_mode():  # lower memory than no_grad
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

# Decode only the new tokens
response = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print(response)