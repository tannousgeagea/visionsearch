from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image, ImageFont, ImageDraw


def create_visual_bbox_image(image: Image.Image, bbox: list, 
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

def run_llava_inference(image_path: str, prompt_text: str, bbox:list, max_new_tokens: int = 100,) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load processor and model
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)

    # Load local image
    image = Image.open(image_path).convert("RGB")
    bbox_image = create_visual_bbox_image(image, bbox)

    # Chat format (LLaVA expects text + image in a specific format)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image"}
            ],
        }
    ]
    formatted_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Tokenize inputs
    inputs = processor(images=bbox_image, text=formatted_prompt, return_tensors="pt").to(device)

    # Generate response
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)

    return processor.decode(output[0], skip_special_tokens=True).strip()


# 🔧 Example usage
if __name__ == "__main__":
    image_path = "/media/WasteAnt_gate01_top_2025_08_05_11_26_15_0b2e231d-981a-466a-b19e-29c3edf0d80a.jpg"  # 👈 replace with your image path
    bbox = [0.5365500450134277, 0.8005996346473694, 0.7194855213165283, 1.0]

    prefix = f"""
    You are analyzing waste in a bunker. The RED BOUNDING BOX shows the exact object to analyze.
    Focus ONLY on the object inside the red box. Ignore everything outside."""


    prompt = f"""
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

    response = run_llava_inference(image_path, prompt, bbox)
    print("LLaVA response:", response)
