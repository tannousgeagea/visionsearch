from common_utils.generative_ai.vlm.gemma.core import GemmaVLM
from common_utils.generative_ai.vlm.base import VLMConfig, AnalysisType
import json
import os
import time

imgs = [
        # "/home/appuser/src/archive/AGR_gate02_right_2025-05-28_08-52-53_1879cda1-72d9-49d1-af9a-c85c0c1419c4.jpg",
        "/home/appuser/src/archive/AGR_gate01_left_2025-05-28_12-23-58_d50208f3-a0c8-47a8-8080-06f83ff7aadd.jpg",
        # "/home/appuser/src/archive/AGR_gate01_left_2025-05-27_08-20-08_00a3ab1a-4328-4a30-99c6-a53ce76752d4.jpg",
        # "/home/appuser/src/archive/AGR_gate01_left_2025-05-27_11-35-11_782ab537-adf5-469f-9d59-996aedd30878.jpg",
        # "/home/appuser/src/archive/AGR_gate01_left_2025-05-06_08-52-16_fedf3335-f615-4fc8-a35e-ab804941075f 3.jpg",
        # "/home/appuser/src/archive/AGR_gate01_left_2025-05-28_05-42-16_d66f07c8-05e8-4507-8958-493496efccf6.jpg"
        ]
    

image_data = []

for path2img in imgs:
    with open(path2img, "rb") as f:
        image_data.append(f.read())

config = VLMConfig(
    api_key="xddxdxddxdd",
    model_name="google/gemma-3-4b-it",
    max_tokens=5000,
    enable_json=True
)
vlm = GemmaVLM(config=config, system_prompt="")


objects = ["pipes", "mattresses", "furniture", "metal objects"]
result_json = {}
for im in imgs:
    result_json.update({os.path.basename(im): []})

# object_info = {
#     "object": {
#         "object_type": "<object_type>",
#         "color": "<object_color>",
#         "material": "<object_material>",  # e.g., metal, plastic, concrete
#         "location": "<spatial_location>",
#         "visibility": "<visibility_level>",
#         "size": "<object_size>",
#         "confidence": "<confidence_score>",
#         "image_number": "<image_number>",
#         "xyxy": "<[xmin, ymin, xmax, ymax]"
#     }
# }

object_info = {
  "object": {
    "object_type": "<'pipe' | 'mattress' | 'furniture' | 'metal object'>",
    "color": "<dominant_color_or_description>",
    "material": "<inferred_material_if_possible>",
    "location": "<'top left' | 'top' | 'top right' | 'center' | 'bottom left' | 'bottom' | 'bottom right'>",
    "visibility": "<'fully visible' | 'partially occluded' | 'heavily occluded'>",
    "size": "<'small' | 'medium' | 'large'>",
    "confidence": "<float between 0.0 and 1.0>",
    "bounding_box": {
      "x_min": "<float between 0.0 and 1.0>",
      "y_min": "<float between 0.0 and 1.0>",
      "x_max": "<float between 0.0 and 1.0>",
      "y_max":" <float between 0.0 and 1.0>"
    },
    "image_number": "<image_index (e.g., 1)>",
    "image_width": "int",
    "image_height": "int"
  }
}

start_time = time.time()
for obj in objects:
    system_prompt = "AI assistant specialized in analyzing images from waste bunkers," \
    f" focusing exclusively on detecting and describing {obj} of various sizes, colors, and materials." \
    f" Provide detailed and accurate information about the {obj} visible in the images as JSON"

    # prompt =(
    #     "Ignore the top section of the image only if a truck is visibly present there, as it may occlude objects. "
    #     "However, do not ignore the waste being dumped, as it may contain pipes or other relevant objects. "
    #     "The color and size of pipes in the other images may differ significantly — they can be much smaller — "
    #     "but they must still belong to the same object category (pipe). "
    #     f"Identify all valid {obj}, including smaller ones if they match the category. "
    #     f"For each detected {obj}, provide the following information in JSON format: "
    #     "Use spatial terms like 'top left', 'top', 'top right', 'center', 'bottom left', etc. for location. "
    #     "Describe visibility as 'fully visible', 'partially occluded', or 'heavily occluded'. "
    #     "Size should be labeled as 'small', 'medium', or 'large'. "
    #     "Confidence should be a float between 0.0 and 1.0 representing the model's certainty. "
    #     f"If you detect no matching {obj}, or are uncertain, do not include anything in the JSON output."
    # )

    prompt = f"""
        You are analyzing 1 image containing waste objects.
        Instructions:
        Ignore the top section of the image only if a truck is visibly present there, as it may occlude important objects.
        Do not ignore waste material being dumped, especially if it may include pipes or other relevant items.
        The color, size, or condition of pipes may vary significantly across images — even small or fragmented pipes must be identified if they match the category.
        Focus on detecting objects belonging to these categories:
        {obj}
        For each valid detection, provide a structured output in the following JSON format:
        Important notes:
            The bounding_box values must be normalized coordinates (relative to image width and height).
            Only include objects that clearly match one of the target categories with sufficient confidence.
            If no relevant object is detected or if uncertain, do not return anything.
            Use consistent spatial, visual, and material descriptors to support semantic image search using embedding models.
        """
    
    vlm.system_prompt = system_prompt

    response = vlm.analyze_multiple_images(
        images=image_data,
        prompt=prompt,
        obj=obj,
        kv_pair=object_info
    )
    if not response.success:
        print(response.error_message)
    else:
        response_dict = response.raw_response
        print(response_dict)
        for obj in response_dict:
            img_number = obj['object']['image_number']
            result_json.get(os.path.basename(imgs[img_number - 1])).append(obj)
       
        with open("/home/appuser/src/captions/caption4_.json", "w") as f:
            json.dump(result_json, f, indent=4)
    
print(f"{time.time() - start_time}")
