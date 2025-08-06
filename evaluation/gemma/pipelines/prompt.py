def get_object_system_prompt(objects):
    system_prompt = "AI assistant specialized in analyzing images from waste bunkers," \
        f" focusing exclusively on detecting and describing {objects} of various sizes, colors, and materials." \
        f" Provide detailed and accurate information about the {objects} visible in the images as JSON"
    return system_prompt

def get_object_user_prompt(objects, mode):
    placeholder = ""
    if mode == "single":
        placeholder = "a single image"
    
    if mode == "batch":
        placeholder = "multiple images"
    prompt = f"""
        You are analyzing {placeholder}

        Instructions:

        1. Ignore the top section of the full image **only if a truck is visibly present**, as it may occlude important objects. However, do **not ignore waste being dumped** — especially if it includes pipes or other relevant categories.

        2. Object Category Focus:
        - Detect and annotate only objects belonging to the following target categories:
            {objects}
        - The appearance of these objects may vary significantly (e.g., in color, size, material, or level of fragmentation). Even small, partial, or degraded instances should be included if confidently identified.
        Describe the bunker image and use your findings of objects and its properties for the description

        3. Notes:
        - Bounding box values **must be normalized** (relative to the full image width and height).
        - Only include objects that clearly match the defined categories and are detected with sufficient confidence.
        - Use consistent language for visual and spatial descriptors to support semantic embedding-based retrieval.
        - If no valid object is detected in a region, omit that region from the JSON output entirely.
        """
    
    return prompt

def get_caption_user_prompt(json: dict):
    caption_user_prompt = f"Return informative captions with focus on the overall scene using the JSON: {json}. " \
                "The waste bunker image contains those objects inside the json. " \
                "Do not return or repeat the input JSON in your response—only provide natural-language captions."
    return caption_user_prompt

def get_caption_user_prompt_for_batches(img_keys: list, json_list: list):
    """
    Generates a prompt for captioning multiple images based on associated JSON object data.

    Args:
        img_keys (list): List of image identifiers (e.g., filenames).
        json_list (list): List of corresponding JSON dictionaries describing image contents.

    Returns:
        str: A natural-language prompt that guides caption generation for each image.
    """
    assert len(img_keys) == len(json_list), "img_keys and json_list must be the same length."

    prompt_lines = [
        "Return informative, natural-language captions focusing on the overall scene context for each image.",
        "Each image is described by a JSON structure containing its detected objects.",
        "Do not repeat or return the JSON directly—only generate captions.",
        ""
    ]

    for img_key, json_data in zip(img_keys, json_list):
        prompt_lines.append(f"Image: {img_key}")
        prompt_lines.append(f"JSON: {json_data}")
        prompt_lines.append("")

    return "\n".join(prompt_lines)



caption_system_prompt = "You are an AI assistant specialized in generating natural-language captions for waste bunker images, " \
        "using structured JSON-like outputs that describe objects and their attributes."