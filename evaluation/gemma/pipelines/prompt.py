def get_object_system_prompt(objects):
    system_prompt = "AI assistant specialized in analyzing images from waste bunkers," \
        f" focusing exclusively on detecting and describing {objects} of various sizes, colors, and materials." \
        f" Provide detailed and accurate information about the {objects} visible in the images as JSON"
    return system_prompt

def get_object_user_prompt(objects):
    prompt = f"""
        You are analyzing a single image along with several cropped regions (grids) from the same image. These regions may contain parts or whole instances of waste-related objects.

        Instructions:

        1. Ignore the top section of the full image **only if a truck is visibly present**, as it may occlude important objects. However, do **not ignore waste being dumped** — especially if it includes pipes or other relevant categories.

        2. Object Category Focus:
        - Detect and annotate only objects belonging to the following target categories:
            {objects}
        - The appearance of these objects may vary significantly (e.g., in color, size, material, or level of fragmentation). Even small, partial, or degraded instances should be included if confidently identified.
        Describe the bunker image and use your findings of objects and its properties for the description
        3. Region Analysis:
        - Each region (grid) must be treated as part of the larger image context.
        - Take into account both local and global visual cues when identifying objects.
        - Objects may be fully or partially captured across different regions.

        4. Notes:
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

caption_system_prompt = "You are an AI assistant specialized in generating natural-language captions for waste bunker images, " \
        "using structured JSON-like outputs that describe objects and their attributes."