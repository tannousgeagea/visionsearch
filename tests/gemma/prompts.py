obj = "pipes"
CONTENT_PIPES = [
                {"type": "image", "image": "/home/appuser/src/archive/AGR_gate02_right_2025-05-28_08-52-53_1879cda1-72d9-49d1-af9a-c85c0c1419c4.jpg"},
                {"type": "image", "image": "/home/appuser/src/archive/AGR_gate01_left_2025-05-27_08-20-08_00a3ab1a-4328-4a30-99c6-a53ce76752d4.jpg"},
                # {"type": "image", "image": "/home/appuser/src/archive/AGR_gate01_left_2025-05-27_11-35-11_782ab537-adf5-469f-9d59-996aedd30878.jpg"},
                # {"type": "image", "image": "/home/appuser/src/archive/AGR_gate01_left_2025-05-06_08-52-16_fedf3335-f615-4fc8-a35e-ab804941075f 3.jpg"},
                # {"type": "image", "image": "/home/appuser/src/archive/AGR_gate01_left_2025-05-28_05-42-16_d66f07c8-05e8-4507-8958-493496efccf6.jpg"},
            ]

PROMPT_WITH_REFERENCE = (
    "The first image contains at least four pipes: "
    "One is in the bottom left, metallic, silver in color, and wrapped in plastic. "
    "The second is located toward the top left and is gray, possibly made of PVC. "
    "The third is directly below the second one and is beige in color. "
    "The fourth is in the center or center-right area and is also gray. "
    "Analyze the remaining images and determine whether any additional pipes are present. "
    "Ignore the top section of the image only if a truck is visibly present there, as it may occlude objects. "
    "However, do not ignore the waste being dumped, as it may contain pipes or other relevant objects. "
    "The color and size of pipes in the other images may differ significantly — they can be much smaller — "
    "but they must still belong to the same object category (pipe). "
    f"Identify all valid {obj}, including smaller ones if they match the category. "
    f"For each detected pipe, provide the following information in JSON format: "
    "{{'object': {{"
    f"'object_type': '{obj}', "
    "'color': <object_color>, "
    "'material': <object_material> (e.g., metal, plastic, concrete), "
    "'location': <spatial_location>, "
    "'visibility': <visibility_level>, "
    "'size': <object_size>, "
    "'confidence': <confidence_score>, "
    "'image_number': <image_number>"
    "}}}}. "
    "Use spatial terms like 'top left', 'top', 'top right', 'center', 'bottom left', etc. for location. "
    "Describe visibility as 'fully visible', 'partially occluded', or 'heavily occluded'. "
    "Size should be labeled as 'small', 'medium', or 'large'. "
    "Confidence should be a float between 0.0 and 1.0 representing the model's certainty. "
    f"If you detect no matching {obj}, or are uncertain, do not include anything in the JSON output."
)
