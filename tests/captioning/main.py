import json

with open("/home/appuser/src/captions/caption_.json", "r") as f:
    objects = json.load(f)

import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM

ckpt = "google/gemma-3-4b-it"
model = Gemma3ForCausalLM.from_pretrained(
    ckpt, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(ckpt)

imgs = [
        "/home/appuser/src/archive/AGR_gate02_right_2025-05-28_08-52-53_1879cda1-72d9-49d1-af9a-c85c0c1419c4.jpg",
        "/home/appuser/src/archive/AGR_gate01_left_2025-05-27_08-20-08_00a3ab1a-4328-4a30-99c6-a53ce76752d4.jpg",
        "/home/appuser/src/archive/AGR_gate01_left_2025-05-27_11-35-11_782ab537-adf5-469f-9d59-996aedd30878.jpg",
        "/home/appuser/src/archive/AGR_gate01_left_2025-05-06_08-52-16_fedf3335-f615-4fc8-a35e-ab804941075f 3.jpg",
        "/home/appuser/src/archive/AGR_gate01_left_2025-05-28_05-42-16_d66f07c8-05e8-4507-8958-493496efccf6.jpg"
        ]
    

image_data = []

for path2img in imgs:
    with open(path2img, "rb") as f:
        image_data.append(f.read())

captions = []

for x, im in enumerate(objects.keys()):
    # filtered_objects = [obj for obj in objects if obj.get("object").get("image_number") == x + 1]
    messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are an AI assistant specialized in generating natural-language captions for waste bunker images, "
        "using structured JSON-like outputs that describe objects and their attributes."
    }]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": f"Return informative captions with focus on the overall scene using the JSON: {objects[im]}. "
             "The waste bunker image contains those objects inside the json. "
             "Do not return or repeat the input JSON in your response—only provide natural-language captions."},
            {"type": "image", "image": image_data[x]}
        ]
    }
    ]

    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    generation = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
    generation = generation[0][input_len:]

    decoded = tokenizer.decode(generation, skip_special_tokens=True)
    captions.append(decoded)

print(captions)