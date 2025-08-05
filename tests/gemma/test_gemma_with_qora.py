from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from PIL import Image
import torch
import gc

def load_gemma_4bit():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForImageTextToText.from_pretrained(
        "google/gemma-3-4b-it",
        # quantization_config=bnb_config,
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
    return model, processor

def run_inference_qora(image_path: str, prompt: str, model, processor):
    image = Image.open(image_path).convert("RGB")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    }]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs, 
            # input_ids=inputs["input_ids"].to(model.device),
            # attention_mask=inputs.get("attention_mask", None),
            max_new_tokens=60,
            do_sample=True,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,)

    input_len = inputs["input_ids"].shape[-1]
    return processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

if __name__ == "__main__":
    src = "/media/appuser/rt_cvision/experiments/images"
    image_name = "WasteAnt_gate01_top_2025_08_05_05_34_29_f798a1f2-f165-4e6f-a981-310be3b396e6.jpg"

    model, processor = load_gemma_4bit()
    result = run_inference_qora(f"{src}/{image_name}", "What object is shown and what is it made of?", model, processor)
    print(result)
