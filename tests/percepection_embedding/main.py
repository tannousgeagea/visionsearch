import common_utils.perception_models.core.vision_encoder.pe as pe 
import common_utils.perception_models.core.vision_encoder.transforms as transforms
from PIL import Image
import torch 


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'PE-Core-B16-224' 
    model = pe.CLIP.from_config(model_name, pretrained=True)  # Downloads from HF
    model = model.to(device)

    preprocess = transforms.get_image_transform(model.image_size)
    tokenizer = transforms.get_text_tokenizer(model.context_length)

    image = preprocess(Image.open("/home/appuser/src/archive/AGR_gate02_right_2025-05-27_06-23-12_019d20c6-33af-4e2b-98c2-8654eda5dce9.jpg")).unsqueeze(0).to(device)
    captions = ["cables", "pipe", "gas canister", "bomb", "nuke", "hazard"]

    text = tokenizer(captions).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]

    print("Captions:", captions)
    print("Label probs:", ' '.join(['{:.2f}'.format(prob) for prob in text_probs]))  # prints: [[0.00, 0.00, 1.00]]
    print(f"This image is about {captions[text_probs.argmax()]}")
