import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("/home/appuser/src/clip_finetuned.pt", device=device)

image = preprocess(Image.open("/home/appuser/src/archive/000000005037.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a bus"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]