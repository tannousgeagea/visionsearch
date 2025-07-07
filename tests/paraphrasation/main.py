import torch
from transformers import BartForConditionalGeneration, BartTokenizer

input_sentence = "There is a truck unloading waste and in the waste there is a pipe."

model = BartForConditionalGeneration.from_pretrained(
    'eugenesiow/bart-paraphrase',
    trust_remote_code=True,
    use_safetensors=True
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
tokenizer = BartTokenizer.from_pretrained('eugenesiow/bart-paraphrase')
batch = tokenizer(input_sentence, return_tensors='pt').to(device)
generated_ids = model.generate(batch['input_ids'])
generated_sentence = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_sentence)
