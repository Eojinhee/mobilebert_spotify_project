from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
import torch

model = MobileBertForSequenceClassification.from_pretrained("google/mobilebert-uncased", num_labels=2)
tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
model.eval()

samples = [
    "This app is amazing! Great music quality.",
    "Terrible experience. The app keeps crashing.",
    "I love how easy it is to find new songs.",
    "Cannot play anything. Worst update ever."
]

for sentence in samples:
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        label = "긍정" if pred == 1 else "부정"
        print(f"[{label}] {sentence}")
