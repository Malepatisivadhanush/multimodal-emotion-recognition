import sys
import os
import torch
from transformers import DistilBertTokenizer

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from models.text_pipeline.model import TextEmotionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_map = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "surprise",
    6: "sad"
}

tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

model = TextEmotionModel().to(device)

model.load_state_dict(
    torch.load(
        "models/text_pipeline/text_model.pth",
        map_location=device
    )
)

model.eval()

text = input("Enter text: ")

encoding = tokenizer(
    text,
    padding='max_length',
    truncation=True,
    max_length=20,
    return_tensors='pt'
)

input_ids = encoding["input_ids"].to(device)
attention_mask = encoding["attention_mask"].to(device)

with torch.no_grad():
    output = model(input_ids, attention_mask)

prediction = torch.argmax(output, dim=1).item()

print("\nPredicted Emotion:", label_map[prediction])