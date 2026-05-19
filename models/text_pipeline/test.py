import sys
import os
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, classification_report

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from models.text_pipeline.bert_dataset import EmotionTextDataset
from models.text_pipeline.model import TextEmotionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = EmotionTextDataset("data/text_dataset.csv")

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

_, test_data = random_split(
    dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

test_loader = DataLoader(test_data, batch_size=8)

model = TextEmotionModel().to(device)
model.load_state_dict(torch.load("models/text_pipeline/text_model.pth", map_location=device))
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"]

        outputs = model(input_ids, attention_mask)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        y_pred.extend(preds)
        y_true.extend(labels.numpy())

print("Accuracy:", accuracy_score(y_true, y_pred))

print(classification_report(
    y_true,
    y_pred,
    target_names=["angry", "disgust", "fear", "happy", "neutral", "surprise", "sad"]
))