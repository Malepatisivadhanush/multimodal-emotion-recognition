import sys
import os
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, classification_report

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from models.speech_pipeline.dataset import SpeechEmotionDataset
from models.speech_pipeline.model import SpeechBiLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = SpeechEmotionDataset()

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

_, test_data = random_split(dataset, [train_size, test_size])

test_loader = DataLoader(test_data, batch_size=32)

model = SpeechBiLSTM().to(device)
model.load_state_dict(torch.load("models/speech_pipeline/speech_model.pth", map_location=device))
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)

        outputs = model(x)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        y_pred.extend(preds)
        y_true.extend(y.numpy())

print("Accuracy:", accuracy_score(y_true, y_pred))

print(classification_report(
    y_true,
    y_pred,
    target_names=["angry", "disgust", "fear", "happy", "neutral", "surprise", "sad"]
))