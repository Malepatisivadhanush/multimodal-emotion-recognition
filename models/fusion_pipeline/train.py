import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from models.fusion_pipeline.dataset import FusionEmotionDataset
from models.fusion_pipeline.model import FusionEmotionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

dataset = FusionEmotionDataset()

with open("data/splits/train.txt") as f:
    train_indices = [int(x.strip()) for x in f.readlines()]

train_data = torch.utils.data.Subset(
    dataset,
    train_indices
)

train_loader = DataLoader(
    train_data,
    batch_size=8,
    shuffle=True
)

model = FusionEmotionModel().to(device)
speech_weights = torch.load(
    "models/speech_pipeline/speech_model.pth",
    map_location=device
)

fusion_weights = model.state_dict()

for key in speech_weights:
    if key.startswith("lstm"):
        new_key = key.replace("lstm", "speech_lstm")
        if new_key in fusion_weights:
            fusion_weights[new_key] = speech_weights[key]

model.load_state_dict(fusion_weights)

print("Loaded pretrained speech encoder into fusion model")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        speech = batch["speech"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(speech, input_ids, attention_mask)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

torch.save(
    model.state_dict(),
    "models/fusion_pipeline/fusion_model.pth"
)

print("Fusion model saved")