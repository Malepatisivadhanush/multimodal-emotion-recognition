import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from models.speech_pipeline.dataset import SpeechEmotionDataset
from models.speech_pipeline.model import SpeechBiLSTM


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# Dataset
dataset = SpeechEmotionDataset()

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_data, test_data = random_split(
    dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=32
)


# Model
model = SpeechBiLSTM().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


epochs = 10


for epoch in range(epochs):

    model.train()

    total_loss = 0

    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(x)

        loss = criterion(outputs, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{epochs}, Loss:{total_loss:.4f}"
    )


torch.save(
    model.state_dict(),
    "models/speech_pipeline/speech_model.pth"
)

print("Model saved")