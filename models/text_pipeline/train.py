import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from models.text_pipeline.bert_dataset import EmotionTextDataset
from models.text_pipeline.model import TextEmotionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

dataset = EmotionTextDataset("data/text_dataset.csv")

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_data, test_data = random_split(
    dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)

model = TextEmotionModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "models/text_pipeline/text_model.pth")

print("Text model saved")