import sys
import os
import random

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, project_root)

from models.fusion_pipeline.dataset import FusionEmotionDataset

dataset = FusionEmotionDataset()

indices = list(range(len(dataset)))

random.seed(42)
random.shuffle(indices)

split = int(0.8 * len(indices))

train_indices = indices[:split]
test_indices = indices[split:]

os.makedirs("data/splits", exist_ok=True)

with open("data/splits/train.txt", "w") as f:
    for idx in train_indices:
        f.write(f"{idx}\n")

with open("data/splits/test.txt", "w") as f:
    for idx in test_indices:
        f.write(f"{idx}\n")

print("Train/Test split created")