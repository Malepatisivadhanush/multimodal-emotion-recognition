import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.fusion_pipeline.dataset import FusionEmotionDataset

dataset = FusionEmotionDataset()

sample = dataset[0]

print("Total samples:", len(dataset))
print("Speech shape:", sample["speech"].shape)
print("Input IDs shape:", sample["input_ids"].shape)
print("Attention mask shape:", sample["attention_mask"].shape)
print("Label:", sample["label"])