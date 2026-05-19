import sys
import os

project_root = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, project_root)

from models.text_pipeline.bert_dataset import EmotionTextDataset

dataset = EmotionTextDataset(
    "data/text_dataset.csv"
)

sample = dataset[0]

print("Input IDs shape:",
      sample["input_ids"].shape)

print("Attention shape:",
      sample["attention_mask"].shape)

print("Label:",
      sample["label"])