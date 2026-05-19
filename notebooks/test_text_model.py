import sys
import os
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.text_pipeline.model import TextEmotionModel

model = TextEmotionModel()

input_ids = torch.randint(0, 1000, (4, 20))
attention_mask = torch.ones(4, 20)

output = model(input_ids, attention_mask)

print("Output shape:", output.shape)