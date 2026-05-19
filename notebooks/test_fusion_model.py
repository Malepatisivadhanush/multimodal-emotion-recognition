import sys
import os
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.fusion_pipeline.model import FusionEmotionModel

model = FusionEmotionModel()

speech = torch.randn(4, 130, 40)
input_ids = torch.randint(0, 1000, (4, 20))
attention_mask = torch.ones(4, 20)

output = model(speech, input_ids, attention_mask)

print("Output shape:", output.shape)