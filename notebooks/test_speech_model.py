import sys
import os
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.speech_pipeline.model import SpeechBiLSTM

model = SpeechBiLSTM()

dummy_input = torch.randn(8, 130, 40)

output = model(dummy_input)

print("Input shape:", dummy_input.shape)
print("Output shape:", output.shape)