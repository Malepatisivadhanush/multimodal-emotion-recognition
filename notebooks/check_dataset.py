import sys
import os

# Add project root path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.speech_pipeline.dataset import SpeechEmotionDataset

dataset = SpeechEmotionDataset()

print("Total samples:", len(dataset))

x, y = dataset[0]

print("Feature shape:", x.shape)
print("Label:", y)