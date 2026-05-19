import sys
import os
import torch
from torch.utils.data import Dataset

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from models.speech_pipeline.dataset import SpeechEmotionDataset
from models.text_pipeline.bert_dataset import EmotionTextDataset


class FusionEmotionDataset(Dataset):

    def __init__(self):
        self.speech_dataset = SpeechEmotionDataset()
        self.text_dataset = EmotionTextDataset("data/text_dataset.csv")

    def __len__(self):
        return len(self.speech_dataset)

    def __getitem__(self, index):
        speech_features, speech_label = self.speech_dataset[index]
        text_sample = self.text_dataset[index]

        return {
            "speech": torch.tensor(speech_features, dtype=torch.float32),
            "input_ids": text_sample["input_ids"],
            "attention_mask": text_sample["attention_mask"],
            "label": torch.tensor(speech_label, dtype=torch.long)
        }