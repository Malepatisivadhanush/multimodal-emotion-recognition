import pandas as pd
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer


class EmotionTextDataset(Dataset):

    def __init__(self, csv_file):

        self.df = pd.read_csv(csv_file)

        self.tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )

        self.label_map = {
            "angry": 0,
            "disgust": 1,
            "fear": 2,
            "happy": 3,
            "neutral": 4,
            "surprise": 5,
            "sad": 6
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        text = self.df.iloc[index]["text"]

        label = self.label_map[
            self.df.iloc[index]["emotion"]
        ]

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=20,
            return_tensors='pt'
        )

        return {
            "input_ids":
            encoding["input_ids"].squeeze(),

            "attention_mask":
            encoding["attention_mask"].squeeze(),

            "label": label
        }