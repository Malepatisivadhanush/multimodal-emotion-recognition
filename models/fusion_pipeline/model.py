import torch
import torch.nn as nn

from models.speech_pipeline.model import SpeechBiLSTM
from transformers import DistilBertModel


class FusionEmotionModel(nn.Module):

    def __init__(self, num_classes=7):
        super(FusionEmotionModel, self).__init__()

        self.speech_lstm = nn.LSTM(
            input_size=40,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )

        self.text_bert = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )

        self.fusion_classifier = nn.Sequential(
            nn.Linear(256 + 768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, speech, input_ids, attention_mask):
        speech_output, _ = self.speech_lstm(speech)
        speech_embedding = torch.mean(speech_output, dim=1)

        text_output = self.text_bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        text_embedding = text_output.last_hidden_state[:, 0, :]

        combined = torch.cat(
            (speech_embedding, text_embedding),
            dim=1
        )

        return self.fusion_classifier(combined)