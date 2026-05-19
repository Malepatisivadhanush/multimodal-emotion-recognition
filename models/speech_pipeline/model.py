import torch
import torch.nn as nn


class SpeechBiLSTM(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, num_classes=7):
        super(SpeechBiLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        output, _ = self.lstm(x)

        # Mean pooling instead of last timestep
        pooled_output = torch.mean(output, dim=1)

        return self.fc(pooled_output)