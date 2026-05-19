import os
import librosa
import numpy as np
from torch.utils.data import Dataset


class SpeechEmotionDataset(Dataset):
    def __init__(self, dataset_path="data/TESS", max_len=130):
        self.dataset_path = dataset_path
        self.max_len = max_len
        self.samples = []
        self.label_map = {
            "angry": 0,
            "disgust": 1,
            "fear": 2,
            "happy": 3,
            "neutral": 4,
            "pleasant_surprise": 5,
            "pleasant_surprised": 5,
            "surprise": 5,
            "surprised": 5,
            "sad": 6
        }

        for folder in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, folder)

            if os.path.isdir(folder_path):
                emotion = "_".join(folder.split("_")[1:]).lower()

                for file in os.listdir(folder_path):
                    if file.endswith(".wav"):
                        self.samples.append(
                            (os.path.join(folder_path, file), self.label_map[emotion])
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        audio_path, label = self.samples[index]

        audio, sr = librosa.load(audio_path, sr=16000)
        audio, _ = librosa.effects.trim(audio)

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc = mfcc.T

        if mfcc.shape[0] < self.max_len:
            pad_width = self.max_len - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode="constant")
        else:
            mfcc = mfcc[:self.max_len, :]

        return mfcc.astype(np.float32), label