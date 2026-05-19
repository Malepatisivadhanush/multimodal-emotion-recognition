import sys
import os
import librosa
import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from models.speech_pipeline.model import SpeechBiLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_map = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "surprise",
    6: "sad"
}

model = SpeechBiLSTM().to(device)

model.load_state_dict(
    torch.load(
        "models/speech_pipeline/speech_model.pth",
        map_location=device
    )
)

model.eval()


audio_path = input("Enter audio file path: ")

audio, sr = librosa.load(audio_path, sr=16000)

audio, _ = librosa.effects.trim(audio)

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=40
)

mfcc = mfcc.T

max_len = 130

if mfcc.shape[0] < max_len:
    pad = max_len - mfcc.shape[0]

    mfcc = np.pad(
        mfcc,
        ((0, pad), (0, 0)),
        mode='constant'
    )

else:
    mfcc = mfcc[:max_len]

mfcc = torch.tensor(
    mfcc,
    dtype=torch.float32
).unsqueeze(0)

mfcc = mfcc.to(device)

with torch.no_grad():

    output = model(mfcc)

    prediction = torch.argmax(
        output,
        dim=1
    ).item()

print("\nPredicted Emotion:", label_map[prediction])