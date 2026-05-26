import streamlit as st
import torch
import librosa
import numpy as np
import tempfile
import sys
import os
from transformers import DistilBertTokenizer
from models.text_pipeline.model import TextEmotionModel

sys.path.insert(0, os.path.abspath("."))

from models.speech_pipeline.model import SpeechBiLSTM

st.set_page_config(
    page_title="Multimodal Emotion Recognition",
    layout="centered"
)

label_map = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Surprise",
    6: "Sad"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_speech_model():
    model = SpeechBiLSTM().to(device)
    model.load_state_dict(
        torch.load(
            "models/speech_pipeline/speech_model.pth",
            map_location=device
        )
    )
    model.eval()
    return model


def predict_speech(audio_path):
    model = load_speech_model()

    audio, sr = librosa.load(audio_path, sr=16000)
    audio, _ = librosa.effects.trim(audio)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = mfcc.T

    max_len = 130

    if mfcc.shape[0] < max_len:
        pad = max_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad), (0, 0)), mode="constant")
    else:
        mfcc = mfcc[:max_len]

    mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(mfcc)
        pred = torch.argmax(output, dim=1).item()

    return label_map[pred]

@st.cache_resource
def load_text_model():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    model = TextEmotionModel().to(device)
    model.load_state_dict(
        torch.load(
            "models/text_pipeline/text_model.pth",
            map_location=device
        )
    )
    model.eval()

    return tokenizer, model


def predict_text(text):
    tokenizer, model = load_text_model()

    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=20,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        pred = torch.argmax(output, dim=1).item()

    return label_map[pred]


st.title("Multimodal Emotion Recognition")

st.write("""
This project predicts emotions using Speech, Text, and Multimodal Fusion models.
""")

option = st.selectbox(
    "Choose Prediction Mode",
    ["Speech Emotion Recognition", "Text Emotion Recognition", "Multimodal Fusion"]
)

if option == "Speech Emotion Recognition":
    st.subheader("Speech Emotion Recognition")

    audio_file = st.file_uploader(
        "Upload a WAV audio file",
        type=["wav"]
    )

    if audio_file is not None:
        st.audio(audio_file)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            temp_audio_path = tmp.name

        if st.button("Predict Speech Emotion"):
            prediction = predict_speech(temp_audio_path)
            st.success(f"Predicted Emotion: {prediction}")

elif option == "Text Emotion Recognition":
    st.subheader("Text Emotion Recognition")
    text = st.text_area("Enter text")

    if st.button("Predict Text Emotion"):
        if text.strip() == "":
            st.warning("Please enter some text.")
        else:
            prediction = predict_text(text)
            st.success(f"Predicted Emotion: {prediction}")

else:
    st.subheader("Multimodal Fusion")
    audio_file = st.file_uploader("Upload WAV audio file", type=["wav"])
    text = st.text_area("Enter text for fusion")

    if st.button("Predict Fusion Emotion"):
        speech_prediction = "No audio"

        if audio_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_file.read())
                temp_audio_path = tmp.name

            speech_prediction = predict_speech(temp_audio_path)

        text_prediction = "No text"

        if text.strip() != "":
            text_prediction = predict_text(text)

        st.write("Speech Prediction:", speech_prediction)
        st.write("Text Prediction:", text_prediction)

        if speech_prediction != "No audio" and text_prediction != "No text":
            if speech_prediction == text_prediction:
                final_prediction = speech_prediction
            else:
                final_prediction = speech_prediction
        elif speech_prediction != "No audio":
            final_prediction = speech_prediction
        else:
            final_prediction = text_prediction

        st.success(
            f"Final Emotion Prediction: {final_prediction}"
        )