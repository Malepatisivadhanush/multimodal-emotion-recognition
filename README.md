# Multimodal Emotion Recognition using Speech and Text

## Project Overview

This project aims to recognize human emotions using:

1. Speech-only input
2. Text-only input
3. Multimodal input (Speech + Text)

The system extracts emotional information from speech signals and textual data, then compares the performance of different approaches.

---

## Objective

Build an emotion recognition system capable of predicting:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Surprise
- Sad

---

## Dataset

Dataset Used:

Toronto Emotional Speech Set (TESS)

Dataset contains:

- Speech samples (.wav)
- Emotion labels
- Multiple emotion classes

---

## Model Architectures

### Speech Pipeline

Input Audio
→ MFCC Feature Extraction
→ BiLSTM
→ Dense Layer
→ Emotion Prediction

---

### Text Pipeline

Input Text
→ DistilBERT
→ Dense Layer
→ Emotion Prediction

---

### Multimodal Fusion Pipeline

Speech Features → BiLSTM
                           \
                            → Fusion → Dense Layer → Emotion Prediction
                           /
Text Features → DistilBERT

---

## Results

| Model | Accuracy |
|---------|----------|
| Speech-only | 99.82% |
| Text-only | 12.5% |
| Fusion | 36.6% |

---

## Visualizations

Implemented:

- Confusion Matrix
- t-SNE Visualization
- Performance Analysis

---

## Folder Structure

project/

├── models/

│ ├── speech_pipeline/

│ ├── text_pipeline/

│ └── fusion_pipeline/

├── Results/

├── notebooks/

├── data/

├── README.md

└── requirements.txt

---

## Technologies Used

- Python
- PyTorch
- Transformers
- DistilBERT
- Librosa
- Scikit-Learn
- Matplotlib

---

## Future Improvements

- Real-time emotion detection
- Better fusion strategies
- Attention mechanisms
- Streamlit deployment
