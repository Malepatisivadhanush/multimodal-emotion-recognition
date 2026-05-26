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

### Dataset Used
Toronto Emotional Speech Set (TESS)

The dataset contains:
- Speech audio samples (.wav)
- Emotion labels
- Multiple emotion categories

---

## Model Architectures

### Speech Pipeline

```text
Input Audio
→ MFCC Feature Extraction
→ BiLSTM
→ Dense Layer
→ Emotion Prediction
```

### Text Pipeline

```text
Input Text
→ DistilBERT
→ Dense Layer
→ Emotion Prediction
```

### Multimodal Fusion Pipeline

```text
Speech Features → BiLSTM
                           \
                            → Fusion → Dense Layer → Emotion Prediction
                           /
Text Features → DistilBERT
```

---

## Results

| Model | Accuracy |
|-------|-----------|
| Speech-only | 99.82% |
| Text-only | 12.5% |
| Fusion | 99.82% |

---

## Analysis

- Speech-only model achieved very high accuracy because emotional cues are strongly present in speech signals.
- Text-only model performed poorly because the TESS dataset contains emotionally neutral words with limited semantic information.
- Fusion performance improved significantly after:
  - using pretrained speech encoder weights
  - fixing train/test split consistency
  - reducing data leakage risks

The final fusion model achieved near-perfect performance on the TESS dataset.

---

## Visualizations

Implemented:
- Confusion Matrix
- t-SNE Visualization
- Performance Analysis

---

## Streamlit Demo

The project includes a Streamlit-based interactive application for:
- Speech emotion prediction
- Text emotion prediction
- Multimodal fusion prediction

Run the application using:

```bash
streamlit run app.py
```

---

## Folder Structure

```text
project/

├── models/
│   ├── speech_pipeline/
│   ├── text_pipeline/
│   └── fusion_pipeline/

├── results/
├── notebooks/
├── data/
├── app.py
├── README.md
└── requirements.txt
```

---

## Technologies Used

- Python
- PyTorch
- Transformers
- DistilBERT
- Librosa
- Scikit-Learn
- Matplotlib
- Streamlit

---

## Future Improvements

- Real-time emotion detection
- Attention-based fusion
- Better multimodal datasets
- Transformer-based speech models

---

## Project Achievements

- Implemented Speech Emotion Recognition using MFCC + BiLSTM
- Implemented Text Emotion Recognition using DistilBERT
- Built Multimodal Fusion Architecture
- Created Streamlit Interactive Demo
- Generated Confusion Matrix and t-SNE Visualizations
- Fixed data leakage and train/test split issues
- Achieved near-perfect fusion performance
