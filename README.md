# Multimodal Emotion Recognition using Speech and Text

## Project Overview

This project aims to recognize human emotions using:

- Speech-only input
- Text-only input
- Multimodal input (Speech + Text)

The system extracts emotional information from speech signals and textual data, then compares the performance of different approaches.

---

## Objective

Build an emotion recognition system capable of predicting the following emotions:

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

### Dataset Contains
- Speech audio samples (`.wav`)
- Emotion labels
- Multiple emotion categories

---

## Model Architectures

### 1. Speech Pipeline

```text
Input Audio
→ MFCC Feature Extraction
→ BiLSTM
→ Dense Layer
→ Emotion Prediction
```

### 2. Text Pipeline

```text
Input Text
→ DistilBERT
→ Dense Layer
→ Emotion Prediction
```

### 3. Multimodal Fusion Pipeline

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

Implemented visual analysis techniques:

- Confusion Matrix
- t-SNE Visualization
- Performance Analysis

---

## Streamlit Demo

The project includes a Streamlit-based interactive application for:

- Speech emotion prediction
- Text emotion prediction
- Multimodal fusion prediction

### Run Streamlit App

```bash
streamlit run app.py
```

---

## Folder Structure

```text
multimodal-emotion-recognition/

├── models/
│   ├── speech_pipeline/
│   ├── text_pipeline/
│   └── fusion_pipeline/
│
├── results/
│   └── plots/
│
├── notebooks/
│
├── data/
│
├── app.py
├── multimodal_predict.py
├── requirements.txt
├── README.md
└── generate_report.py
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

## Installation and Execution Steps

### 1. Clone Repository

```bash
git clone https://github.com/Malepatisivadhanush/multimodal-emotion-recognition.git
cd multimodal-emotion-recognition
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

---

# Speech Pipeline

## Train Speech Model

```bash
python models/speech_pipeline/train.py
```

## Test Speech Model

```bash
python models/speech_pipeline/test.py
```

## Predict Emotion from Audio

```bash
python models/speech_pipeline/predict.py
```

---

# Text Pipeline

## Train Text Model

```bash
python models/text_pipeline/train.py
```

## Test Text Model

```bash
python models/text_pipeline/test.py
```

## Predict Emotion from Text

```bash
python models/text_pipeline/predict.py
```

---

# Fusion Pipeline

## Train Fusion Model

```bash
python models/fusion_pipeline/train.py
```

## Test Fusion Model

```bash
python models/fusion_pipeline/test.py
```

## Run Multimodal Prediction

```bash
python multimodal_predict.py
```

---

# Streamlit Application

## Run Interactive Demo

```bash
streamlit run app.py
```

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
