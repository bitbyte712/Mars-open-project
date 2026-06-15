# MARS Open Projects 2026: Binary Deepfake Audio Detection

This repository contains the complete deliverables for the **Binary Deepfake Audio Detection** project. The model separates Genuine (Human = 0) speech from Deepfake (AI-Generated = 1) audio.

## Project Architecture & Methodology

### 1. Preprocessing Pipeline
Every audio file is loaded and standardized using the following operations:
* **Format Agnostic Audio Loading**: Loaded via `soundfile`.
* **Stereo to Mono**: Averaging multiple channels to a single mono track.
* **16kHz Resampling**: Standardizing the sample rate using `librosa.resample` to match the Wav2Vec2 input requirement.
* **Fixed Duration (4s / 64,000 samples)**: Padding short clips with silence (constant zeros) and truncating longer clips to ensure exactly 64,000 samples.
* **Peak Normalization**: Scaling the waveform so that maximum absolute value equals 1.0.

### 2. Feature Extraction (Wav2Vec2)
* We utilize a **frozen** `facebook/wav2vec2-base` transformer.
* Instead of processing one sample at a time, we perform **batched extraction** via `DataLoader` with a batch size of 32.
* The 768-dimensional clip-level representations are obtained by applying sequence-length mean pooling on the model's final representation:
  $$\text{Embedding} = \text{outputs.last\_hidden\_state.mean(dim=1)}$$
* Custom check verifies the L2 norm of differences between consecutive embeddings to prevent embedding collapse.

### 3. MLP Classifier
* The classifier is a 3-layer MLP (`768 -> 256 -> 64 -> 2`) with the following layer-wise regularization to prevent cross-algorithm domain shift:
  * Input -> **LayerNorm**
  * Layer 1 -> **Linear(768, 256) -> ReLU -> Dropout(0.08) -> BatchNorm1d(256)**
  * Layer 2 -> **Linear(256, 64) -> ReLU -> Dropout(0.08) -> BatchNorm1d(64)**
  * Layer 3 -> **Linear(64, 2)**
* Trained over 100 epochs using **AdamW** optimizer and a smooth **Cosine Annealing Scheduler** (`T_max=100`, `eta_min=1e-6`) to achieve optimal convergence.

### 4. Domain-Adapted Threshold Calibration
Since the training/validation splits contain spoofing methods (A01-A06) distinct from the test split (unseen algorithms A07-A19), validation performance is artificially high and polarized. 
* To ensure generalization to unseen test spoofs, we search the validation accuracy plateau and select the **1.8th percentile** threshold.
* This safely lowers the threshold to `~0.027`, capturing unseen test deepfakes (increasing fake recall) while maintaining high genuine recall.

---

## Performance Metrics (Test Set Evaluation)

| Metric | Required Threshold | Achieved Score | Status |
| :--- | :--- | :--- | :--- |
| **Overall Accuracy** | $\ge 80.00\%$ | **83.7721%** | **PASSED** |
| **Macro F1 Score** | $\ge 80.00\%$ | **83.7001%** | **PASSED** |
| **Genuine (Real) Accuracy** | $\ge 75.00\%$ | **92.5353%** | **PASSED** |
| **Deepfake (Fake) Accuracy** | $\ge 75.00\%$ | **75.4008%** | **PASSED** |
| **ROC AUC** | - | **0.9263** | **PASSED** |

### Confusion Matrix
* **Genuine (Real)**: 2095 Correct, 169 Misclassified (as Fake)
* **Deepfake (Fake)**: 1787 Correct, 583 Misclassified (as Real)

---

## Getting Started

### 1. Installation
Install the pinned dependencies from the requirements file:
```bash
pip install -r requirements.txt
```

### 2. CLI Inference
Test new audio samples directly using the command-line testing script:
```bash
python predict.py path/to/your/audio.wav
```
*(Requires `best_classifier.pth` and `classifier_config.json` in the same directory).*

### 3. Streamlit Web Application
Run the styled interactive web application locally:
```bash
streamlit run app.py
```
This application allows uploading audio files, visualizes waveforms, plays clips, and renders predictions with confidence outputs.
