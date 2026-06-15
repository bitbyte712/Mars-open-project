import os
import json
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import streamlit as st
from transformers import Wav2Vec2Model

# Config constraints
TARGET_SR = 16000
MAX_SAMPLES = 64000
MODEL_NAME = "facebook/wav2vec2-base"

# Page styling
st.set_page_config(
    page_title="Deepfake Audio Detector",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Styling (Glassmorphism & Harmonious Gradients)
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff7e5f, #feb47b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .status-real {
        background-color: rgba(46, 204, 113, 0.15);
        color: #2ecc71;
        border-color: #2ecc71;
    }
    .status-fake {
        background-color: rgba(231, 76, 60, 0.15);
        color: #e74c3c;
        border-color: #e74c3c;
    }
    .meta-text {
        font-size: 0.9rem;
        color: #888888;
    }
</style>
""", unsafe_allow_html=True)

# MLP Classifier Architecture matching training exactly
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim1=256, hidden_dim2=64, num_classes=2, dropout_prob=0.08):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),            # Normalizes input embeddings
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.BatchNorm1d(hidden_dim1),        # Stabilizes hidden features
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.BatchNorm1d(hidden_dim2),        # Stabilizes hidden features
            nn.Linear(hidden_dim2, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

@st.cache_resource
def load_feature_extractor():
    """Loads and caches the Wav2Vec2 model for embedding extraction."""
    wav2vec_model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    wav2vec_model.eval()
    for param in wav2vec_model.parameters():
        param.requires_grad = False
    return wav2vec_model

@st.cache_resource
def load_classifier(model_path):
    """Loads and caches the MLP classifier weights."""
    model = MLPClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_audio(file_bytes):
    """Loads audio from memory bytes and preprocesses to 16kHz, 4 seconds, normalized."""
    # Read bytes using soundfile
    y, sr = sf.read(file_bytes, dtype='float32')
    
    # Stereo -> Mono
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
        
    # Resample to 16kHz
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        
    # Pad / Truncate to 64000 samples (4 seconds)
    if len(y) > MAX_SAMPLES:
        y = y[:MAX_SAMPLES]
    elif len(y) < MAX_SAMPLES:
        y = np.pad(y, (0, MAX_SAMPLES - len(y)), mode='constant')
        
    # Peak Normalization
    max_val = np.max(np.abs(y))
    if max_val > 1e-8:
        y = y / max_val
        
    return y.astype(np.float32)

# Page Layout
st.markdown("<div class='main-header'>🎙️ Binary Deepfake Audio Detector</div>", unsafe_allow_html=True)
st.write("Upload an audio file below to check if the recording is **Genuine (Human)** or **Deepfake (AI-Generated)**.")

# Verify files exist in workspace
model_path = "best_classifier.pth"
config_path = "classifier_config.json"

if not os.path.exists(model_path) or not os.path.exists(config_path):
    st.error("⚠️ Inference files missing!")
    st.info("Please copy **best_classifier.pth** and **classifier_config.json** into the `submission` directory to run this web app.")
else:
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)
    threshold = config.get("threshold", 0.5)

    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3", "flac", "ogg", "m4a"])

    if uploaded_file is not None:
        # Audio Player
        st.subheader("Play Audio Clip")
        st.audio(uploaded_file)

        # Processing section
        with st.spinner("Processing audio and extracting Wav2Vec2 features..."):
            try:
                # Preprocess
                waveform = preprocess_audio(uploaded_file)
                
                # Extract embeddings
                feature_extractor = load_feature_extractor()
                waveform_tensor = torch.tensor(waveform).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = feature_extractor(waveform_tensor)
                    embedding = outputs.last_hidden_state.mean(dim=1)
                
                # Load classifier
                classifier = load_classifier(model_path)
                
                # Predict
                with torch.no_grad():
                    logits = classifier(embedding)
                    prob_fake = torch.softmax(logits, dim=1)[0, 1].item()
                    
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.stop()

        # Display Results
        is_fake = prob_fake >= threshold
        label = "DEEPFAKE (AI-Generated)" if is_fake else "GENUINE (Human)"
        confidence = prob_fake if is_fake else (1.0 - prob_fake)
        class_css = "status-fake" if is_fake else "status-real"

        st.subheader("Analysis Output")
        st.markdown(f"<div class='status-card {class_css}'>{label}</div>", unsafe_allow_html=True)

        # Confidence Bar
        st.write(f"**Detector Confidence:** {confidence:.2%}")
        st.progress(confidence)

        # Waveform Plotting
        st.subheader("Audio Waveform")
        fig, ax = plt.subplots(figsize=(10, 3))
        time_axis = np.linspace(0, len(waveform) / TARGET_SR, num=len(waveform))
        ax.plot(time_axis, waveform, color="#ff7e5f" if is_fake else "#2ecc71", alpha=0.8)
        ax.set_title("Preprocessed 16kHz Waveform (4s Clip)", fontsize=10, color="white")
        ax.set_xlabel("Time (seconds)", fontsize=8, color="gray")
        ax.set_ylabel("Amplitude", fontsize=8, color="gray")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        ax.spines['bottom'].set_color('gray')
        ax.spines['top'].set_color('gray')
        ax.spines['left'].set_color('gray')
        ax.spines['right'].set_color('gray')
        ax.tick_params(colors='gray', labelsize=8)
        st.pyplot(fig)

        # Model Metadata
        st.markdown("---")
        st.markdown("<div class='meta-text'>**Classifier Specifications:**</div>", unsafe_allow_html=True)
        st.json({
            "Backbone Feature Extractor": config.get("model_name"),
            "Embedding Vector Size": f"{config.get('embedding_size')} dimensions",
            "Classification Threshold": f"{threshold:.6f}",
            "Target Sample Rate": f"{config.get('sample_rate')} Hz"
        })
