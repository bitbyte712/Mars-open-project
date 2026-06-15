#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

# Constants
TARGET_SR = 16000
MAX_SAMPLES = 64000
MODEL_NAME = "facebook/wav2vec2-base"

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

def preprocess_audio(file_path):
    """Loads and preprocesses audio to 16kHz mono, 4 seconds, normalized."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
        
    # 1. Load audio using soundfile
    y, sr = sf.read(file_path, dtype='float32')
    
    # 2. Convert stereo -> mono
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
        
    # 3. Resample to 16kHz
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        
    # 4. Pad/truncate to 64000 samples (4 seconds)
    if len(y) > MAX_SAMPLES:
        y = y[:MAX_SAMPLES]
    elif len(y) < MAX_SAMPLES:
        y = np.pad(y, (0, MAX_SAMPLES - len(y)), mode='constant')
        
    # 5. Normalize waveform (peak normalization)
    max_val = np.max(np.abs(y))
    if max_val > 1e-8:
        y = y / max_val
        
    return y.astype(np.float32)

def extract_embedding(waveform, device):
    """Extracts 768-D embedding from pretrained Wav2Vec2 model."""
    print("Loading Wav2Vec2 model for feature extraction...")
    wav2vec_model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
    wav2vec_model.eval()
    
    # Freeze parameters
    for param in wav2vec_model.parameters():
        param.requires_grad = False
        
    # Prepare waveform tensor (shape: 1, MAX_SAMPLES)
    waveform_tensor = torch.tensor(waveform).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = wav2vec_model(waveform_tensor)
        # Mean pooling along the sequence length dimension (dim=1)
        embedding = outputs.last_hidden_state.mean(dim=1)
        
    return embedding.cpu()

def main():
    parser = argparse.ArgumentParser(description="Binary Deepfake Audio Detector CLI Inference")
    parser.add_argument("audio_path", type=str, help="Path to the audio file to test")
    parser.add_argument("--model", type=str, default="best_classifier.pth", help="Path to best_classifier.pth")
    parser.add_argument("--config", type=str, default="classifier_config.json", help="Path to classifier_config.json")
    args = parser.parse_args()

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Verify model files
    if not os.path.exists(args.model) or not os.path.exists(args.config):
        print(f"ERROR: Model weights '{args.model}' or config '{args.config}' not found.")
        print("Please copy 'best_classifier.pth' and 'classifier_config.json' into the current directory.")
        sys.exit(1)

    # Load configuration metadata
    with open(args.config, "r") as f:
        config = json.load(f)
    threshold = config.get("threshold", 0.5)
    print(f"Loaded config. Model: {config.get('model_name')}, Threshold: {threshold:.6f}")

    # Preprocess
    print(f"Preprocessing audio: {args.audio_path}...")
    try:
        waveform = preprocess_audio(args.audio_path)
    except Exception as e:
        print(f"ERROR preprocessing audio: {e}")
        sys.exit(1)

    # Extract features
    try:
        embedding = extract_embedding(waveform, device)
    except Exception as e:
        print(f"ERROR during embedding extraction: {e}")
        sys.exit(1)

    # Load MLP model
    mlp_model = MLPClassifier()
    mlp_model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
    mlp_model.eval()

    # Predict
    with torch.no_grad():
        logits = mlp_model(embedding)
        prob_fake = torch.softmax(logits, dim=1)[0, 1].item()

    prediction = "DEEPFAKE (Fake)" if prob_fake >= threshold else "GENUINE (Real)"
    confidence = prob_fake if prediction == "DEEPFAKE (Fake)" else (1.0 - prob_fake)

    print("\n================ DETECTOR RESULTS ================")
    print(f"File Path   : {args.audio_path}")
    print(f"Prediction  : {prediction}")
    print(f"Confidence  : {confidence:.2%}")
    print(f"Fake Prob   : {prob_fake:.6f} (Threshold: {threshold:.6f})")
    print("==================================================")

if __name__ == "__main__":
    main()
