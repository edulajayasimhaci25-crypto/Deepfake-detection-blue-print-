import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import librosa
import numpy as np
from PIL import Image
import streamlit as st

class VisionArtifactDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.backbone(x))

class AudioArtifactDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.sigmoid(self.fc(x))

@st.cache_resource
def load_models():
    vision_model = VisionArtifactDetector()
    audio_model = AudioArtifactDetector()
    vision_model.eval()
    audio_model.eval()
    return vision_model, audio_model

def process_video_frames(video_path, model, max_frames=5):
    cap = cv2.VideoCapture(video_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    scores, sampled_frames = [], []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // max_frames)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % step == 0 and len(scores) < max_frames:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            tensor_img = transform(pil_img).unsqueeze(0)
            
            with torch.no_grad():
                pred = model(tensor_img).item()
                
            scores.append(pred)
            sampled_frames.append(img_rgb)
        frame_count += 1
    cap.release()
    return (np.mean(scores) if scores else 0.5), sampled_frames

def process_audio_track(video_path, model):
    try:
        y, sr = librosa.load(video_path, sr=16000, duration=10.0)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_tensor = torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            score = model(mel_tensor).item()
        return score, mel_db
    except Exception:
        return None, None

st.set_page_config(page_title='FT-02 Deepfake Detector', layout='wide')
st.title('Multimodal Deepfake & Synthetic Media Detection System')
vision_model, audio_model = load_models()

uploaded_file = st.file_uploader('Upload video file (.mp4, .avi)', type=['mp4', 'avi'])
if uploaded_file is not None:
    temp_path = 'temp_input.mp4'
    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.read())
    col1, col2 = st.columns(2)
    with col1:
        st.video(temp_path)
    with col2:
        with st.spinner('Analyzing media streams...'):
            v_score, sampled_frames = process_video_frames(temp_path, vision_model)
            a_score, mel_db = process_audio_track(temp_path, audio_model)
            final_score = (v_score * 0.6 + a_score * 0.4) if a_score is not None else v_score
            fake_prob = final_score * 100
            if final_score > 0.5:
                st.error(f'DEEPFAKE DETECTED ({fake_prob:.2f}% Confidence)')
            else:
                st.success(f'AUTHENTIC MEDIA ({(100 - fake_prob):.2f}% Confidence)')
                
    st.markdown('---')
    st.subheader('Extracted Frame Samples')
    cols = st.columns(min(len(sampled_frames), 5))
    for idx, col in enumerate(cols):
        if idx < len(sampled_frames):
            col.image(sampled_frames[idx], caption=f'Frame {idx+1}')
