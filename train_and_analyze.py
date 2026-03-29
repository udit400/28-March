import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms, datasets, models
import numpy as np
import cv2
import soundfile as sf
import librosa
from tqdm import tqdm

# Paths
IMG_DIR = r"c:\Users\partn\Downloads\28 march img"
VIDEO_DIR = r"c:\Users\partn\Downloads\28 march videos"
VOICE_DIR = r"c:\Users\partn\Downloads\28 march voice"

# Image Model Training (Transfer Learning with ResNet18)
def train_image_model():
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(IMG_DIR, transform=data_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(dataset.classes))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(2):
        running_loss = 0.0
        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    torch.save(model.state_dict(), "image_model.pth")
    return f"Image model trained. Classes: {dataset.classes}"

# Video Model (Frame-based simple classifier)
def analyze_videos():
    results = []
    for file in os.listdir(VIDEO_DIR):
        if not file.lower().endswith((".mp4", ".avi", ".mov")):
            continue
        cap = cv2.VideoCapture(str(Path(VIDEO_DIR)/file))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        results.append({"file": file, "frames": frame_count})
        cap.release()
    return results

# Voice Model (MFCC feature extraction)
def analyze_voice():
    results = []
    for file in os.listdir(VOICE_DIR):
        if not file.lower().endswith((".wav", ".mp3")):
            continue
        path = str(Path(VOICE_DIR)/file)
        y, sr = librosa.load(path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        results.append({"file": file, "mfcc_shape": mfcc.shape})
    return results

if __name__ == "__main__":
    print("--- Image Model Training ---")
    print(train_image_model())
    print("\n--- Video Analysis ---")
    print(analyze_videos())
    print("\n--- Voice Analysis ---")
    print(analyze_voice())
