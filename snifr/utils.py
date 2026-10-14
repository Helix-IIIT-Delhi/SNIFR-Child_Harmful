"""Utility Functions for SNIFR"""

import torch
import numpy as np
import cv2
import librosa


def load_video(video_path: str, num_frames: int = 30, sample_rate: int = 16000):
    """Load video and extract frames and audio"""
    cap = cv2.VideoCapture(video_path)
    frames = []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, total_frames // num_frames)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % stride == 0 and len(frames) < num_frames:
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)

        frame_count += 1

    cap.release()

    # Load audio
    audio_path = video_path.replace('.mp4', '.wav')
    audio, _ = librosa.load(audio_path, sr=sample_rate)

    # Convert frames to tensor
    frames = np.array(frames)
    frames = torch.from_numpy(frames).float() / 255.0
    frames = frames.permute(0, 3, 1, 2)  # [num_frames, 3, H, W]

    # Convert audio to features
    audio = torch.from_numpy(audio).float()

    return frames, audio


def seed_everything(seed: int = 42) -> None:
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)


HARM_CATEGORIES = {
    0: "Safe",
    1: "Violence",
    2: "Explicit",
    3: "Self-Harm",
    4: "Hate-Speech",
    5: "Disturbing",
}


def category_id_to_label(category_id: int) -> str:
    """Convert category ID to label"""
    return HARM_CATEGORIES.get(category_id, "Unknown")
