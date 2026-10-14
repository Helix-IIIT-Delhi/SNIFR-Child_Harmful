"""Visual and Audio Encoders for SNIFR"""

import torch
import torch.nn as nn


class VisualEncoder(nn.Module):
    """
    Visual encoder for frame-level feature extraction.

    Uses ResNet backbone with transformer encoder for temporal modeling.
    """

    def __init__(self, output_dim: int = 2048, pretrained: bool = True):
        super().__init__()
        self.output_dim = output_dim

        # Pretend ResNet backbone (simplified)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=8,
            dim_feedforward=256,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=3
        )

        # Projection to output dimension
        self.projection = nn.Linear(64, output_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features from video frames.

        Args:
            frames: [batch, num_frames, 3, H, W]

        Returns:
            Visual features [batch, num_frames, output_dim]
        """
        batch_size, num_frames = frames.shape[:2]

        # Reshape for CNN
        frames = frames.view(batch_size * num_frames, 3, frames.shape[3], frames.shape[4])

        # Extract features
        features = self.backbone(frames)
        features = features.view(batch_size * num_frames, -1)
        features = features.view(batch_size, num_frames, -1)

        # Transformer encoding
        encoded = self.transformer_encoder(features)

        # Project to output dimension
        output = self.projection(encoded)

        return output


class AudioEncoder(nn.Module):
    """
    Audio encoder for acoustic feature extraction.

    Combines speech and sound event features.
    """

    def __init__(self, output_dim: int = 768, pretrained: bool = True):
        super().__init__()
        self.output_dim = output_dim

        # Audio feature processor
        self.processor = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
        )

        # Transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=3
        )

        # Projection to output dimension
        self.projection = nn.Linear(128, output_dim)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Extract audio features from audio signal.

        Args:
            audio_features: [batch, time_steps, feature_dim]

        Returns:
            Audio features [batch, time_steps, output_dim]
        """
        # Process audio features
        processed = self.processor(audio_features)

        # Transformer encoding
        encoded = self.transformer_encoder(processed)

        # Project to output dimension
        output = self.projection(encoded)

        return output
