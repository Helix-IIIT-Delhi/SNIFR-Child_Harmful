"""SNIFR Model for Child Harmful Content Detection"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from .encoders import VisualEncoder, AudioEncoder
from .cross_transformer import CascadedCrossTransformer


class SNIFRModel(nn.Module):
    """
    SNIFR: Audio-Visual Alignment for Child Harmful Content Detection

    Uses cascaded cross-transformer to align visual and audio modalities
    for fine-grained harmful content detection.

    Args:
        num_classes: Number of harmful content categories
        visual_dim: Visual feature dimension
        audio_dim: Audio feature dimension
        hidden_dim: Hidden dimension for fusion
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        pretrained: Load pretrained weights
    """

    def __init__(
        self,
        num_classes: int = 6,
        visual_dim: int = 2048,
        audio_dim: int = 768,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim

        # Encoders
        self.visual_encoder = VisualEncoder(
            output_dim=visual_dim,
            pretrained=pretrained
        )
        self.audio_encoder = AudioEncoder(
            output_dim=audio_dim,
            pretrained=pretrained
        )

        # Feature projection to common space
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)

        # Cascaded cross-transformer
        self.cross_transformer = CascadedCrossTransformer(
            d_model=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim * 4,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        video_frames: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for SNIFR model.

        Args:
            video_frames: Video frames [batch, num_frames, 3, H, W]
            audio_features: Audio features [batch, time_steps, audio_dim]

        Returns:
            Dict with harmful content logits and probabilities
        """
        # Encode visual features
        visual_features = self.visual_encoder(video_frames)  # [batch, num_frames, visual_dim]
        visual_features = self.visual_proj(visual_features)  # [batch, num_frames, hidden_dim]

        # Encode audio features
        audio_encoded = self.audio_encoder(audio_features)  # [batch, time_steps, audio_dim]
        audio_encoded = self.audio_proj(audio_encoded)  # [batch, time_steps, hidden_dim]

        # Cascaded cross-transformer for audio-visual alignment
        fused_visual, fused_audio = self.cross_transformer(
            visual_features,
            audio_encoded
        )

        # Pool temporal dimensions
        visual_pooled = torch.mean(fused_visual, dim=1)  # [batch, hidden_dim]
        audio_pooled = torch.mean(fused_audio, dim=1)   # [batch, hidden_dim]

        # Concatenate and classify
        combined = torch.cat([visual_pooled, audio_pooled], dim=1)
        logits = self.classifier(combined)
        probs = torch.softmax(logits, dim=1)

        return {
            "logits": logits,
            "probs": probs,
            "visual_features": fused_visual,
            "audio_features": fused_audio,
        }

    def predict(self, video_frames: torch.Tensor, audio_features: torch.Tensor) -> Dict:
        """
        Predict harmful content category.

        Args:
            video_frames: Video frames
            audio_features: Audio features

        Returns:
            Prediction dict with label and confidence
        """
        with torch.no_grad():
            output = self.forward(video_frames, audio_features)
            class_id = torch.argmax(output["probs"], dim=1)
            confidence = torch.max(output["probs"], dim=1)[0]

        categories = {
            0: "Safe",
            1: "Violence",
            2: "Explicit",
            3: "Self-Harm",
            4: "Hate-Speech",
            5: "Disturbing",
        }

        return {
            "class_id": class_id.item(),
            "label": categories[class_id.item()],
            "confidence": confidence.item(),
        }
