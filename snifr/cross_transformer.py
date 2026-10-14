"""Cascaded Cross-Transformer for Audio-Visual Alignment"""

import torch
import torch.nn as nn


class CascadedCrossTransformer(nn.Module):
    """
    Cascaded cross-transformer for audio-visual alignment.

    Progressive layers of cross-attention between audio and visual modalities.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        hidden_dim: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        # Cascaded cross-attention layers
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, num_heads, hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        visual_features: torch.Tensor,
        audio_features: torch.Tensor,
    ):
        """
        Progressive audio-visual alignment.

        Args:
            visual_features: [batch, num_frames, d_model]
            audio_features: [batch, num_audio_frames, d_model]

        Returns:
            Aligned visual and audio features
        """
        for layer in self.layers:
            visual_features, audio_features = layer(visual_features, audio_features)

        return visual_features, audio_features


class CrossAttentionLayer(nn.Module):
    """
    Single cross-attention layer for audio-visual alignment.
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, hidden_dim: int = 2048):
        super().__init__()

        # Visual self-attention
        self.visual_self_attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )

        # Audio self-attention
        self.audio_self_attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )

        # Cross-attention (visual attends to audio)
        self.visual_cross_attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )

        # Cross-attention (audio attends to visual)
        self.audio_cross_attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )

        # Feed-forward networks
        self.visual_ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )

        self.audio_ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )

        # Layer normalization
        self.visual_norm1 = nn.LayerNorm(d_model)
        self.visual_norm2 = nn.LayerNorm(d_model)
        self.visual_norm3 = nn.LayerNorm(d_model)

        self.audio_norm1 = nn.LayerNorm(d_model)
        self.audio_norm2 = nn.LayerNorm(d_model)
        self.audio_norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        visual_features: torch.Tensor,
        audio_features: torch.Tensor,
    ):
        """
        Cross-attention alignment between visual and audio.

        Args:
            visual_features: [batch, num_frames, d_model]
            audio_features: [batch, num_audio, d_model]

        Returns:
            Aligned visual and audio features
        """
        # Visual self-attention
        visual_attn, _ = self.visual_self_attn(visual_features, visual_features, visual_features)
        visual_features = self.visual_norm1(visual_features + self.dropout(visual_attn))

        # Audio self-attention
        audio_attn, _ = self.audio_self_attn(audio_features, audio_features, audio_features)
        audio_features = self.audio_norm1(audio_features + self.dropout(audio_attn))

        # Visual cross-attention (attends to audio)
        visual_cross, _ = self.visual_cross_attn(visual_features, audio_features, audio_features)
        visual_features = self.visual_norm2(visual_features + self.dropout(visual_cross))

        # Audio cross-attention (attends to visual)
        audio_cross, _ = self.audio_cross_attn(audio_features, visual_features, visual_features)
        audio_features = self.audio_norm2(audio_features + self.dropout(audio_cross))

        # Feed-forward networks
        visual_ffn = self.visual_ffn(visual_features)
        visual_features = self.visual_norm3(visual_features + self.dropout(visual_ffn))

        audio_ffn = self.audio_ffn(audio_features)
        audio_features = self.audio_norm3(audio_features + self.dropout(audio_ffn))

        return visual_features, audio_features
