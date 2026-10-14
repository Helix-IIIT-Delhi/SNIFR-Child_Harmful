"""SNIFR: Audio-Visual Alignment for Child Harmful Content Detection"""

__version__ = "1.0.0"
__author__ = "Orchid Chetia Phukan, Mohd Mujtaba Akhtar, Girish, and others"

from .model import SNIFRModel
from .cross_transformer import CascadedCrossTransformer

__all__ = ["SNIFRModel", "CascadedCrossTransformer"]
