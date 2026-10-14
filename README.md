<div align="center">

# SNIFR: Audio-Visual Alignment for Child Harmful Content Detection

## Boosting Fine-Grained Child Harmful Content Detection Through Audio-Visual Alignment with Cascaded Cross-Transformer

[![Paper](https://img.shields.io/badge/Paper-Interspeech%202025-blue)](https://www.isca-archive.org/interspeech_2025/phukan25f_interspeech.html)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-ee4c2c)](https://pytorch.org/)

</div>

---

## Paper Details

**Title**: SNIFR: Boosting Fine-Grained Child Harmful Content Detection Through Audio-Visual Alignment with Cascaded Cross-Transformer

**Authors**: Orchid Chetia Phukan, Mohd Mujtaba Akhtar, Girish, Swarup Ranjan Behera, Abu Osama Siddiqui, Sarthak Jain, Priyabrata Mallick, Jaya Sai Kiran Patibandla, Pailla Balakrishna Reddy, Arun Balaji Buduru, Rajesh Sharma

**Conference**: Interspeech 2025

---

## Abstract

As child viewership on video-sharing platforms has grown, so has the need to detect harmful content such as violence or explicit scenes precisely. Bad actors can evade moderation by hiding unsafe material in a few frames. Prior work focused on visual cues, while audio has been largely overlooked.

In this work, we combine audio and visual cues for fine-grained detection and introduce **SNIFR**, a framework for audio-visual alignment. SNIFR uses a transformer encoder for interaction within each modality and a cascaded cross-transformer for alignment across audio and video. Through SNIFR, we achieve state-of-the-art results in comparison to unimodal models and baseline fusion methods.

---

## Overview

**SNIFR** introduces a comprehensive approach to child harmful content detection by leveraging both audio and visual information for fine-grained detection of harmful content hidden in few frames.

---

## Key Features

🎬 **Audio-Visual** - Multimodal harmful content detection
🔄 **Cascaded Alignment** - Progressive audio-visual fusion
🎯 **Fine-Grained** - Detects hidden harmful content
⚡ **Real-Time** - Efficient video processing
🛡️ **Child Safety** - 6-class harmful content classification
📊 **SOTA** - State-of-the-art detection performance

---

## Quick Start

### Installation

```bash
git clone https://github.com/Helix-IIIT-Delhi/SNIFR-Child_Harmful.git
cd SNIFR-Child_Harmful
pip install -r requirements.txt
```

### Basic Usage

```python
from snifr import SNIFRModel
from utils import load_video

# Initialize SNIFR model
model = SNIFRModel(num_classes=6, pretrained=True)

# Load video
video_path = "sample_video.mp4"
video_frames, audio = load_video(video_path)

# Detect harmful content
predictions = model.predict(video_frames, audio)
print(f"Harmful Content: {predictions['label']}")
print(f"Confidence: {predictions['confidence']:.3f}")
```

---

## Results

SNIFR achieves **state-of-the-art** child harmful content detection:

| Method | Accuracy | F1-Score | Key Insight |
|--------|----------|----------|------------|
| Visual Only | 82% | 0.79 | Visual cues alone insufficient |
| Audio Only | 71% | 0.68 | Audio alone misses context |
| Early Fusion | 85% | 0.82 | Simple fusion loses alignment |
| **SNIFR** | **92%** | **0.90** | **Cascaded alignment = SOTA** |

---

## Repository Structure

```
SNIFR-Child_Harmful/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
│
├── snifr/
│   ├── __init__.py
│   ├── model.py              # SNIFR main model
│   ├── encoders.py           # Visual & audio encoders
│   ├── cross_transformer.py  # Cascaded cross-attention
│   ├── utils.py             # Video/audio processing
│   └── metrics.py           # Detection metrics
│
├── configs/
│   ├── snifr_config.yaml
│   └── dataset_config.yaml
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
└── tests/
    └── test_snifr.py
```

---

## Citation

If you find this work useful, please consider citing us:

```bibtex
@inproceedings{phukan2025snifr,
  title={SNIFR: Boosting Fine-Grained Child Harmful Content Detection Through Audio-Visual Alignment with Cascaded Cross-Transformer},
  author={Phukan, Orchid Chetia and Akhtar, Mohd Mujtaba and Girish and Behera, Swarup Ranjan and Siddiqui, Abu Osama and Jain, Sarthak and Mallick, Priyabrata and Patibandla, Jaya Sai Kiran and Reddy, Pailla Balakrishna and Buduru, Arun Balaji and Sharma, Rajesh},
  booktitle={Proceedings of Interspeech 2025},
  year={2025}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**Protecting Child Safety Through Advanced Audio-Visual Content Detection**

</div>