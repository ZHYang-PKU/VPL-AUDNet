A PyTorch implementation of VPL-AUDNet, a Transformer based model for user activity detection in massive MIMO systems, adapting to varying pilot lengths.
This is an unofficial replication of the paper:
"Generative Diffusion Model Driven Massive Random Access in Massive MIMO Systems" https://arxiv.org/abs/2505.12382


## 📖 Overview

VPLAUDNet is a deep learning framework that leverages Transformer architecture for efficient user activity detection in wireless communication systems. The model processes pilot signals in massive MIMO environments to detect active users with high accuracy.

## ✨ Features

- **Transformer-based Architecture**: Utilizes multi-head self-attention mechanisms for robust feature extraction
- **Massive MIMO Support**: Designed for systems with large antenna arrays (256+ antennas)
- **Pilot-assisted Detection**: Processes pilot signals for user activity identification
- **Flexible Configuration**: Configurable model dimensions, layers, and attention heads
- **Comprehensive Training**: Includes training, validation, and testing pipelines

## 🚀 Installation

### Prerequisites
- Python 3.6+
- PyTorch 1.9+
- CUDA-capable GPU (recommended)
