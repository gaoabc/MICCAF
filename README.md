# MICCAF
Multimodal Information Compression, Completion, and Adaptive Fusion for Cancer Survival Prediction
# Data source
[https://portal.gdc.cancer.gov/](https://portal.gdc.cancer.gov/)
# Requirements

## 1. Create a new conda environment.

```bash
conda create -n miccaf python=3.10
conda activate miccaf

## 2.Install the required packages.

```bash
torch == 2.3.0+cu121
timm == 0.9.8
torchvision == 0.18.0
numpy == 1.24.3

## or directly install environment by yaml file.

```bash
conda create -n miccaf -f requirements.yaml
