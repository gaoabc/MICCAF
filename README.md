# MICCAF
Multimodal Information Compression, Completion, and Adaptive Fusion for Cancer Survival Prediction
# Data source
[https://portal.gdc.cancer.gov/](https://portal.gdc.cancer.gov/)
# Requirements

## 1. Create a new conda environment.

```bash
conda create -n miccaf python=3.8.0
conda activate miccaf
```
## 2.Install the required packages.

```bash
python==3.8.0
pandas=2.0.3
torch==2.3.1
matplotlib=3.7.1
torchvision==0.18.1
scikit-survival==0.22.2
opencv-python=4.10.0.84
...
```
## or directly install environment by yaml file.

```bash
conda create -n miccaf -f requirements.yaml
```
