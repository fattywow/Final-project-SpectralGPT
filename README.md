# Final-project-SpectralGPT
# SpectralGPT: Semantic Change Detection for Remote Sensing Imagery

This repository contains code and experiments for **semantic change detection** using SpectralGPT, a transformer-based model adapted for remote sensing tasks. The focus is to detect semantic changes between satellite image patches (T1 and T2) and generate corresponding change maps and transition labels.

## Dataset Structure

Each `.npz` file contains:
- `T1`: (10, 128, 128) image at time 1
- `T2`: (10, 128, 128) image at time 2
- `T1_label`, `T2_label`: (128, 128) semantic labels
- `change_map`: (128, 128) binary map
- `transition_label`: (128, 128) semantic transitions

## Usage

Run `SpectralGPT.ipynb` in Jupyter Notebook. Modify the dataset path as needed. The notebook trains a change detection model and evaluates it using F1 score and transition accuracy.

## Environment

Install dependencies with:

```bash
pip install torch torchvision einops matplotlib scikit-learn
