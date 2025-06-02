# Final-project-SpectralGPT  
## SpectralGPT: Semantic Change Detection for Remote Sensing Imagery

Due to GitHub file size limits, the dataset used in this project—**SegMunich**—is hosted externally on Zenodo (Given by the paper)  
You can download it here:  
🔗 [https://zenodo.org/records/8412455](https://zenodo.org/records/8412455)

---

## Motivation: Why Semantic Change Detection?

Semantic change detection plays a vital role in Earth observation and remote sensing. Unlike binary change detection—which only determines whether a change has occurred—**semantic** change detection aims to **classify the type of change** between two satellite images. This is critical for:

Semantic change detection plays a vital role in Earth observation and remote sensing.  
Unlike binary change detection—which only determines whether a change has occurred—**semantic** change detection aims to **classify the type of change** between two satellite images. This is crucial for understanding not just *that* something changed, but *how* it changed.

For example, **crops near a river may be gradually eroded and replaced by water**, but are **unlikely to suddenly become rock**. Understanding this **contextual and semantic transformation** is beyond the capacity of traditional change detection.

Simply feeding time-point images (T1 and T2) into semantic segmentation models independently **cannot capture temporal differences or transitions**, as they lack awareness of prior state.  
This is where semantic change detection excels—it leverages temporal and contextual information to **understand and classify meaningful transitions**, rather than treating each image in isolation.

Traditional methods often rely on pixel-wise differences or shallow models, which struggle with complex scenes.  
**SpectralGPT**, built upon transformer architecture, learns deeper temporal and semantic representations to address these challenges effectively.

---

## Data Preprocessing

This project includes an automated preprocessing pipeline to generate training samples suitable for semantic change detection using the SegMunich dataset. The pipeline performs the following steps:

---

## 1. Paired Sample Selection

To generate meaningful temporal image pairs (`T1`, `T2`), the script computes spectral and spatial similarity between Segmunich dataset:

- Each 10-band image is flattened into pixel vectors and reduced via PCA.
- Cosine similarity is calculated across all image vectors.
- The top-N most similar image pairs (default: 2000) are selected, ensuring each image is only used once.

This selection strategy ensures that the paired images are spectrally close but can differ semantically, simulating real-world gradual transitions like urban growth or seasonal land cover changes.

---

## 2. Binary Change Map Generation (`change_map`)

The change map identifies whether each pixel has undergone a semantic change between two time points.

For each image pair, their semantic segmentation labels `T1_label` and `T2_label` are compared pixel-by-pixel:

- Pixels where labels differ are marked as `1` (changed).
- Pixels where labels are identical remain `0` (unchanged).

This produces a binary mask indicating change regions, which is used as supervision for binary change detection.

---

## 3. Transition Label Encoding (`transition_label`)

Beyond detecting that a change occurred, we encode **what type of change** happened using a transition label.

Each unique class-to-class transition is assigned a single integer using the formula:

```python
transition_label = T1_index * num_classes + T2_index
```

Where:
- `T1_index`: class index at time T1
- `T2_index`: class index at time T2
- `num_classes`: total number of semantic classes (e.g., 13)

For example, a transition from class 3 to 7 with 13 total classes is encoded as:

```
transition_label = 3 * 13 + 7 = 46
```

This allows the model to learn and predict both the source and target semantics in one unified label map.

---

## 4. Output `.npz` Structure

After preprocessing, each `.npz` file contains all the necessary components for semantic change detection training:

- `T1`: Satellite image at time 1, shape `[10, 128, 128]`
- `T2`: Satellite image at time 2, shape `[10, 128, 128]`
- `T1_label`: Semantic label map for T1
- `T2_label`: Semantic label map for T2
- `change_map`: Binary change mask (`0`: no change, `1`: change)
- `transition_label`: Encoded semantic transitions (e.g., 3 → 7 → 46)

All `.npz` files are saved under the directory:

```
/paired_data/with_transition/
```

These files serve as direct input for training SpectralGPT on semantic change detection tasks.

---

## Dataset Structure (.npz files)

Each `.npz` file represents a paired sample and contains:

- `T1`: Satellite image at time 1, shape `(10, 128, 128)`
- `T2`: Satellite image at time 2, shape `(10, 128, 128)`
- `T1_label`: Semantic label for T1, shape `(128, 128)`
- `T2_label`: Semantic label for T2, shape `(128, 128)`
- `change_map`: Binary map showing changed pixels, shape `(128, 128)`
- `transition_label`: Semantic transitions between classes, shape `(128, 128)`

---

## Usage

Run `SpectralGPT.ipynb` or `SpectralGPT.py` (They are the same)

1. Modify the dataset directory path.
2. Load paired `.npz` files into the dataloader.
3. Train the model using the provided SpectralGPT architecture.
4. Evaluate performance using:
   - Transition label accuracy
   - T1/T2 accuracy
---

## Model Overview

SpectralGPT adapts transformer-based architecture to handle multi-spectral, temporal data.  
It processes `T1` and `T2` images through separate encoders, fuses temporal features using a decoder, and predicts both:

- **Change map**: Whether a pixel has changed (Binary)
- **Transition map**: What it has changed into (e.g., grass → building)

This dual-task learning enhances contextual understanding and model robustness.

---

## Environment Setup

Install dependencies with:

```bash
pip install torch torchvision einops matplotlib scikit-learn
