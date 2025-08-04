# Uncertainty-Aware Flood Segmentation with Conformal Prediction

> Official implementation of our IGARSS 2025 paper:  
> **Uncertainty-Aware Flood Segmentation from Sentinel-2 Observations with Conformal Prediction**  
> *Ioannis Konidakis, Klea Panayidou, Grigorios Tsagkatakis, Panagiotis Tsakalides*  
> [[PDF](./2025-IGARSS-UQ_CP.pdf)]

---

## Overview

This repository provides a framework for **uncertainty-aware flood segmentation** from **Sentinel-2 RGB imagery** using **Conformal Prediction (CP)**.

We apply both:
- **Inductive CP (ICP)**  
- **K-Fold CV+ CP**

...on top of the **OmbriaNet** bitemporal U-Net architecture.

![Block Diagram](./Figures/block_diagram_ICP_2.png)
![Block Diagram](./Figures/block_diagram_K_fold.png.png)

---

## Highlights

- Sentinel-2 (RGB) bitemporal input  
- Pixel-wise uncertainty via CP  
- Evaluation: coverage, inefficiency, accuracy

---

## Dataset Format

Each sample includes:

```
data/
├── BEFORE/   # pre-flood RGB image 
├── AFTER/    # post-flood RGB image 
└── MASK/     # binary label (1 = flood, 0 = background)
```

---

### Train with Inductive CP

```bash
python train_icp.py \
  --path ./data \
  --save_dir ./results_icp \
  --epochs 20 \
  --perc 0.9 \
  --temperature_scaling
```

---

### Evaluate on Test Set

```bash
python evaluate_cp_kfold.py \
  --model_path ./results_icp/Ombria_models_ICP0.9_ep20.pkl \
  --score_path ./results_icp/Ombria_scores_ICP0.9_ep20.pkl \
  --test_path ./data/test \
  --alpha 0.1
```

---

## Example Results

| Method     | Coverage (α=0.1) | Inefficiency | Mean IoU |
|------------|------------------|--------------|----------|
| ICP        | 0.89             | 1.66         | 0.70     |
| CV+ (K=5)  | 0.89             | 1.19         | 0.81     |

---

## Visualization

![Prediction Map](./Figures/coverage_inefficiency_diagram_histogram_05.png)

- Yellow: uncertain pixels (both 0 and 1 predicted)
- Blue: confident flood
- Green: confident background

---

## Acknowledgements

Supported by the **TITAN ERA Chair** (EU Horizon Europe, grant no. 101086741).
