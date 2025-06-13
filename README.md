# 🌊 Uncertainty-Aware Flood Segmentation with Conformal Prediction

![Flood Segmentation](docs/figures/block_diagram_ICP_2.png)
![Flood Segmentation](docs/figures/block_diagram_K_fold.png)
*Figure: Comparison of K-Fold CV+ and Inductive Conformal Prediction in segmenting flood zones with uncertainty estimates.*

## 📄 Overview

This repository contains the official implementation of the paper:

> **Uncertainty-Aware Flood Segmentation from Sentinel 2 Observations with Conformal Prediction**  
> *Ioannis Konidakis, Klea Panayidou, Grigorios Tsagkatakis, Panagiotis Tsakalides*  
> Presented at IGARSS 2025  
> [[PDF](docs/IGARSS_2025.pdf)]

We introduce Conformal Prediction (CP) for flood segmentation using Sentinel-2 data and evaluate both **Inductive CP (ICP)** and **K-Fold Cross-Validation CP (CV+)** on the **OmbriaNet** architecture. This approach enhances trust in predictions by offering statistically valid uncertainty estimates.

---

## ✨ Key Features

- **Reliable Uncertainty Estimation** with Conformal Prediction
- **Comparison of CP techniques**: ICP vs K-Fold CV+
- Built on top of a **bitemporal U-Net (OmbriaNet)** for flood segmentation

---

## 🧠 Methodology

Our pipeline includes:
1. Training the OmbriaNet model for segmentation.
2. Applying Inductive CP or K-Fold CV+ to quantify uncertainty.
3. Visualizing uncertainty-aware prediction sets.

![Coverage Diagram](docs/figures/coverage_inefficiency_diagram_histogram_05.png)
*Figure: Coverage and inefficiency comparison between naive thresholding and CP.*

---

## 📁 Dataset Structure

The code supports datasets structured as follows:

