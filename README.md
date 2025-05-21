# 🌊 Uncertainty-Aware Flood Segmentation with Conformal Prediction

![Flood Segmentation](docs/figures/cv_vs_icp.png)
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

- ✅ **Reliable Uncertainty Estimation** with Conformal Prediction
- 🔍 **Comparison of CP techniques**: ICP vs K-Fold CV+
- 🌐 Based on real **Sentinel-2 satellite imagery**
- 🧠 Built on top of a **bitemporal U-Net (OmbriaNet)** for flood segmentation
- 📈 Includes metrics like empirical coverage, inefficiency, and accuracy

---

## 🧠 Methodology

Our pipeline includes:
1. Preprocessing Sentinel-2 bitemporal imagery.
2. Training the OmbriaNet model for segmentation.
3. Applying Inductive CP or K-Fold CV+ to quantify uncertainty.
4. Visualizing uncertainty-aware prediction sets.

![Coverage Diagram](docs/figures/coverage_diagram.png)
*Figure: Coverage and inefficiency comparison between naive thresholding and CP.*

---

## 📁 Dataset Structure

The code supports datasets structured as follows:

