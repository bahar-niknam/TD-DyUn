# TD-DyUn: Dynamic Uncertainty-Aware Dataset Pruning

This repository presents **TD-DyUn**, a dynamic uncertainty-aware framework for dataset pruning in deep learning.

---

## Overview
Training deep neural networks on large-scale datasets is increasingly expensive in terms of computation, memory, and energy, while performance gains often saturate.  
TD-DyUn addresses this challenge by **dynamically identifying and removing low-impact training samples during training**, while preserving model performance and training stability.

The method exploits:
- Temporal dynamics of model predictions
- Predictive uncertainty evolution across epochs
- Distributional instability measured via temporally smoothed divergence

This enables efficient dataset reduction with minimal accuracy degradation.

---

## Key Contributions
- **Dynamic sample importance scoring** based on temporal uncertainty rather than static heuristics  
- **Dual-depth temporal analysis** capturing both short-term instability and long-term learning behavior  
- **Importance-weighted retraining** to compensate for aggressive data pruning  
- Strong empirical performance on CIFAR-10 and CIFAR-100 across multiple architectures

---

## Paper
**Large-Scale Dataset Pruning Using Dynamic Uncertainty-Aware Scoring**  

*Manuscript in preparation.*

This repository is intended to accompany the manuscript and provide an overview of the proposed method.

---

## Method Summary
TD-DyUn operates in two main phases:

### Phase I: Importance Estimation
- Train the model on the full dataset
- Record logits for all training samples across epochs
- Compute:
  - Weighted Dynamic Uncertainty (fluctuations of correct-class probability)
  - Temporally Smoothed Dual-Depth Score (instability of output distributions)

### Phase II: Pruning and Retraining
- Rank samples based on the combined TD-DyUn importance score
- Retain the top $(1 - p)$ fraction of training samples
- Retrain the model from scratch using an importance-weighted loss

---

## Repository Status
This repository currently contains documentation only.

