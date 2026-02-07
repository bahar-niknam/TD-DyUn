# TD-DyUn: Dynamic Uncertainty-Aware Dataset Pruning

This repository accompanies the paper:

**Large-Scale Dataset Pruning Using Dynamic Uncertainty-Aware Scoring**

## 📌 Overview
Training deep neural networks on large-scale datasets is computationally expensive and often inefficient due to redundant or weakly informative samples.  
**TD-DyUn** is a dynamic, uncertainty-aware dataset pruning framework that identifies and removes low-impact samples during training while preserving model performance.

The method leverages:
- Temporal dynamics of model outputs
- Predictive uncertainty evolution
- Distributional instability via temporally smoothed KL-divergence

TD-DyUn enables substantial data reduction with minimal accuracy degradation, making it suitable for large-scale and resource-constrained training scenarios.

---

## 🧠 Key Contributions
- **Dynamic sample scoring** based on temporal uncertainty, rather than static heuristics  
- **Dual-depth temporal analysis** combining short-term instability and long-term learning trends  
- **Importance-weighted retraining** to compensate for aggressive data pruning  
- Strong empirical performance on **CIFAR-10** and **CIFAR-100** across multiple architectures

---

## 📄 Paper
**Large-Scale Dataset Pruning Using Dynamic Uncertainty-Aware Scoring**  
Bahar Niknam, Hedieh Sajedi  
Department of Computer Science, University of Tehran  

📎 PDF: [`TD_DyUn_Paper.pdf`](./paper/TD_DyUn_Paper.pdf)

---

## 🧩 Method Summary
TD-DyUn operates in two phases:

### Phase I: Importance Estimation
- Train the model on the full dataset
- Record logits for all samples across epochs
- Compute:
  - Weighted Dynamic Uncertainty (correct-class probability fluctuation)
  - Temporally Smoothed Dual-Depth Score (distributional instability)

### Phase II: Pruning & Retraining
- Rank samples based on combined importance score
- Retain top $(1 - p)$ fraction of data
- Retrain the model using importance-weighted loss

---

## 🏗 Repository Structure
