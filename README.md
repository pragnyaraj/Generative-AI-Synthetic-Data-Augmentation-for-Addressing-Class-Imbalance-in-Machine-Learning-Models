# 🧠 Generative AI-Driven Synthetic Data Augmentation for Addressing Class Imbalance in Machine Learning Models

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Research-purple?style=for-the-badge)

**A comparative study of Generative AI methods (VAE, WGAN-GP, Hybrid GAN-VAE) for synthetic minority oversampling in highly imbalanced credit card fraud detection.**

*Errolla Pragnya · Kuppam Akash ·  — Dept. of CSE (AIML), Chandigarh University, Mohali, India*

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Proposed Framework](#-proposed-framework)
- [Methods Compared](#-methods-compared)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Key Findings](#-key-findings)
- [Graphical Abstract](#-graphical-abstract)
- [Applications](#-applications)
- [Authors](#-authors)
- [Citation](#-citation)
- [License](#-license)

---

## 🔍 Overview

Class imbalance is one of the most persistent challenges in real-world machine learning — particularly in domains like **financial fraud detection**, **medical diagnosis**, and **cybersecurity intrusion detection**, where the minority class is catastrophically underrepresented.

This project proposes and evaluates a **Hybrid GAN-VAE model** that combines the probabilistic encoding strengths of a **Variational Autoencoder (VAE)** with the adversarial refinement power of a **Wasserstein GAN with Gradient Penalty (WGAN-GP)** to generate high-quality, diverse synthetic minority-class samples.

The augmented data is used to train an **XGBoost classifier**, and performance is benchmarked against five competing methods across six evaluation metrics.

---

## ⚠️ Problem Statement

The **MLG-ULB Credit Card Fraud Detection** dataset (Kaggle) exhibits an extreme class imbalance:

| Attribute | Value |
|---|---|
| Total Transactions | 284,807 |
| Legitimate (Class 0) | 284,315 **(99.83%)** |
| Fraudulent (Class 1) | 492 **(0.17%)** |
| **Imbalance Ratio (IR)** | **578 : 1** |
| Features | V1–V28 (PCA), Amount, Time |

This severe imbalance causes standard classifiers to become **heavily biased toward the majority class**, resulting in low minority-class recall, misleading accuracy, and poor MCC/F1-Score.

---

## 📦 Dataset

**Source:** [MLG-ULB Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 transactions by European cardholders (September 2013)
- Features `V1–V28` are PCA-transformed for confidentiality
- `Amount` and `Time` are the only non-transformed features
- Binary target: `0` = Legitimate, `1` = Fraudulent

> **Note:** Due to licensing, the dataset is not included. Download it from Kaggle and place it at `data/creditcard.csv`.

---

## 🏗️ Proposed Framework

```
Raw Imbalanced Data
        │
        ▼
┌─────────────────────┐
│   Data Preprocessing │  ← Normalization, train/test split
└─────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  Generative Model Training│  ← VAE / GAN / Hybrid GAN-VAE
└──────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│ Synthetic Data Generation +   │  ← Minority class augmentation
│ Statistical Validation        │
└──────────────────────────────┘
        │
        ▼
┌────────────────────┐
│  XGBoost Classifier │
└────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│  6-Metric Evaluation (5-Fold CV)          │
│  Recall · Precision · F1 · BalAcc · AUC · MCC │
└──────────────────────────────────────────┘
```

---

## 🔬 Methods Compared

### Traditional Baselines

| Method | Description |
|---|---|
| **Baseline** | No augmentation; raw imbalanced data fed to XGBoost. |
| **Random Oversampling** | Randomly duplicates minority samples. Prone to overfitting. |
| **SMOTE** | Synthetic samples via linear interpolation of k-nearest neighbors. |

### Generative AI Methods

| Method | Description |
|---|---|
| **VAE-Based** | Encodes minority class into latent space `z ~ N(0,I)`; decoded samples used for augmentation. Loss: `Reconstruction + KL Divergence`. |
| **GAN-Based (WGAN-GP)** | Generator learns minority distribution; Discriminator enforces realism. Gradient Penalty stabilizes training. |
| **⭐ Hybrid GAN-VAE (Proposed)** | VAE probabilistic encoding feeds into GAN adversarial refinement for high-quality, diverse samples. |

---

## 📊 Results

All results from **5-fold cross-validation**. Paired t-test p = 0.004 (p < 0.01). Variance ±1.2%.

| Method | Recall (%) | Precision (%) | F1-Score | Bal. Acc. (%) | AUC-ROC | MCC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline | 62.4 | 74.1 | 0.68 | 71.5 | 0.82 | 0.59 |
| Random Oversampling | 74.8 | 78.6 | 0.76 | 79.2 | 0.87 | 0.70 |
| SMOTE | 78.3 | 80.4 | 0.79 | 82.5 | 0.89 | 0.74 |
| VAE-Based | 84.7 | 83.2 | 0.84 | 86.8 | 0.91 | 0.78 |
| GAN (WGAN-GP) | 88.6 | 85.9 | 0.87 | 89.3 | 0.93 | 0.81 |
| **⭐ Hybrid GAN-VAE** | **91.4** | **87.5** | **0.89** | **91.1** | **0.95** | **0.83** |

---

## 📁 Project Structure

```
📦 imbalance-augmentation/
├── 📂 data/
│   └── creditcard.csv
├── 📂 models/
│   ├── vae_model.py
│   ├── wgan_gp_model.py
│   └── hybrid_gan_vae.py
├── 📂 augmentation/
│   ├── baseline.py
│   ├── random_oversampling.py
│   ├── smote_augmentation.py
│   └── generative_augmentation.py
├── 📂 evaluation/
│   ├── metrics.py
│   └── cross_validation.py
├── 📂 notebooks/
│   └── imbalance_project.ipynb
├── 📂 outputs/
│   ├── graphical_abstract.html
│   └── results/
├── 📂 paper/
│   └── Generative_AI_Synthetic_Data_Augmentation.pdf
├── requirements.txt
├── README.md
└── LICENSE
```

---

## ⚙️ Installation

```bash
git clone https://github.com/<pragnyaraj/](https://github.com/pragnyaraj/Generative-AI-Synthetic-Data-Augmentation-for-Addressing-Class-Imbalance-in-Machine-Learning-Models.git)imbalance-augmentation.git
cd imbalance-augmentation
pip install -r requirements.txt
```

**requirements.txt:**
```
tensorflow>=2.13.0
xgboost>=1.7.6
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.11.0
jupyter>=1.0.0
xgboost
```

---

## 🚀 Usage

```bash
jupyter notebook notebooks/imbalance_project.ipynb
```

```python
# Hybrid GAN-VAE (Proposed)
from models.hybrid_gan_vae import HybridGANVAE

hybrid = HybridGANVAE(latent_dim=16, input_dim=30)
hybrid.train(X_minority, epochs=500, batch_size=64)
X_synthetic = hybrid.generate(n_samples=5000)

# Evaluate all methods
from evaluation.cross_validation import run_full_comparison
results = run_full_comparison(X, y, n_splits=5)
```

---

## 🏆 Key Findings

| Metric | Baseline | Hybrid GAN-VAE | Improvement |
|---|:---:|:---:|:---:|
| Recall | 62.4% | **91.4%** | **+29.0%** |
| Precision | 74.1% | **87.5%** | **+13.4%** |
| F1-Score | 0.68 | **0.89** | **+0.21** |
| Balanced Accuracy | 71.5% | **91.1%** | **+19.6%** |
| AUC-ROC | 0.82 | **0.95** | **+0.13** |
| MCC | 0.59 | **0.83** | **+0.24** |

The Hybrid model succeeds because VAE provides a smooth, structured latent space while the GAN adversarial loop refines samples to be statistically indistinguishable from real data — achieving diversity and realism neither model reaches alone.

---

## 🌐 Applications

- 🏥 **Healthcare** — rare disease diagnosis, cancer detection
- 🔐 **Cybersecurity** — intrusion detection, malware classification
- 💳 **Financial Services** — fraud detection, anti-money laundering
- 🏭 **Manufacturing** — defect detection in quality control
- 📡 **Telecommunications** — churn prediction, SIM fraud

---

## 👥 Authors

| Name | Affiliation |
|---|---|
| **Errolla Pragnya** | Dept. of CSE (AIML), Chandigarh University, Mohali, India |
| **Kuppam Akash** | Dept. of CSE (AIML), Chandigarh University, Mohali, India |


---

## 📄 Citation

```bibtex
@article{pragnya2026ganvae,
  title     = {Generative AI-Driven Synthetic Data Augmentation for Addressing Class Imbalance in Machine Learning Models},
  author    = {Errolla Pragnya and Kuppam Akash and Reema},
  year      = {2026},
  institution = {Chandigarh University, Mohali, India}
}
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  Chandigarh University, Mohali, India
⭐ If this helped your research, please give it a star!
</div>
