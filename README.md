# Generative-AI-Synthetic-Data-Augmentation-for-Addressing-Class-Imbalance-in-Machine-Learning-Models
Designed a Hybrid GAN-VAE model for credit card fraud detection, solving class imbalance. Improved recall (62.4%→91.4%) and AUC (0.82→0.95). Built a complete Python pipeline for data augmentation, validation, and retraining, outperforming standalone GAN/VAE methods.

ABSTRACT
Class imbalance has been a challenging phenomenon
for supervised learning, resulting in biased classifiers that are
biased towards the majority classes and show poor performance
for minority classes. This paper presents a Generative AIbased synthetic data augmentation framework that attempts to
alleviate the effects of imbalanced data. This approach exploits
the capabilities of deep generative learning techniques, such as
GANs and VAEs, for the generation of synthetic data for minority
classes. This framework combines data preprocessing, generative
model training, synthetic data validation, and model retraining
into a single Python-based framework. Experiments conducted
with the proposed approach show significant improvements in
metrics such as precision, recall, F1-score, and balanced accuracy,
along with a reduction in the overall bias towards majority
classes. When compared with traditional resampling methods, the
generative model-based synthetic data augmentation shows better
generalization and robustness. This demonstrates the potential
for the development of synthetic data generation techniques for
improving the overall performance and fairness of imbalanced
machine learning classifiers
