
# 🧠 Brain Tumor Segmentation using Diffusion-Augmented Swin Transformer


> <em>"Combining CNN efficiency, diffusion robustness, and transformer intelligence for next-gen medical imaging."</em>

---

## 📌 Overview

This project introduces a powerful deep learning pipeline for automatic **brain tumor segmentation** from MRI scans. By integrating a lightweight CNN, a diffusion module, and a Swin Transformer, the model achieves high precision in identifying tumor boundaries.

---

## 🧠 Key Highlights

- ✅ **MobileNetV3** as a fast and efficient feature extractor.  
- 💨 **Diffusion Module** enhances noisy and subtle tumor regions.  
- 🧩 **Swin Transformer** captures complex spatial relationships.  
- 🎯 **Custom Segmentation Head** for pixel-level mask generation.  
- 🔁 **Hybrid Loss Function** (BCE + Dice) for accurate training.  

---

## 🗂️ Dataset Details

- **Source**: [LGG Brain MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)  
- **Image Size**: 256 × 256 pixels  
- **Train/Val/Test Split**: 70% / 15% / 15%  

### 🔄 Augmentation Techniques

- Channel Dropout  
- Random Brightness & Contrast  
- Color Jitter  
- ImageNet-based normalization  

---

## 🧱 Architecture Overview

### 1️⃣ Feature Extraction  
Pretrained **MobileNetV3-small** extracts low-level spatial features.

### 2️⃣ Diffusion Enhancement  
A **100-timestep diffusion process** is used to refine features and recover subtle tumor boundaries.

### 3️⃣ Swin Transformer  
A **4-stage Transformer** with self-attention captures both global and local spatial context.

### 4️⃣ Segmentation Head  
Lightweight decoder projects features into a single-channel binary segmentation mask.

---

## ⚙️ Training Overview

- **Loss Function**: BCE + Dice hybrid  
- **Optimizer**: AdamW (lr = 1e-4, weight decay = 1e-4)  
- **Scheduler**: OneCycleLR  
- **Epochs**: 50 (with early stopping)  
- **Mixed Precision**: Enabled via `autocast` and `GradScaler`  
- **Device**: CUDA-enabled GPU  

---

## 📊 Evaluation Metrics

| Metric         | Score |
|----------------|--------|
| 🎯 Dice        | 0.92   |
| 📏 IoU         | 0.86   |
| 🎯 Precision   | 0.89   |
| 🔁 Recall      | 0.94   |
| 🧮 Accuracy    | 0.98   |

🧾 **Confusion Matrix**:

add the pictures


---

## 🌟 Innovations

| Component              | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| Diffusion Module       | Enhances low-contrast features by simulating noise removal and reconstruction. |
| Swin Transformer       | Provides attention across spatial regions for better boundary detection.     |
| Hybrid Architecture    | Combines CNN speed with transformer precision for medical segmentation.      |
| Gradient Checkpointing | Reduces memory usage to allow larger batch sizes during training.            |

---

## 🧪 Applications in Healthcare

- 🧠 Accurate tumor volume estimation  
- 🩺 Pre-operative surgical planning  
- 📈 Monitoring treatment response  
- 📆 Longitudinal study support  

---

## ⚠️ Limitations & Future Work

| Current Challenge                      | Future Direction                                                        |
|----------------------------------------|--------------------------------------------------------------------------|
| Requires high-quality MRI inputs       | Integrate super-resolution preprocessing or denoising techniques         |
| 2D slice-wise segmentation only        | Extend to full 3D volumetric segmentation                                |
| Focused on a single modality (T1)      | Expand to multi-modal fusion (T1, T2, FLAIR)                             |
| High compute demand during training    | Apply model pruning or distillation for edge deployment                  |

---

## 🚀 Getting Started

1. **Install dependencies** from `requirements.txt`  
2. **Prepare the dataset** using the structure outlined  
3. **Train the model** with the training script  
4. **Evaluate** performance on the test set  

---


---

## 🙌 Acknowledgements

- 📊 Dataset by Mateusz Buda on [Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)  
- ⚙️ Frameworks: PyTorch, Albumentations, timm, and torchvision  
- 💻 CUDA for GPU acceleration  

---

## 📬 Contact

**Shreyak Mukherjee**  
📧 shreyakmukherjeedgp@gmail.com  
📍 Durgapur, West Bengal  
📱 +91-9832188947  

---

⭐ *If this project helped you, consider giving it a star on GitHub!*



