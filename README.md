
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

<div align="center">

| Metric         | Score |
|----------------|--------|
| 🎯 Dice        | 0.92   |
| 📏 IoU         | 0.86   |
| 🎯 Precision   | 0.89   |
| 🔁 Recall      | 0.94   |
| 🧮 Accuracy    | 0.98   |

</div>


🧾 **Output Images**:

🧾 **Output Images**:

<div align="center">
  <img src="https://github.com/shreyakmukherjee/brain-tumor-segmentation-diffusion-swin/blob/09ce25befedc627331d3eec924ee9658131db843/Images/Confusion_Matrix.png" alt="Image 1" width="45%" style="margin: 10px;" />
  <img src="https://github.com/shreyakmukherjee/brain-tumor-segmentation-diffusion-swin/blob/09ce25befedc627331d3eec924ee9658131db843/Images/Dice_Coefficient.png" alt="Image 2" width="45%" style="margin: 10px;" />
</div>

<div align="center">
  <img src="https://github.com/shreyakmukherjee/brain-tumor-segmentation-diffusion-swin/blob/09ce25befedc627331d3eec924ee9658131db843/Images/Hitmap.png" alt="Image 3" width="45%" style="margin: 10px;" />
  <img src="https://github.com/shreyakmukherjee/brain-tumor-segmentation-diffusion-swin/blob/09ce25befedc627331d3eec924ee9658131db843/Images/Loss_curve.png" alt="Image 4" width="45%" style="margin: 10px;" />
</div>

<div align="center">
  <img src="https://github.com/shreyakmukherjee/brain-tumor-segmentation-diffusion-swin/blob/09ce25befedc627331d3eec924ee9658131db843/Images/ROC_curve.png" alt="Image 5" width="45%" style="margin: 10px;" />
  <img src="https://github.com/shreyakmukherjee/brain-tumor-segmentation-diffusion-swin/blob/09ce25befedc627331d3eec924ee9658131db843/Images/Recall_curve.png" alt="Image 6" width="45%" style="margin: 10px;" />
</div>



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


## 🚀 How to Execute This Project

Follow the steps below to clone and run this project on your machine:

---

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/brain-tumor-segmentation.git
cd brain-tumor-segmentation
```

---

### 2️⃣ Install Required Libraries

Make sure Python 3.8+ is installed. Then install the dependencies using:

```bash
pip install -r requirements.txt
```

✅ All required packages (including PyTorch, torchvision, timm, albumentations, etc.) are listed in the `requirements.txt`.

---

### 3️⃣ Download the Pretrained Model

Before running the app, manually download the trained model weights from Google Drive:  
🔗 **[Download brain_tumor_model.pth](https://drive.google.com/file/d/1qWyeTUFHzbaq1ELvTxrW46USQ1Ll_Zrs/view?usp=sharing)**

Place the downloaded `brain_tumor_model.pth` file in the root directory of the project like so:

```
brain_tumor_model.pth
```

---

### 4️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

- Upload an MRI brain scan (`.jpg`, `.png`)
- Click **🔍 Segment Tumor**
- View original, mask, and overlay results center-aligned and ready for PDF export

---

## 📂 Project Structure

```bash
.
├── app.py                        # Streamlit UI
├── brain-tumor-detection.ipynb  # Model training notebook
├── brain_tumor_model.pth        # [Download from Google Drive]
├── requirements.txt             # Dependencies
├── README.md                    # Project overview
└── Brain Tumor Segmentation.pdf # Project report
```

---

## 🙌 Acknowledgements

📊 Dataset by [Mateusz Buda on Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)  
⚙️ Frameworks: **PyTorch**, **Albumentations**, **timm**, **torchvision**  
💻 CUDA for GPU acceleration

---

## 📬 Contact

**Shreyak Mukherjee**  
📧 shreyakmukherjeedgp@gmail.com  
📍 Durgapur, West Bengal  
📱 +91-9832188947




