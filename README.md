
<h1 align="center">🧠 Brain Tumor Segmentation using Diffusion-Augmented Swin Transformer</h1>



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

<h2>⚙️ Training Overview &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 📊 Evaluation Metrics</h2>
<div align="center">
<table>
  <tr>
    <td>

<!-- Left Table -->
<b>⚙️ Training Overview</b>

<table border="1" cellpadding="6" cellspacing="0">
  <tr><th>Component</th><th>Details</th></tr>
  <tr><td>Loss Function</td><td>BCE + Dice hybrid</td></tr>
  <tr><td>Optimizer</td><td>AdamW (lr = 1e-4, weight decay = 1e-4)</td></tr>
  <tr><td>Scheduler</td><td>OneCycleLR</td></tr>
  <tr><td>Epochs</td><td>50 (with early stopping)</td></tr>
  <tr><td>Mixed Precision</td><td><code>autocast</code> + <code>GradScaler</code></td></tr>
  <tr><td>Device</td><td>CUDA-enabled GPU</td></tr>
</table>

  </td>
    <td style="width: 40px;"></td>
  <td>

<!-- Right Table -->
<b>📊 Evaluation Metrics</b>

<table border="1" cellpadding="6" cellspacing="0">
  <tr><th>Metric</th><th>Score</th></tr>
  <tr><td>🎯 Dice</td><td><b>0.92</b></td></tr>
  <tr><td>📏 IoU</td><td><b>0.86</b></td></tr>
  <tr><td>🎯 Precision</td><td><b>0.89</b></td></tr>
  <tr><td>🔁 Recall</td><td><b>0.94</b></td></tr>
  <tr><td>🧮 Accuracy</td><td><b>0.98</b></td></tr>
</table>

  </td>
  </tr>
</table>
</div>


🧾 **Output Images**

---

### 🧠 Brain MRI Output 1
<div align="center">
  <img src="https://raw.githubusercontent.com/shreyakmukherjee/brain-tumor-segmentation-diffusion-swin/main/Images/Output_image_1.png" alt="Tumor Output 1" width="90%" style="margin: 15px auto; object-fit: contain;" />
</div>

---

### 🧠 Brain MRI Output 2
<div align="center">
  <img src="https://raw.githubusercontent.com/shreyakmukherjee/brain-tumor-segmentation-diffusion-swin/main/Images/Output_image_2.png" alt="Tumor Output 2" width="90%" style="margin: 15px auto; object-fit: contain;" />
</div>

---

### 📊 Confusion Matrix & Hitmap
<div align="center">
  <img src="https://github.com/shreyakmukherjee/brain-tumor-segmentation-diffusion-swin/blob/main/Images/Confusion_Matrix.png?raw=true" alt="Confusion Matrix" width="45%" style="margin: 15px; object-fit: contain;" />
  <img src="https://github.com/shreyakmukherjee/brain-tumor-segmentation-diffusion-swin/blob/main/Images/Hitmap.png?raw=true" alt="Hitmap" width="45%" style="margin: 15px; object-fit: contain;" />
</div>

---

### 📈 Dice Coefficient & Loss Curve
<div align="center">
  <img src="https://github.com/shreyakmukherjee/brain-tumor-segmentation-diffusion-swin/blob/main/Images/Dice_Coefficient.png?raw=true" alt="Dice Coefficient" width="45%" style="margin: 15px; object-fit: contain;" />
  <img src="https://github.com/shreyakmukherjee/brain-tumor-segmentation-diffusion-swin/blob/main/Images/Loss_curve.png?raw=true" alt="Loss Curve" width="45%" style="margin: 15px; object-fit: contain;" />
</div>

---

### 📉 ROC & Recall Curves
<div align="center">
  <img src="https://github.com/shreyakmukherjee/brain-tumor-segmentation-diffusion-swin/blob/main/Images/ROC_curve.png?raw=true" alt="ROC Curve" width="45%" style="margin: 15px; object-fit: contain;" />
  <img src="https://github.com/shreyakmukherjee/brain-tumor-segmentation-diffusion-swin/blob/main/Images/Recall_curve.png?raw=true" alt="Recall Curve" width="45%" style="margin: 15px; object-fit: contain;" />
</div>


---

## 🌟 Innovations


<div align="center">

| Component              | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| Diffusion Module       | Enhances low-contrast features by simulating noise removal and reconstruction. |
| Swin Transformer       | Provides attention across spatial regions for better boundary detection.     |
| Hybrid Architecture    | Combines CNN speed with transformer precision for medical segmentation.      |
| Gradient Checkpointing | Reduces memory usage to allow larger batch sizes during training.            |

</div>

---

## 🧪 Applications in Healthcare

- 🧠 Accurate tumor volume estimation  
- 🩺 Pre-operative surgical planning  
- 📈 Monitoring treatment response  
- 📆 Longitudinal study support  

---

## ⚠️ Limitations & Future Work

<div align="center">

| Current Challenge                      | Future Direction                                                        |
|----------------------------------------|--------------------------------------------------------------------------|
| Requires high-quality MRI inputs       | Integrate super-resolution preprocessing or denoising techniques         |
| 2D slice-wise segmentation only        | Extend to full 3D volumetric segmentation                                |
| Focused on a single modality (T1)      | Expand to multi-modal fusion (T1, T2, FLAIR)                             |
| High compute demand during training    | Apply model pruning or distillation for edge deployment                  |

</div>

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




