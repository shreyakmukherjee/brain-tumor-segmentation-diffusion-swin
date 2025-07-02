import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
from timm.models.swin_transformer import SwinTransformer
from torchvision.models import mobilenet_v3_small

# -------------------- Configuration --------------------
IMG_SIZE = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "brain_tumor_model.pth"

# -------------------- Diffusion Model --------------------
class DiffusionModel(nn.Module):
    def __init__(self, in_channels, num_timesteps=100):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.betas = self.linear_beta_schedule(num_timesteps).to(DEVICE)
        self.alphas = 1. - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1)
        )

    def linear_beta_schedule(self, timesteps):
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

    def forward(self, x, t):
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod[t])[:, None, None, None]
        noise = torch.randn_like(x)
        noisy_x = sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * noise
        predicted_noise = self.model(noisy_x)
        return predicted_noise, noise

# -------------------- Main Model --------------------
class BrainTumorSegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = mobilenet_v3_small(pretrained=True)
        self.feature_extractor.classifier = nn.Identity()
        self.feature_adapt = nn.Sequential(
            nn.Conv2d(576, 96, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Upsample(size=(64, 64), mode='bilinear')
        )
        self.diffusion = DiffusionModel(96, num_timesteps=100)
        self.swin = SwinTransformer(
            img_size=64,
            patch_size=4,
            in_chans=96,
            num_classes=1,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=8,
            drop_rate=0.1
        )
        self.channel_adjust = nn.Conv2d(768, 96, kernel_size=1)
        self.seg_head = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Upsample(size=(IMG_SIZE, IMG_SIZE), mode='bilinear'),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward_features(self, x):
        x = self.feature_extractor.features(x)
        x = self.feature_adapt(x)
        return x

    def forward(self, x, t=None):
        features = self.forward_features(x)
        if t is None:
            t = torch.randint(0, 100, (x.size(0),), device=DEVICE)
        predicted_noise, _ = self.diffusion(features, t)
        refined_features = features - predicted_noise
        swin_out = self.swin.forward_features(refined_features)
        swin_out = swin_out.permute(0, 3, 1, 2)
        swin_out = self.channel_adjust(swin_out)
        swin_out = nn.functional.interpolate(swin_out, size=refined_features.shape[2:], mode='bilinear', align_corners=False)
        seg = self.seg_head(swin_out)
        return seg

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    model = BrainTumorSegmentationModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

# -------------------- Preprocess Image --------------------
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img_resized = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    transform = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    img_tensor = transform(img_tensor)
    return img_tensor.unsqueeze(0), image

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Brain Tumor Segmentation", layout="centered", initial_sidebar_state="collapsed")

# ✨ Custom Light Theme UI
# ✨ Custom Light Theme UI with Styled File Uploader
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #e0f7fa, #f0f4c3);  /* Light cyan to pale yellow */
        color: #1e293b;  /* Slate gray text */
    }
    .block-container {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        margin-top: 2rem;
    }
    h1 {
        color: #0ea5e9;  /* Sky blue */
        font-weight: 800;
        text-align: center;
        font-size: 2.5rem;
    }
    .stButton>button {
        background-color: #38bdf8;
        color: #1e293b;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-weight: 600;
        border: none;
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0ea5e9;
        transform: scale(1.05);
    }
    .stFileUploader label {
        color: #1e293b;
        font-weight: 600;
        font-size: 1rem;
    }
    section[data-testid="stFileUploader"] > div {
        background-color: #e0f2fe !important;  /* light sky blue background */
        border: 2px dashed #38bdf8 !important; /* cyan border */
        border-radius: 10px;
        padding: 1rem;
        transition: 0.3s ease;
    }
    section[data-testid="stFileUploader"] button {
        background-color: #38bdf8 !important;
        color: #0f172a !important;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.4rem 1rem;
        border: none;
    }
    section[data-testid="stFileUploader"] button:hover {
        background-color: #0ea5e9 !important;
        transform: scale(1.05);
    }
    .stSpinner {
        color: #1e293b;
        font-size: 1.1rem;
    }
    .stSubheader {
        color: #1e293b;
    }
    </style>
""", unsafe_allow_html=True)


# -------------------- App Logic --------------------
import base64
from io import BytesIO

def center_image(image, caption, width=400):
    """Display an image centered using HTML + base64."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    st.markdown(f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{img_str}" width="{width}">
            <p style="font-size: 0.9rem; color: #334155;"><i>{caption}</i></p>
        </div>
    """, unsafe_allow_html=True)

st.title("🧠 TumorScope: MRI Brain Tumor Segmentation")
st.write("Upload an MRI scan to visualize the segmented tumor region using a deep learning model with Swin Transformer and Diffusion guidance.")

uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    center_image(Image.open(uploaded_file), "🧾 Uploaded MRI")

    img_tensor, orig_pil = preprocess_image(uploaded_file)

    if st.button("🔍 Segment Tumor"):
        with st.spinner("Running segmentation..."):
            with torch.no_grad():
                output = model(img_tensor.to(DEVICE))[0][0].cpu().numpy()
                mask = (output > 0.5).astype(np.uint8) * 255
                overlay = np.array(orig_pil.resize((IMG_SIZE, IMG_SIZE))).copy()
                overlay[mask > 0] = [255, 0, 0]  # Red overlay

        st.subheader("🖼️ Segmentation Result")

        center_image(overlay, "🧠 Tumor Region Highlighted")
        center_image(mask, "🧪 Predicted Tumor Mask")

        # Add spacing to avoid pushing last lines to page 3 in PDF
        st.markdown("<br><br>", unsafe_allow_html=True)


# # -------------------- App Logic --------------------
# st.title("🧠 TumorScope: MRI Brain Tumor Segmentation")
# st.write("Upload an MRI scan to visualize the segmented tumor region using a deep learning model with Swin Transformer and Diffusion guidance.")

# uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     st.image(uploaded_file, caption="🧾 Uploaded MRI", use_container_width=True)
#     img_tensor, orig_pil = preprocess_image(uploaded_file)

#     if st.button("🔍 Segment Tumor"):
#         with st.spinner("Running segmentation..."):
#             with torch.no_grad():
#                 output = model(img_tensor.to(DEVICE))[0][0].cpu().numpy()
#                 mask = (output > 0.5).astype(np.uint8) * 255
#                 overlay = np.array(orig_pil.resize((IMG_SIZE, IMG_SIZE))).copy()
#                 overlay[mask > 0] = [255, 0, 0]  # Red overlay

#         st.subheader("🖼️ Segmentation Result")
#         st.image(overlay, caption="🧠 Tumor Region Highlighted", use_container_width=True)
#         st.image(mask, caption="🧪 Predicted Tumor Mask", use_container_width=True)
