import copy
import numpy as np
import cv2
import pydicom
import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import densenet121
from PIL import Image
from pathlib import Path

st.title("Chest X-Ray Pneumonia Classification")
st.write("Upload a chest X-ray image to get a prediction.")
st.warning(
    "Educational/Research Use Only: This application is a machine-learning demonstration "
    "and is not a medical diagnostic device. Predictions should not be used to make clinical decisions."
)

st.caption("Do not upload real patient data or images that could identify a person.")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
FINAL_TEMPERATURE = 1.0842  
MODEL_PATH = Path(__file__).resolve().parent.parent / 'models' / 'densenet121_weighted_bestf1.pt'


def build_densenet121():
    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model


@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_densenet121().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input, output):
        self.activations = output
        self.activations.retain_grad()

    def generate(self, input_tensor):
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        output.backward()
        weights = self.activations.grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1).squeeze())
        cam = cv2.resize(cam.detach().cpu().numpy(), (224, 224))
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def load_image_as_array(uploaded_file):
    """Return a 2D float32 grayscale array from a DICOM or a jpg/png upload."""
    if uploaded_file.name.lower().endswith('.dcm'):
        arr = pydicom.dcmread(uploaded_file).pixel_array.astype(np.float32)
        if arr.max() > 255:  # rescale 12/16-bit DICOMs to 0-255
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
    else:
        arr = np.array(Image.open(uploaded_file).convert('L')).astype(np.float32)
    return arr


def predict(img_array, model, device):
    if img_array.ndim != 2 or min(img_array.shape) < 32:
        return {'error': f'Invalid image shape: {img_array.shape}'}

    img = cv2.resize(img_array, (224, 224))
    x = np.stack([img] * 3, axis=-1) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        p_pneumonia = torch.sigmoid(model(x) / FINAL_TEMPERATURE).item()

    cam_model = copy.deepcopy(model)
    cam = GradCAM(cam_model, cam_model.features.norm5).generate(x)

    heatmap = cv2.cvtColor(cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    orig_rgb = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(orig_rgb, 0.6, heatmap, 0.4, 0)

    return {'p_pneumonia': p_pneumonia, 'original': orig_rgb,
            'heatmap': heatmap, 'overlay': overlay, 'error': None}


model, device = load_model()

uploaded_file = st.file_uploader("Upload X-ray (JPG, PNG, or DICOM)", type=["jpg", "jpeg", "png", "dcm"])

if uploaded_file is not None:
    try:
        img_array = load_image_as_array(uploaded_file)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    st.image(cv2.resize(img_array, (448, 448)).astype(np.uint8), caption="Uploaded X-ray", clamp=True)

    if st.button("Analyze Image"):
        with st.spinner("Analyzing..."):
            result = predict(img_array, model, device)

        if result['error']:
            st.error(result['error'])
        else:
            p = result['p_pneumonia']
            label = 'Pneumonia' if p > 0.5 else 'Normal'
            confidence = p if p > 0.5 else 1 - p
            st.subheader(f"Prediction: {label}")
            st.write(f"Confidence: {confidence * 100:.1f}%  (pneumonia probability: {p * 100:.1f}%)")
            if 0.35 < p < 0.65:
                st.info("The model is uncertain on this image (probability close to 50%). Treat this result as inconclusive.")
            

            c1, c2, c3 = st.columns(3)
            c1.image(result['original'], caption="Original")
            c2.image(result['heatmap'], caption="Grad-CAM heatmap")
            c3.image(result['overlay'], caption="Overlay")