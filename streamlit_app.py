import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from collections import OrderedDict
import os

st.set_page_config(page_title="Brain Tumor Model — Streamlit Demo", layout="centered")

st.title("🧠 Brain Tumor Detection Demo")
st.markdown("Upload an MRI image to classify whether it shows a **Tumor** or **No Tumor**.")

# ---------------------
# 🔧 Model definition
# ---------------------
def create_model():
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # 2 classes: Tumor / No Tumor
    return model

# ---------------------
# 🔧 Model loader
# ---------------------
def load_model(path: str, device: str = "cpu", model_class=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Model file not found: {path}")

    map_location = torch.device(device)
    loaded = torch.load(path, map_location=map_location)

    # Case 1: Directly a full model
    if isinstance(loaded, nn.Module):
        st.info("✅ Loaded full model object directly.")
        return loaded.to(map_location)

    # Case 2: state_dict or checkpoint
    if isinstance(loaded, dict):
        for key in ("model_state_dict", "state_dict", "model", "net", "model_state"):
            if key in loaded and isinstance(loaded[key], dict):
                state_dict = loaded[key]
                break
        else:
            state_dict = loaded

        # 🩺 Fix prefix "resnet."
        new_state_dict = OrderedDict()
        prefix_fixed = False
        for k, v in state_dict.items():
            if k.startswith("resnet."):
                new_state_dict[k.replace("resnet.", "")] = v
                prefix_fixed = True
            else:
                new_state_dict[k] = v
        if prefix_fixed:
            st.warning("🧩 Detected and removed 'resnet.' prefix from state_dict keys.")

        # Build model class if provided
        if model_class is not None:
            model = model_class()
            model.load_state_dict(new_state_dict, strict=False)
            model.to(map_location)
            st.success("✅ Model loaded successfully.")
            return model

        st.error("⚠️ This is a state_dict. Provide model_class to rebuild the model.")
        return None

    st.error("⚠️ Unknown model format. Expected .pth or state_dict.")
    return None

# ---------------------
# ⚙️ Load model section
# ---------------------
model_path = st.text_input("**Model path:**", value=r"D:\mri\best_brain_tumor_model.pth")
device = "cpu"

if st.button("Load Model"):
    try:
        model = load_model(model_path, device=device, model_class=create_model)
        if model:
            st.session_state["model"] = model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")

# ---------------------
# 🧾 Image upload & inference
# ---------------------
if "model" in st.session_state:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded MRI Image", use_container_width=True)

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        input_tensor = preprocess(image).unsqueeze(0)

        model = st.session_state["model"]
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()

        labels = ["No Tumor", "Tumor"]
        st.markdown("### 🩺 Prediction Result:")
        st.success(f"**{labels[pred_idx]}** (Confidence: {confidence*100:.2f}%)")

else:
    st.info("👆 Enter your model path and click 'Load Model' to begin.")
