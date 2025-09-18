# app.py
import io
import os
from typing import Tuple
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import streamlit as st
from UNet import UNet

# -------------------------
@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str = "models/best_model-3.pth", device_name: str = None):
    device = torch.device(device_name) if device_name else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    model = UNet().to(device)
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        # detect if state_dict or full model
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        elif isinstance(state, dict):
            model.load_state_dict(state)
        else:
            model = state  # full model
            model.to(device)
    else:
        st.error(f"Checkpoint not found at {checkpoint_path}")
    model.eval()
    return model, device


# -------------------------
def predict_image(
    model: nn.Module,
    device: torch.device,
    image_pil: Image.Image,
    input_size: Tuple[int, int] = (512, 512),
    use_autocast: bool = True,
) -> Image.Image:
    """
    Takes a PIL image, returns PIL corrected image.
    """
    original_size = image_pil.size  # (W, H)

    transform = T.Compose([
        T.Resize(input_size),
        T.ToTensor(),
    ])

    x = transform(image_pil).unsqueeze(0).to(device)  # [1,3,H,W]

    with torch.no_grad():
        if device.type == "cuda" and use_autocast:
            with torch.cuda.amp.autocast():
                out = model(x)
        else:
            out = model(x)

    out_img = out.squeeze(0).cpu().clamp(0, 1)
    pil_out = T.ToPILImage()(out_img)

    # resize back to original image size
    pil_out = pil_out.resize(original_size, Image.BILINEAR)
    return pil_out


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Color Correction - U-Net", layout="centered")
st.title("🎨 Color Correction Demo (U-Net)")
st.markdown("Upload a RAW image and the model will return a color-corrected version.")

# ================= Model Input Settings =================
st.sidebar.header("Model / Inference settings")
checkpoint_path = st.sidebar.text_input("Checkpoint path", value="models/best_model.pth")

use_gpu = st.sidebar.checkbox("Use GPU (if available)", value=True)
device_name = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"

input_size = st.sidebar.selectbox("Inference size", options=[128, 256, 512], index=2)
input_size = (input_size, input_size)

use_autocast = st.sidebar.checkbox("Use autocast (mixed precision) on GPU", value=True)

# Load model (cached)
with st.spinner("Loading model..."):
    model, device = load_model(checkpoint_path=checkpoint_path, device_name=device_name)

# ================= File Uploader =================
uploaded_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")

    if st.button("Run Color Correction"):
        with st.spinner("Running inference..."):
            try:
                result_img = predict_image(model, device, image, input_size=input_size, use_autocast=use_autocast)

                # Show Input vs Corrected Side by Side
                st.subheader("Comparison")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Input (Raw)", use_column_width=True)
                with col2:
                    st.image(result_img, caption="Predicted (Corrected)", use_column_width=True)

                # provide download
                buf = io.BytesIO()
                result_img.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button("Download corrected image (PNG)", data=byte_im, file_name="corrected.png", mime="image/png")

            except Exception as e:
                st.error(f"Inference failed: {e}")

# Footer
st.markdown("---")
st.markdown("✅ Ensure `checkpoint_path` matches your saved `.pth` model.\n"
            "If you trained with a different input size, select it in the sidebar.")
st.markdown("Developed by Ahmed Balta.")
