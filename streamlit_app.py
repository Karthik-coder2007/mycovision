
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="MycoVision - Mushroom Classification",
    page_icon="🍄",
    layout="wide",
)

st.title("🍄 MycoVision")
st.subheader("AI-powered Mushroom Classification")
st.write(
    "Upload a mushroom image to predict its class, "
    "view model confidence, and inspect Grad-CAM."
)

# --------------------------------------------------
# 2. Device and class labels
# --------------------------------------------------
DEVICE = torch.device("cpu")

CLASSES = [
    "Agaricus",
    "Amanita",
    "Boletus",
    "Cortinarius",
    "Entoloma",
    "Hygrocybe",
    "Lactarius",
    "Russula",
    "Suillus",
]

# --------------------------------------------------
# 3. Import your existing model architecture
# --------------------------------------------------
BACKEND_DIR = Path(__file__).parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

try:
    from model import ProposedMushroomModel
except Exception as exc:
    st.error(f"Could not import backend/model.py: {exc}")
    st.stop()


# --------------------------------------------------
# 4. Load trained model from Hugging Face
# --------------------------------------------------
@st.cache_resource
def load_model():
    token = st.secrets.get("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HF_TOKEN is missing. Add it under "
            "Streamlit Cloud → App settings → Secrets."
        )

    checkpoint_path = hf_hub_download(
        repo_id="kar343/mycovision-mushroom-model",
        filename="proposed_mushroom_best.pth",
        token=token,
    )

    model = ProposedMushroomModel(num_classes=len(CLASSES))

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    return model


# --------------------------------------------------
# 5. Image preprocessing
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# --------------------------------------------------
# 6. Grad-CAM calculation
# --------------------------------------------------
def generate_gradcam(model, image_tensor, class_index):
    """
    Generate Grad-CAM from the last convolutional layer.

    This requires the selected layer to produce spatial
    feature maps and the predicted logit to depend on them.
    """
    target_layer = None

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module

    if target_layer is None:
        raise RuntimeError("No Conv2d layer was found in the model.")

    saved = {}

    def forward_hook(module, inputs, output):
        saved["activation"] = output

        if not output.requires_grad:
            raise RuntimeError(
                "The selected convolution output does not track gradients."
            )

        output.register_hook(
            lambda gradient: saved.__setitem__("gradient", gradient)
        )

    handle = target_layer.register_forward_hook(forward_hook)

    try:
        model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            output = model(image_tensor)

            # Support models that return logits or (logits, extra_output).
            logits = (
                output[0]
                if isinstance(output, (tuple, list))
                else output
            )

            if logits.ndim != 2 or logits.shape[1] != len(CLASSES):
                raise RuntimeError(
                    "Unexpected model output shape: "
                    f"{tuple(logits.shape)}. Expected batch x 9 classes."
                )

            score = logits[0, class_index]
            score.backward()

        if "gradient" not in saved or "activation" not in saved:
            raise RuntimeError(
                "Grad-CAM could not capture activations and gradients."
            )

        activations = saved["activation"].detach()
        gradients = saved["gradient"].detach()

        # Average gradients across spatial dimensions.
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of feature maps.
        cam = torch.relu(
            (weights * activations).sum(dim=1, keepdim=True)
        )

        # Resize heatmap to the original image dimensions.
        cam = F.interpolate(
            cam,
            size=(image_tensor.shape[2], image_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


# --------------------------------------------------
# 7. Convert heatmap to an overlay
# --------------------------------------------------
def create_gradcam_overlay(original_image, cam):
    original_image = original_image.convert("RGB")

    cam_image = Image.fromarray(
        np.uint8(np.clip(cam, 0, 1) * 255)
    ).resize(original_image.size)

    # Create a red-yellow heatmap.
    heat = np.asarray(cam_image, dtype=np.float32) / 255.0

    red = np.clip(heat * 2.0, 0, 1)
    green = np.clip((heat - 0.25) * 1.6, 0, 1)
    blue = np.zeros_like(heat)

    heat_rgb = np.stack([red, green, blue], axis=-1)
    heat_rgb = Image.fromarray(
        np.uint8(np.clip(heat_rgb, 0, 1) * 255)
    )

    overlay = Image.blend(original_image, heat_rgb, alpha=0.45)

    return heat_rgb, overlay


# --------------------------------------------------
# 8. Upload image
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a mushroom image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    try:
        original_image = Image.open(uploaded_file).convert("RGB")

        st.image(
            original_image,
            caption="Uploaded mushroom image",
            use_container_width=True,
        )

        if st.button("Classify Mushroom", type="primary"):
            try:
                with st.spinner(
                    "Loading model and generating prediction..."
                ):
                    model = load_model()

                    image_tensor = transform(original_image).unsqueeze(0)
                    image_tensor = image_tensor.to(DEVICE)

                    # Prediction.
                    with torch.no_grad():
                        output = model(image_tensor)
                        logits = (
                            output[0]
                            if isinstance(output, (tuple, list))
                            else output
                        )

                        probabilities = torch.softmax(logits, dim=1)[0]
                        predicted_index = int(
                            probabilities.argmax().item()
                        )

                    # Grad-CAM needs gradients, so run it separately.
                    cam = generate_gradcam(
                        model,
                        image_tensor,
                        predicted_index,
                    )

                    heatmap, overlay = create_gradcam_overlay(
                        original_image,
                        cam,
                    )

                # --------------------------------------
                # 9. Display prediction
                # --------------------------------------
                st.success(
                    f"Predicted class: {CLASSES[predicted_index]}"
                )

                st.metric(
                    "Model confidence",
                    f"{probabilities[predicted_index].item():.2%}",
                )

                # --------------------------------------
                # 10. Display Grad-CAM
                # --------------------------------------
                st.header("Grad-CAM Visualization")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.image(
                        original_image,
                        caption="Original image",
                        use_container_width=True,
                    )

                with col2:
                    st.image(
                        heatmap,
                        caption="Grad-CAM heatmap",
                        use_container_width=True,
                    )

                with col3:
                    st.image(
                        overlay,
                        caption="Grad-CAM overlay",
                        use_container_width=True,
                    )

                st.caption(
                    "Red and yellow regions indicate areas that "
                    "contributed more strongly to the selected class "
                    "under this Grad-CAM calculation."
                )

                # --------------------------------------
                # 11. Display class probabilities
                # --------------------------------------
                st.header("Class Probabilities")

                probability_data = {
                    CLASSES[i]: float(probabilities[i].item())
                    for i in range(len(CLASSES))
                }

                st.bar_chart(probability_data)

                with st.expander("View exact probabilities"):
                    for class_name, probability in probability_data.items():
                        st.write(
                            f"**{class_name}:** {probability:.2%}"
                        )

                st.warning(
                    "Research demonstration only. Grad-CAM is an "
                    "explanation aid, not proof of correctness. Never "
                    "use this classifier to decide whether a wild "
                    "mushroom is safe to eat."
                )

            except Exception as exc:
                st.error(f"Prediction or Grad-CAM failed: {exc}")
                st.info(
                    "Check the app logs if this error persists. "
                    "The model architecture and checkpoint must match."
                )

    except Exception as exc:
        st.error(f"Could not read the uploaded image: {exc}")

else:
    st.info("Upload an image to get started.")

st.divider()
st.caption("MycoVision | Experimental mushroom image classification")
