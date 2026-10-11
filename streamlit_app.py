
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from huggingface_hub import hf_hub_download
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))
from model import ProposedMushroomModel

st.set_page_config(page_title="MycoVision", page_icon="🍄")
st.title("🍄 MycoVision")
st.write("AI-powered mushroom image classification")

CLASSES = [
    "Agaricus", "Amanita", "Boletus", "Cortinarius",
    "Entoloma", "Hygrocybe", "Lactarius", "Russula", "Suillus"
]

@st.cache_resource
def load_model():
    token = st.secrets["HF_TOKEN"]
    checkpoint = hf_hub_download(
        repo_id="kar343/mycovision-mushroom-model",
        filename="proposed_mushroom_best.pth",
        token=token
    )
    model = ProposedMushroomModel(num_classes=len(CLASSES))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    return model

uploaded = st.file_uploader(
    "Upload a mushroom image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded mushroom", use_container_width=True)

    if st.button("Classify mushroom"):
        try:
            with st.spinner("Loading model and predicting..."):
                model = load_model()
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]
                    )
                ])
                x = transform(image).unsqueeze(0)

                with torch.no_grad():
                    logits, _ = model(x)
                    probs = torch.softmax(logits, dim=1)[0]

                idx = int(probs.argmax())
                st.success(f"Predicted class: {CLASSES[idx]}")
                st.metric("Model confidence", f"{probs[idx].item():.1%}")
                st.subheader("Class probabilities")
                st.bar_chart({
                    "Probability": {
                        CLASSES[i]: float(probs[i])
                        for i in range(len(CLASSES))
                    }
                })
                st.warning(
                    "This is an experimental image classifier. "
                    "Do not use its prediction to decide whether a "
                    "wild mushroom is safe to eat."
                )
        except Exception as e:
            st.error(f"Prediction failed: {e}")

