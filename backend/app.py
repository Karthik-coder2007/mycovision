import os, io, base64
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from model import ProposedMushroomModel

CLASSES = ["Agaricus","Amanita","Boletus","Cortinarius","Entoloma","Hygrocybe","Lactarius","Russula","Suillus"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = Path(os.getenv("CHECKPOINT_PATH", str(Path(__file__).parent / "proposed_mushroom_best.pth")))
app = FastAPI(title="Mushroom Classification API")
app.add_middleware(CORSMiddleware, allow_origins=[x.strip() for x in os.getenv("FRONTEND_ORIGIN","http://localhost:3000").split(",")], allow_methods=["*"], allow_headers=["*"])

transform = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
model, load_error = None, None
try:
    net = ProposedMushroomModel(len(CLASSES))
    try:
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(CHECKPOINT, map_location="cpu")
    net.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    net.to(DEVICE).eval()
    model = net
except Exception as exc:
    load_error = str(exc)

def normalize_map(arr):
    arr = np.asarray(arr, dtype=np.float32)
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    return arr


def overlay_gradcam(original_image, cam, alpha=0.40):
    # Resize the heatmap to match the original image
    original = original_image.convert("RGB")
    cam = Image.fromarray(
        (normalize_map(cam) * 255).astype(np.uint8)
    ).resize(original.size, Image.Resampling.BILINEAR)

    heat = np.asarray(cam, dtype=np.uint8)

    # Create a red-yellow-blue heatmap
    heat_rgb = np.stack([
        np.clip(heat * 2, 0, 255),
        np.clip(255 - np.abs(heat.astype(np.int16) - 128) * 2, 0, 255),
        np.clip(255 - heat * 2, 0, 255)
    ], axis=-1).astype(np.uint8)

    heat_image = Image.fromarray(heat_rgb)
    blended = Image.blend(original, heat_image, alpha)

    buffer = io.BytesIO()
    blended.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(
        buffer.getvalue()
    ).decode()


def overlay_spatial_mask(original_image, mask, alpha=0.35):
    original = original_image.convert("RGB")
    mask = Image.fromarray(
        (normalize_map(mask) * 255).astype(np.uint8)
    ).resize(original.size, Image.Resampling.BILINEAR)

    mask_array = np.asarray(mask, dtype=np.uint8)
    green_overlay = np.zeros(
        (mask_array.shape[0], mask_array.shape[1], 3),
        dtype=np.uint8
    )
    green_overlay[:, :, 1] = mask_array

    blended = Image.blend(original, Image.fromarray(green_overlay), alpha)

    buffer = io.BytesIO()
    blended.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(
        buffer.getvalue()
    ).decode()

@app.get("/")
def home():
    return {"message":"Mushroom Classification API"}

@app.get("/health")
def health():
    return {"status":"ok" if model else "model_not_loaded", "checkpoint_exists":CHECKPOINT.exists(), "error":load_error}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(503, f"Model is not loaded: {load_error}")
    if file.content_type not in {"image/jpeg","image/png","image/webp"}:
        raise HTTPException(400, "Upload a JPG, PNG or WEBP image.")
    try:
        raw = await file.read()
        if len(raw) > 10*1024*1024: raise HTTPException(413, "Image must be under 10 MB.")
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except HTTPException: raise
    except Exception: raise HTTPException(400, "Could not read the image.")
    x = transform(image).unsqueeze(0).to(DEVICE)
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        logits, mask = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(probs.argmax().item())
        logits[0,idx].backward()
        feats = model.last_features
        if feats is None or feats.grad is None: raise HTTPException(500, "Grad-CAM gradients unavailable.")
        weights = feats.grad.mean(dim=(2,3), keepdim=True)
        cam = torch.relu((weights*feats).sum(dim=1,keepdim=True))
        cam = F.interpolate(cam, size=(224,224), mode="bilinear", align_corners=False)[0,0].detach().cpu().numpy()
    mask_img = F.interpolate(mask.detach(), size=(224,224), mode="bilinear", align_corners=False)[0,0].cpu().numpy()
    model.zero_grad(set_to_none=True)
    return {
        "predicted_class":CLASSES[idx],
        "confidence":round(float(probs[idx].detach().cpu()),4),
        "probabilities":{CLASSES[i]:round(float(probs[i].detach().cpu()),4) for i in range(len(CLASSES))},
        "gradcam_png": overlay_gradcam(image, cam),
"spatial_mask_png": overlay_spatial_mask(image, mask_img),
        "note":"Grad-CAM is a visualization, not proof of causal reasoning. The learned mask amplifies features in this architecture; it does not explicitly remove background."
    }
