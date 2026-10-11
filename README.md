# Mushroom AI — Vercel frontend + Python API

This project uses a Next.js frontend deployed to Vercel and a separate FastAPI/PyTorch backend. Vercel does not natively host a persistent Streamlit server.

## Backend setup (VS Code)
1. Copy your trained checkpoint to `backend/proposed_mushroom_best.pth`.
2. In a terminal:
   ```
   cd backend
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   $env:CHECKPOINT_PATH="proposed_mushroom_best.pth"
   $env:FRONTEND_ORIGIN="http://localhost:3000"
   uvicorn app:app --reload --port 8000
   ```
3. Check `http://127.0.0.1:8000/health`.

## Frontend setup
1. In a second terminal:
   ```
   cd frontend
   npm install
   ```
2. Create `frontend/.env.local` containing:
   `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000`
3. Run `npm run dev`, then open `http://localhost:3000`.

## Deploy
Deploy `frontend/` to Vercel and set `NEXT_PUBLIC_API_URL` to the public URL of your separately hosted Python API. Deploy the backend to a Python-capable host, set `CHECKPOINT_PATH` and `FRONTEND_ORIGIN`, and make sure the host can install PyTorch/timm and access your checkpoint.

The model code must match the architecture used when the checkpoint was trained. If loading fails with missing keys or size mismatches, use the exact training-time model class. The example preprocessing is Resize(224,224) + ImageNet normalization; match your training transforms.

The learned mask is returned by the model. The current starter returns a Grad-CAM heatmap too. Neither visualization proves that the model is correct. This is not a food-safety tool; never eat a mushroom based on an AI prediction.
