import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pathlib import Path
from PIL import Image
import io

from model import build_model, load_model, get_device
from inference import preprocess_image, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    checkpoint_path = Path("../shared/models/best_checkpoint.pt")
    if not checkpoint_path.exists():
        raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")
    model, _ = load_model(checkpoint_path, device)
    print("Model loaded.")
    yield


app = FastAPI(
    title="Chest X-Ray Classifier",
    description="Efficient-B3 peumonia detection inference service",
    version="1.0.0",
    lifespan=lifespan
)

device = get_device()
model = None

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return JSONResponse(content={
        "status": "healthy",
        "model_leaded": model is not None
    })

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Accept an X-ray image ad return a classifiction result.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be in image"
        )
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = preprocess_image(image)
        result = predict(model, tensor, device)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )