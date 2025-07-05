from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import os
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# API key setup
os.environ["HF_HOME"] = "./hf_cache"
API_KEY = os.getenv("NSFW_API_KEY")

if not API_KEY:
    logger.error("API key not set. Please set NSFW_API_KEY in the environment.")
    raise RuntimeError("API key not set in environment variable: NSFW_API_KEY")

# FastAPI app initialization
app = FastAPI(
    title="NSFW Detection API",
    description="Detects NSFW content using Hugging Face model",
    version="1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"]
)

# API key validation dependency
def verify_api_key(request: Request):
    client_key = request.headers.get("X-API-Key")
    if client_key != API_KEY:
        logger.warning("Unauthorized request received with invalid or missing API key")
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API Key")


# Model loading
try:
    classifier = pipeline(
        "image-classification",
        model="Falconsai/nsfw_image_detection",
        feature_extractor="Falconsai/nsfw_image_detection",
        use_fast=True
    )
    logger.info("NSFW model loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    raise RuntimeError("Model loading failed")

# Constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB

# Routes
@app.get("/", dependencies=[Depends(verify_api_key)])
def root():
    return {
        "message": "NSFW Detection API is running",
        "docs": "/docs",
        "predict_endpoint": "/predict"
    }

@app.get("/health", dependencies=[Depends(verify_api_key)])
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "max_file_size": f"{MAX_FILE_SIZE / (1024 * 1024):.0f}MB",
        "supported_formats": list(ALLOWED_EXTENSIONS)
    }

@app.get("/version", dependencies=[Depends(verify_api_key)])
def version():
    return {
        "app": "nsfw-detector",
        "version": "1.0",
        "model": "Falconsai/nsfw_image_detection"
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    _=Depends(verify_api_key)
):
    filename = file.filename
    ext = filename.rsplit('.', 1)[-1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}")

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image format or corrupted image")

    try:
        result = classifier(image)
        return {
            "filename": filename,
            "result": result
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process image")
