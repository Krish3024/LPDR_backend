# main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil
import uuid
from ultralytics import YOLO

app = FastAPI()

# ------------------- CORS -------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Allow any domain (you can restrict later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- Model Setup -------------------
MODEL_PATH = "best.pt"

# Download model from Google Drive if it doesn't exist
if not os.path.exists(MODEL_PATH):
    import gdown
    print("Downloading model...")
    gdown.download(
        "https://drive.google.com/uc?id=1rufCfEoeSlNkSE3V--1h2oopm2zNBade",
        MODEL_PATH,
        quiet=False
    )

# Load YOLO model
model = YOLO(MODEL_PATH)

# ------------------- API Route -------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded image temporarily
        temp_file = f"temp_{uuid.uuid4()}.jpg"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run detection
        results = model.predict(temp_file)
        result = results[0]

        names = result.names
        predictions = []

        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            predictions.append({
                "class": names[cls],
                "confidence": round(conf, 4)
            })

        # Clean up temp image
        os.remove(temp_file)

        return JSONResponse(content={"predictions": predictions})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ------------------- Run Locally (Optional) -------------------
# Uncomment the lines below if running locally
# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000)
