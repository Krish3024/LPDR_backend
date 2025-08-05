from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import cv2
import numpy as np
import re
import shutil
import torch
from ultralytics import YOLO
import easyocr

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the API"}

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update if you have a specific frontend domain
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO("best.pt")  # Ensure it's in your repo or accessible path
reader = easyocr.Reader(['en'], gpu=False)
latest_detected_text = ""

# ----------------- OCR Helper -----------------
def extract_significant_text(text_list):
    results = []
    for _, text, _ in text_list:
        cleaned_parts = re.findall(r'[A-Z0-9]+', text.upper())
        results.extend(cleaned_parts)

    full_text = "".join(results)
    full_text = re.sub(r'^(IND)', '', full_text, flags=re.IGNORECASE)
    full_text = re.sub(r'(IND)$', '', full_text, flags=re.IGNORECASE)

    pattern = r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}'
    match = re.search(pattern, full_text)
    return match.group(0) if match else full_text

# -------------- Detection Logic ----------------
def detect_license_plate_from_frame(image):
    global latest_detected_text
    results = model(image)[0]
    extracted_texts = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cropped_plate = image[y1:y2, x1:x2]
        if cropped_plate is None or cropped_plate.size == 0:
            continue

        gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
        ocr_results = reader.readtext(gray)
        text = extract_significant_text(ocr_results)

        if text:
            extracted_texts.append(text)
            cv2.putText(image, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    final_texts = list(set(extracted_texts))
    latest_detected_text = " ".join(final_texts) if final_texts else "No valid text found"
    return image, latest_detected_text

# -------------- Endpoints ----------------------

@app.post("/detect")
async def detect_license_plate(image: UploadFile = File(...)):
    try:
        # Save uploaded image
        image_path = os.path.join(UPLOAD_FOLDER, image.filename)
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        img = cv2.imread(image_path)
        if img is None:
            return JSONResponse({"error": "Could not read image"}, status_code=500)

        processed_image, detected_text = detect_license_plate_from_frame(img)

        # Save processed image
        output_path = os.path.join(OUTPUT_FOLDER, f"output_{image.filename}")
        cv2.imwrite(output_path, processed_image)

        return JSONResponse({
            "ocr_text": detected_text,
            "image_url": f"/output/{image.filename}"
        })
    except Exception as e:
        return JSONResponse({"error": f"Failed to process image: {str(e)}"}, status_code=500)

@app.get("/output/{filename}")
def get_output_image(filename: str):
    filepath = os.path.join(OUTPUT_FOLDER, f"output_{filename}")
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="image/jpeg")
    return JSONResponse({"error": "File not found"}, status_code=404)

@app.get("/latest_text")
def get_latest_text():
    return {"detected_text": latest_detected_text}

@app.post("/clear_text")
def clear_detected_text():
    global latest_detected_text
    latest_detected_text = ""
    return {"message": "Detected text cleared."}

# ----------------- Run Locally ------------------
# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
