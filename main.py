# # from flask import Flask, request, jsonify, send_file, Response
# # from flask_cors import CORS
# # import cv2
# # import numpy as np
# # import pytesseract
# # import torch
# # from ultralytics import YOLO
# # import os
# # import re

# # app = Flask(__name__)
# # CORS(app)

# # latest_detected_text = ""

# # # Set path to Tesseract (Windows only)
# # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# # # Load YOLOv8 model
# # model = YOLO("best.pt")  # Replace with your trained model path

# # UPLOAD_FOLDER = "uploads"
# # OUTPUT_FOLDER = "outputs"
# # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# # def extract_significant_text(ocr_text):
# #     """Extracts meaningful alphanumeric sequences from OCR output."""
# #     words = re.findall(r'[A-Z0-9]{5,}', ocr_text)  # Extracts alphanumeric strings of at least 5 characters
# #     return " ".join(words)

# # # def detect_license_plate_from_frame(frame):
# # #     """Detects number plates and extracts text from a video frame."""
# # #     results = model(frame)[0]
# # #     extracted_texts = []
    
# # #     for box in results.boxes:
# # #         x1, y1, x2, y2 = map(int, box.xyxy[0])
# # #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
# # #         cropped_plate = frame[y1:y2, x1:x2]
# # #         if cropped_plate is None or cropped_plate.size == 0:
# # #             continue

# # #         # Convert to grayscale for OCR
# # #         gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)

# # #         # Apply OCR with updated config
# # #         raw_text = pytesseract.image_to_string(gray, config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789').strip()
# # #         significant_text = extract_significant_text(raw_text)

# # #         if significant_text:
# # #             extracted_texts.append(significant_text)
# # #             cv2.putText(frame, significant_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# # #     return frame, " ".join(extracted_texts) if extracted_texts else "No valid text found"

# # def detect_license_plate_from_frame(frame):
# #     global latest_detected_text  # Use the global variable
# #     results = model(frame)[0]
# #     extracted_texts = []
    
# #     for box in results.boxes:
# #         x1, y1, x2, y2 = map(int, box.xyxy[0])
# #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

# #         cropped_plate = frame[y1:y2, x1:x2]
# #         if cropped_plate is None or cropped_plate.size == 0:
# #             continue

# #         gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
# #         raw_text = pytesseract.image_to_string(gray, config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789').strip()
# #         significant_text = extract_significant_text(raw_text)

# #         if significant_text:
# #             extracted_texts.append(significant_text)
# #             cv2.putText(frame, significant_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# #     latest_detected_text = " ".join(extracted_texts) if extracted_texts else "No valid text found"
# #     return frame, latest_detected_text


# # @app.route('/detect', methods=['POST'])
# # def detect_license_plate():
# #     if 'image' not in request.files:
# #         return jsonify({"error": "No image uploaded"}), 400

# #     file = request.files['image']
# #     image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    
# #     try:
# #         file.save(image_path)
# #         image = cv2.imread(image_path)
# #         if image is None:
# #             return jsonify({"error": "Could not read image"}), 500

# #         processed_image, detected_text = detect_license_plate_from_frame(image)

# #         # Save processed image
# #         output_path = os.path.join(OUTPUT_FOLDER, f"output_{file.filename}")
# #         cv2.imwrite(output_path, processed_image)

# #         return jsonify({
# #             "ocr_text": detected_text,
# #             "image_url": f"http://127.0.0.1:5000/output/{file.filename}"
# #         })

# #     except Exception as e:
# #         return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

# # @app.route('/output/<filename>')
# # def get_output_image(filename):
# #     return send_file(os.path.join(OUTPUT_FOLDER, f"output_{filename}"), mimetype='image/jpeg')

# # # def generate_frames():
# # #     cap = cv2.VideoCapture(0)
# # #     while True:
# # #         success, frame = cap.read()
# # #         if not success:
# # #             break
# # #         processed_frame, detected_text = detect_license_plate_from_frame(frame)
        
# # #         # Store the latest detected text globally
# # #         latest_detected_text = detected_text

# # #         _, buffer = cv2.imencode('.jpg', processed_frame)
# # #         frame_bytes = buffer.tobytes()
# # #         yield (b'--frame\r\n'
# # #                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
# # #     cap.release()

# # # def generate_frames():
# # #     global latest_detected_text
# # #     cap = cv2.VideoCapture(0)
# # #     while True:
# # #         success, frame = cap.read()
# # #         if not success:
# # #             break
# # #         processed_frame, detected_text = detect_license_plate_from_frame(frame)
# # #         latest_detected_text = detected_text  # Update latest detected text
# # #         _, buffer = cv2.imencode('.jpg', processed_frame)
# # #         frame_bytes = buffer.tobytes()
# # #         yield (b'--frame\r\n'
# # #                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
# # #     cap.release()

# # def generate_frames():
# #     global latest_detected_text
# #     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_DSHOW for faster capture on Windows
# #     cap.set(cv2.CAP_PROP_FPS, 15)

# #     while True:
# #         success, frame = cap.read()
# #         if not success:
# #             break

# #         # Resize to improve speed
# #         frame = cv2.resize(frame, (640, 480))

# #         processed_frame, detected_text = detect_license_plate_from_frame(frame)
# #         latest_detected_text = detected_text

# #         # Encode as JPEG
# #         _, buffer = cv2.imencode('.jpg', processed_frame)
# #         frame_bytes = buffer.tobytes()

# #         # Yield for streaming
# #         yield (b'--frame\r\n'
# #                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# #     cap.release()



# # @app.route('/video_feed')
# # def video_feed():
# #     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # @app.route('/latest_text')
# # def latest_text():
# #     return jsonify({"detected_text": latest_detected_text})

# # if __name__ == '__main__':
# #     app.run(debug=True)




# from flask import Flask, request, jsonify, send_file, Response
# from flask_cors import CORS
# import cv2
# import numpy as np
# import torch
# from ultralytics import YOLO
# import os
# import re
# import easyocr  # EasyOCR for OCR

# app = Flask(__name__)
# CORS(app)

# latest_detected_text = ""

# # Load YOLOv8 model (replace with your model path)
# model = YOLO("best.pt")

# # Initialize EasyOCR
# reader = easyocr.Reader(['en'], gpu=False)

# # Create necessary folders
# UPLOAD_FOLDER = "uploads"
# OUTPUT_FOLDER = "outputs"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# # -------- Helper Function to Clean OCR Text ----------

# def extract_significant_text(text_list):
#     """
#     Filters EasyOCR output, concatenates text without spaces,
#     removes 'IND' from start or end if present,
#     and extracts the pattern 2 alphabets + 2 digits + 2 alphabets + 4 digits.
#     """
#     results = []
#     for _, text, _ in text_list:
#         # Extract all alphanumeric parts, uppercase for uniformity
#         cleaned_parts = re.findall(r'[A-Z0-9]+', text.upper())
#         results.extend(cleaned_parts)

#     full_text = "".join(results)

#     # Remove 'IND' from start or end (case-insensitive)
#     full_text = re.sub(r'^(IND)', '', full_text, flags=re.IGNORECASE)  # Remove at start
#     full_text = re.sub(r'(IND)$', '', full_text, flags=re.IGNORECASE)  # Remove at end

#     # Define regex pattern for Indian-style vehicle number plate:
#     # 2 alphabets, 2 digits, 2 alphabets, 4 digits
#     pattern = r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}'

#     match = re.search(pattern, full_text)
#     if match:
#         return match.group(0)  # Return the matched plate
#     else:
#         return full_text  # Return full cleaned text if no pattern match


# # --------- Detection Logic ---------------------------
# def detect_license_plate_from_frame(frame):
#     global latest_detected_text
#     results = model(frame)[0]
#     extracted_texts = []

#     for box in results.boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#         cropped_plate = frame[y1:y2, x1:x2]
#         if cropped_plate is None or cropped_plate.size == 0:
#             continue

#         gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)

#         # OCR using EasyOCR
#         ocr_results = reader.readtext(gray)
#         text = extract_significant_text(ocr_results)

#         if text:
#             extracted_texts.append(text)
#             cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     final_texts = list(set(extracted_texts))  # Remove duplicates
#     latest_detected_text = " ".join(final_texts) if final_texts else "No valid text found"
#     return frame, latest_detected_text

# # ------------- Image Upload Endpoint -----------------
# @app.route('/detect', methods=['POST'])
# def detect_license_plate():
#     if 'image' not in request.files:
#         return jsonify({"error": "No image uploaded"}), 400

#     file = request.files['image']
#     image_path = os.path.join(UPLOAD_FOLDER, file.filename)

#     try:
#         file.save(image_path)
#         image = cv2.imread(image_path)
#         if image is None:
#             return jsonify({"error": "Could not read image"}), 500

#         processed_image, detected_text = detect_license_plate_from_frame(image)

#         output_path = os.path.join(OUTPUT_FOLDER, f"output_{file.filename}")
#         cv2.imwrite(output_path, processed_image)

#         return jsonify({
#             "ocr_text": detected_text,
#             "image_url": f"http://127.0.0.1:5000/output/{file.filename}"
#         })

#     except Exception as e:
#         return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

# # ------------- Serve Processed Image -----------------
# @app.route('/output/<filename>')
# def get_output_image(filename):
#     return send_file(os.path.join(OUTPUT_FOLDER, f"output_{filename}"), mimetype='image/jpeg')

# # ------------- Serve Live Video Feed -----------------
# # @app.route('/video_feed')
# # def video_feed():
# #     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # ------------- Latest OCR Text API -------------------
# @app.route('/latest_text')
# def latest_text():
#     return jsonify({"detected_text": latest_detected_text})

# @app.route('/clear_text', methods=['POST'])
# def clear_detected_text():
#     global latest_detected_text
#     latest_detected_text = ""
#     return jsonify({"message": "Detected text cleared."}), 200

# # ------------- Live Camera Frame Generator -----------
# # def generate_frames():
# #     global latest_detected_text
# #     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# #     cap.set(cv2.CAP_PROP_FPS, 15)

# #     while True:
# #         success, frame = cap.read()
# #         if not success:
# #             break

# #         frame = cv2.resize(frame, (640, 480))
# #         processed_frame, detected_text = detect_license_plate_from_frame(frame)
# #         latest_detected_text = detected_text

# #         _, buffer = cv2.imencode('.jpg', processed_frame)
# #         frame_bytes = buffer.tobytes()

# #         yield (b'--frame\r\n'
# #                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# #     cap.release()

# # ------------------- Run Server ----------------------
# if __name__ == '__main__':
#     app.run(debug=True)
#     # app.run(host='0.0.0.0', port=5000, debug=True)

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
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
