from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import numpy as np

app = FastAPI(
    title="Indian Food Detection API",
    description="YOLOv11 food & vegetable detector (20 classes)",
    version="1.0.0"
)

# Load YOLO
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names


@app.get("/")
def root():
    return FileResponse("static/index.html")


# -----------------------------
# 1️⃣ RETURN JSON
# -----------------------------
@app.post("/predict-json")
async def predict_json(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    results = model.predict(image, imgsz=640, conf=0.25, verbose=False)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class_id": cls_id,
            "class_name": CLASS_NAMES.get(cls_id, str(cls_id)),
            "confidence": round(conf, 4),
            "box_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
        })

    return JSONResponse({
        "num_detections": len(detections),
        "detections": detections
    })


# -----------------------------
# 2️⃣ RETURN IMAGE STREAM (downloadable)
# -----------------------------
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    results = model.predict(image, imgsz=640, conf=0.25, verbose=False)[0]

    annotated = results.plot()[..., ::-1]  # BGR → RGB

    pil_img = Image.fromarray(annotated)
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/jpeg")


# -----------------------------
# 3️⃣ RETURN IMAGE AS FILE — SWAGGER CAN PREVIEW IT
# -----------------------------
@app.post("/download-predicted-image", summary="Download Predicted Image")
async def download_predicted_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    results = model(image)[0]
    annotated = results.plot()

    output_path = "result.jpg"
    Image.fromarray(annotated).save(output_path)

    return FileResponse(output_path, media_type="image/jpeg", filename="prediction.jpg")


# -----------------------------
# 4️⃣ SERVE STATIC UI (index.html)
# -----------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")
