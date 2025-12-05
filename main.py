from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import time

# Creating the FastAPI app
app = FastAPI(
    title="Indian Food Detection API",
    description="YOLOv11 food & vegetable detector (20 classes)",
    version="1.0.0"
)

# Loading YOLO
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names

# Returning index.html
@app.get("/")
def root():
    return FileResponse("static/index.html")

# Predict-JSON Route
@app.post("/predict-json")
async def predict_json(file: UploadFile = File(...)):
    start_time = time.time()  # Start inference timer

    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Run YOLO model
    results = model.predict(image, imgsz=640, conf=0.25, verbose=False)[0]

    end_time = time.time()
    inference_time = round((end_time - start_time) * 1000, 2)  # in ms

    # Count Objects
    class_count = {}
    total_objects = 0

    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        class_name = CLASS_NAMES.get(cls_id, str(cls_id))

        class_count[class_name] = class_count.get(class_name, 0) + 1
        total_objects += 1

    unique_classes = list(class_count.keys())

    # JSON Format
    return JSONResponse({
        "unique_classes": unique_classes,
        "counts": class_count,
        "total_objects_detected": total_objects,
        "inference_time_ms": inference_time
    })


# Predict-Image Route
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


# Download-Predicted-Image Route
@app.post("/download-predicted-image", summary="Download Predicted Image")
async def download_predicted_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    results = model(image)[0]
    annotated = results.plot()

    output_path = "result.jpg"
    Image.fromarray(annotated).save(output_path)

    return FileResponse(output_path, media_type="image/jpeg", filename="prediction.jpg")

# Mounting index.html to Serve
app.mount("/static", StaticFiles(directory="static"), name="static")
