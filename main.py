from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print(" Loading model...")
MODEL = tf.keras.models.load_model("C:/CODE/Potato_Project/models/potato_model.keras")
print(" Model loaded successfully.")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    print(" /ping endpoint hit.")
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data))
        print(f"📷 Image opened successfully. Mode: {image.mode}, Size: {image.size}")
        image = image.resize((256, 256))  # Add resize if needed
        image = image.convert("RGB")      # Ensure 3 channels
        return np.array(image)
    except Exception as e:
        print(f" Error reading image: {e}")
        return None

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        print(f" Received file: {file.filename}")
        image_data = await file.read()
        image = read_file_as_image(image_data)

        if image is None:
            return {"error": "Invalid image format."}

        img_batch = np.expand_dims(image, 0)
        print(f" Image batch shape: {img_batch.shape}")

        predictions = MODEL.predict(img_batch)
        print(f" Raw predictions: {predictions}")

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        print(f" Predicted class: {predicted_class} (Confidence: {confidence})")

        return {
            'class': predicted_class,
            'confidence': float(confidence)
        }

    except Exception as e:
        print(f" Prediction error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
