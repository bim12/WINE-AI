# app.py
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input  # AJOUT
import os

MODEL_PATH = "./Models/DL_WINE_06_TL_FT.keras"
IMG_SIZE = 224
CLASS_NAMES = ['Black Rot', 'ESCA', 'Healthy', 'Leaf Blight']

app = FastAPI(
    title="Grapevine Disease Classifier",
    description="API de classification des maladies de la vigne",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

model = tf.keras.models.load_model(MODEL_PATH)

def prepare_image(img: Image.Image) -> np.ndarray:
    # Redimensionne et préprocess pour EfficientNet (IMPORTANT)
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32)
    arr = preprocess_input(arr)   # <---- CRUCIAL pour EfficientNetB0
    return np.expand_dims(arr, axis=0)  # (1,224,224,3)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file)
    except Exception as e:
        return JSONResponse(content={"error": f"Erreur lors de l'ouverture de l'image: {str(e)}"}, status_code=400)

    try:
        arr = prepare_image(img)
    except Exception as e:
        return JSONResponse(content={"error": f"Erreur lors du préprocessing: {str(e)}"}, status_code=400)

    preds = model.predict(arr)
    preds = preds.flatten()
    pred_dict = {cls: float(f"{100*p:.2f}") for cls, p in zip(CLASS_NAMES, preds)}
    top_idx = int(np.argmax(preds))
    top_label = CLASS_NAMES[top_idx]
    result = {
        "prediction": top_label,
        "probabilities (%)": pred_dict
    }
    return JSONResponse(content=result)

if __name__ == "__main__":
    import sys
    port = 8888
    if len(sys.argv) > 2 and sys.argv[1] == "--port":
        port = int(sys.argv[2])
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
