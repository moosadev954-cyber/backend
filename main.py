from fastapi import FastAPI, File, UploadFile
from tensorflow import keras
from PIL import Image
import numpy as np
import io

app = FastAPI()

model = keras.models.load_model("mnist_model.keras")


@app.get("/")
def home():
    return {"message": "MNIST API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  
    img = img.resize((28, 28))

  
    img_array = np.array(img).astype("float32") / 255
    img_array = img_array.reshape(1, 28 * 28)

    prediction = model.predict(img_array)
    predicted_digit = int(np.argmax(prediction))

    return {"predicted_digit": predicted_digit}
