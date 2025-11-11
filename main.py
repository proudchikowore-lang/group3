from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import zipfile
import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import json

app = FastAPI()

# Globals
uploaded_extract_dir: str | None = None
model: keras.Model | None = None
class_names: list[str] = []

# Templates
templates_dir = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".zip"):
        return {"error": "Invalid file type. Expected .zip"}

    # Save zip file
    zip_file_path = Path(file.filename)
    with open(zip_file_path, "wb") as f:
        f.write(await file.read())

    # Extract zip
    extract_dir = zip_file_path.stem
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    # Remember dataset location
    global uploaded_extract_dir
    uploaded_extract_dir = extract_dir

    # Debugging: inspect the extracted directory
    extracted_files = []
    for dirpath, dirnames, filenames in os.walk(extract_dir):
        extracted_files.append((dirpath, dirnames, filenames))

    # Log the structure of the extracted files
    print("Extracted files:")
    for path, dirs, files in extracted_files:
        print(f"Directory: {path}, Subdirectories: {dirs}, Files: {files}")

    # Check for 'train' directory within the nested path
    train_dir = Path(uploaded_extract_dir) / "fruit_classification" / "train"
    if not train_dir.exists():
        return {"error": f"'train' folder not found inside {uploaded_extract_dir}/fruit_classification"}

    # Prepare image counts
    image_counts = {}
    for class_dir in train_dir.iterdir():
        if class_dir.is_dir():
            image_counts[class_dir.name] = len(list(class_dir.glob("*.jpg")))  # assuming jpg images

    return {
        "message": "Dataset uploaded and extracted",
        "dataset_root": extract_dir,
        "training_classes": image_counts,
    }


# Training parameters
batch_size = 32
img_height = 180
img_width = 180


@app.post("/train")
async def train():
    global uploaded_extract_dir, model, class_names

    if not uploaded_extract_dir:
        return {"error": "No dataset uploaded. Please upload a dataset first."}

    # Check for 'train' directory within the nested path
    train_dir = Path(uploaded_extract_dir) / "fruit_classification" / "train"
    if not train_dir.exists():
        return {"error": f"'train' folder not found inside {uploaded_extract_dir}/fruit_classification"}

    # Get class names from subfolders of train
    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    if not class_names:
        return {"error": "No classes found in the train folder."}

    # Save class names for later use
    with open("class_names.json", "w") as f:
        json.dump(class_names, f)

    # Build datasets
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    num_classes = len(class_names)

    # Build CNN model
    model = keras.Sequential([
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes)  # Number of classes
    ])

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    from tensorflow.keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train
    epochs = 5
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping]
    )

    # Save model
    model.save("my_model.keras")

    return {
        "message": "Training completed and model saved.",
        "classes": class_names,
        "final_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model, class_names

    # Load model if not already in memory
    if model is None:
        try:
            model = keras.models.load_model("my_model.keras")
        except Exception as e:
            return {"error": f"Failed to load model: {str(e)}"}

    # Load class names if missing
    if not class_names:
        try:
            with open("class_names.json", "r") as f:
                class_names = json.load(f)
        except Exception:
            return {"error": "Class names not found. Please train first."}

    # Preprocess uploaded image
    img = Image.open(file.file).convert("RGB")
    img = img.resize((img_width, img_height))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[int(np.argmax(score))]

    return {
        "predicted_class": predicted_class,
        "confidence": float(100 * np.max(score)),
    }

# To run the app, use: uvicorn main:app --reload