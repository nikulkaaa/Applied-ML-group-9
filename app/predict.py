import os
# tensorfloww issuess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import argparse
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage

MODEL_PATH = "saved_models/baseline_model.keras"
IMG_EXTS = (".jpg", ".jpeg", ".png")

_model = None

def load_model():
    """Load and cache the saved Keras model."""
    global _model
    if _model is None:
        try:
            _model = tf.keras.models.load_model(MODEL_PATH)
            _model.trainable = False
        except Exception as e:
            print(json.dumps({"error": f"Error loading model: {e}"}))
            sys.exit(1)
    return _model


def load_and_normalize(img_path: str) -> np.ndarray:
    """
    Load a 128x128 image and scale pixels to [0,1]. Returns shape (1,128,128,3).
    """
    img = kimage.load_img(img_path, target_size=(128,128))
    arr = kimage.img_to_array(img)
    arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


def predict_on_image(model, img_path: str) -> dict:
    """
    Predict a single image, returning JSON-serializable dict with label & confidence.
    """
    x = load_and_normalize(img_path)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = "real" if idx == 1 else "fake"
    confidence = float(probs[idx])
    return {"label": label, "confidence": confidence}


def normalize_missing_dir(path: str) -> str:
    """
    If path doesn’t exist but ends with (.jpg|.jpeg)_preprocessed,
    convert it to *_preprocessed (matching preproc_inference output).
    """
    if os.path.exists(path):
        return path
    for ext in (".jpg_preprocessed", ".jpeg_preprocessed"):
        if path.endswith(ext):
            alt = path[:-len(ext)] + "_preprocessed"
            if os.path.isdir(alt):
                return alt
    return path


def main(input_path: str):
    input_path = normalize_missing_dir(input_path)
    model = load_model()

    # Determine target file or directory
    if os.path.isfile(input_path):
        img_file = input_path
    elif os.path.isdir(input_path):
        dir_path = input_path
        imgs = [f for f in os.listdir(dir_path) if f.lower().endswith(IMG_EXTS)]
        if not imgs:
            print(json.dumps({"error": f"No images found in directory: {dir_path}"}))
            sys.exit(1)
        img_file = os.path.join(dir_path, imgs[0])
    else:
        print(json.dumps({"error": f"Path not found: {input_path}"}))
        sys.exit(1)

    # Predict on this single file
    result = predict_on_image(model, img_file)
    print(json.dumps(result))
    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify a preprocessed face image or folder of preprocessed images"
    )
    parser.add_argument("path", help="Path to image file or folder")
    args = parser.parse_args()
    main(args.path)
