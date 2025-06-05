"""
Inference script for the baseline.
Predicts whether a preprocessed face is real or fake
and saves the Grad-CAM.
"""

import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
import cv2


MODEL_PATH = "saved_models/baseline_model.keras"
IMG_EXTS   = (".jpg", ".jpeg", ".png")
UPLOADS_ROOT = "uploads"

_model = None

def load_model():
    """Lazy-load the saved Keras model and attach .last_conv_layer_name."""
    global _model
    if _model is None:
        try:
            _model = tf.keras.models.load_model(MODEL_PATH)
            _model.trainable = False
            if not hasattr(_model, "last_conv_layer_name"):
                for layer in reversed(_model.layers):
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        _model.last_conv_layer_name = layer.name
                        break
                else:
                    raise ValueError("No Conv2D layer found in the model.")
        except Exception as e:
            print(json.dumps({"error": f"Error loading model: {e}"}))
            return None
    return _model


def load_and_normalize(img_path: str) -> np.ndarray:
    img = kimage.load_img(img_path, target_size=(128, 128))
    arr = kimage.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        class_chan = preds[:, 0]
    grads        = tape.gradient(class_chan, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_heatmap_on_image(orig, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return np.uint8(heatmap * alpha + orig)


def predict_on_image(model, img_path: str) -> dict:
    x = load_and_normalize(img_path)
    if x is None:
        return {"error": "Failed to load image."}

    prob_from_model_is_P_fake = model.predict(x, verbose=0)[0][0]

    if prob_from_model_is_P_fake >= 0.5:
        label = "fake"
    else:
        label = "real"

    if label == "fake":
        # Fake prob is P(fake)
        confidence = float(prob_from_model_is_P_fake)
    else: # label == "real"
        # Real prob is P(real) = 1.0 - P(fake)
        confidence = float(1.0 - prob_from_model_is_P_fake)

    result = {"label": label, "confidence": confidence}

    try:
        orig = kimage.img_to_array(
            kimage.load_img(img_path, target_size=(128, 128))
        ).astype(np.uint8)
        heatmap = make_gradcam_heatmap(x, model, model.last_conv_layer_name)
        overlay = overlay_heatmap_on_image(orig, heatmap)

        saliency_path_abs = os.path.splitext(img_path)[0] + "_saliency.png"
        cv2.imwrite(
            saliency_path_abs,
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        )
        saliency_rel = os.path.relpath(saliency_path_abs, start=UPLOADS_ROOT)
        result["saliency"] = f"{UPLOADS_ROOT}/{saliency_rel}".replace(os.sep, "/")
    except Exception as e:
        result["saliency_error"] = f"Grad-CAM failed: {str(e)}"
        if "error" not in result:
             result["error_details_gradcam"] = f"Grad-CAM failed: {str(e)}"


    return result


def main(target: str):
    model = load_model()
    if model is None:
        return

    if os.path.isfile(target):
        img_file = target
    elif os.path.isdir(target):
        imgs = [f for f in os.listdir(target) if f.lower().endswith(IMG_EXTS)]
        if not imgs:
            print(json.dumps({"error": f"No images in {target}"}))
            return
        img_file = os.path.join(target, imgs[0])
    else:
        print(json.dumps({"error": f"Path not found: {target}"}))
        return

    print(json.dumps(predict_on_image(model, img_file)))
    sys.stdout.flush()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path", help="Image file or folder")
    args = p.parse_args()
    main(args.path)
