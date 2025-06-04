import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from metrics import ModelMetrics
import matplotlib.pyplot as plt
import pandas as pd
import cv2

# Set image size and batch size
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 1
K_FOLDS = 2
DATA_ROOT = "data/preprocessed_dataset/preprocessed_no_background"

def build_baseline_model(input_shape=(128, 128, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for f in [16, 32, 64, 128]:
        x = layers.Conv2D(f, (3, 3), padding="same", name=f"conv_{f}")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs)
    model.last_conv_layer_name = "conv_128"
    return model

def load_filepaths_and_labels(root_dir):
    """
    Collect all image filepaths under root_dir/real and root_dir/fake, 
    returning (filepaths, labels) arrays.
    """
    filepaths = []
    labels = []
    for label_str, label_val in [("real", 0), ("fake", 1)]:
        class_dir = os.path.join(root_dir, label_str)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                filepaths.append(os.path.join(class_dir, fname))
                labels.append(label_val)
    return np.array(filepaths), np.array(labels)

def preprocess_path_label(path, label, augment=False):
    """
    Given a file path and integer label (0 or 1), load the image, resize, normalize,
    and optionally augment. Return (image_tensor, label_tensor).
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0

    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_contrast(img, 0.9, 1.1)

    label_tensor = tf.cast(label, tf.float32)
    return img, label_tensor

def make_dataset(filepaths, labels, batch_size, shuffle=False, augment=False):
    """
    Create a tf.data.Dataset from arrays of (filepaths, labels). Labels are 0/1 scalars.
    """
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(filepaths))
    ds = ds.map(lambda p, l: preprocess_path_label(p, l, augment=augment),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def plot_loss(history, fold_idx):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"Fold {fold_idx} Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, 0]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(image, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + image
    superimposed_img = np.uint8(superimposed_img)
    plt.figure(figsize=(6, 6))
    plt.imshow(superimposed_img)
    plt.axis("off")
    plt.title("Grad-CAM Overlay")
    plt.show()

def run_kfold():
    # Load all filepaths & labels
    filepaths, labels = load_filepaths_and_labels(DATA_ROOT)

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_metrics = []

    save_dir = "saved_models_test"
    os.makedirs(save_dir, exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(filepaths, labels), start=1):
        print(f"\n=== Fold {fold_idx}/{K_FOLDS} ===")

        train_paths = filepaths[train_idx]
        train_labels = labels[train_idx]
        val_paths = filepaths[val_idx]
        val_labels = labels[val_idx]

        # 3) Build tf.data.Datasets (labels are scalars now)
        train_ds = make_dataset(train_paths, train_labels, BATCH_SIZE, shuffle=True, augment=True)
        val_ds   = make_dataset(val_paths,   val_labels,   BATCH_SIZE, shuffle=False, augment=False)

        # Build and compile model with sigmoid + binary_crossentropy
        model = build_baseline_model(input_shape=(128, 128, 3))
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        # Train
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
        plot_loss(history, fold_idx)

        # Evaluation on validation set
        y_true = []
        y_pred = []
        y_scores = []
        for images_batch, labels_batch in val_ds:
            preds = model.predict(images_batch)[:, 0]
            true_labels = labels_batch.numpy().astype(int)
            predicted_labels = (preds > 0.5).astype(int)
            y_true.extend(true_labels.tolist())
            y_pred.extend(predicted_labels.tolist())
            y_scores.extend(preds.tolist())

        metrics = ModelMetrics(y_true=y_true, y_pred=y_pred, y_scores=y_scores)
        fold_result = metrics.get_all_metrics()
        fold_metrics.append(fold_result)
        print(f"Fold {fold_idx} metrics: {fold_result}")

        shown = 0
        for images_batch, labels_batch in val_ds:
            # images_batch: shape (batch_size, 128,128,3), values in [0,1]
            for i in range(images_batch.shape[0]):
                # 1) Prepare a single image for Grad-CAM
                img_tensor = tf.expand_dims(images_batch[i], axis=0)  # shape (1,128,128,3)
                
                # 2) Compute Grad-CAM heatmap
                heatmap = make_gradcam_heatmap(img_tensor, model, model.last_conv_layer_name)
                
                # 3) Convert image back to uint8 for display
                orig_img = (images_batch[i].numpy() * 255).astype(np.uint8)
                
                # 4) Overlay and show
                display_gradcam(orig_img, heatmap)
                
                shown += 1
                if shown >= 5:
                    break
            if shown >= 5:
                break

        # Save this fold’s model
        model.save(os.path.join(save_dir, f"baseline_fold_{fold_idx}.keras"))

    # Summarize across folds
    import pandas as pd
    df = pd.DataFrame(fold_metrics)
    numeric_df = df.drop(columns=["confusion_matrix"], errors="ignore")
    summary = numeric_df.agg(["mean", "std"])
    print("\n=== K-Fold Summary (Baseline) ===")
    print(summary)
    return summary

if __name__ == "__main__":
    run_kfold()
