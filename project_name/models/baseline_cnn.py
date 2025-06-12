import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold, train_test_split
from metrics import ModelMetrics
import matplotlib.pyplot as plt
import cv2

IMG_SIZE   = (128, 128)
BATCH_SIZE = 32
EPOCHS     = 20

K_FOLDS   = 5
TEST_SIZE = 0.20

DATA_ROOT = "data/preprocessed_dataset/preprocessed_no_background"

# Where all weights get saved
MODEL_DIR = "saved_models"
FOLD_DIR  = os.path.join(MODEL_DIR, "baseline_folds")
os.makedirs(FOLD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


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
    filepaths, labels = [], []
    for label_str, label_val in [("real", 0), ("fake", 1)]:
        class_dir = os.path.join(root_dir, label_str)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                filepaths.append(os.path.join(class_dir, fname))
                labels.append(label_val)
    return np.array(filepaths), np.array(labels)

def preprocess_path_label(path, label, augment=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0

    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_contrast(img, 0.9, 1.1)

    return img, tf.cast(label, tf.float32)

def make_dataset(filepaths, labels, batch_size, shuffle=False, augment=False):
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
    plt.title(f"Fold {fold_idx} Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
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
    plt.figure(figsize=(5, 5))
    plt.imshow(np.uint8(superimposed_img))
    plt.axis("off")
    plt.show()

def run_experiment():
    filepaths, labels = load_filepaths_and_labels(DATA_ROOT)
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        filepaths,
        labels,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=42,
        shuffle=True,
    )

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_metrics = []

    for fold_idx, (tr_idx, val_idx) in enumerate(
        skf.split(train_paths, train_labels), start=1
    ):
        print(f"\n=== Fold {fold_idx}/{K_FOLDS} ===")
        tr_paths, tr_labels = train_paths[tr_idx], train_labels[tr_idx]
        val_paths, val_labels = train_paths[val_idx], train_labels[val_idx]

        tr_ds  = make_dataset(tr_paths,  tr_labels, BATCH_SIZE,
                              shuffle=True,  augment=True)
        val_ds = make_dataset(val_paths, val_labels, BATCH_SIZE,
                              shuffle=False, augment=False)

        model = build_baseline_model(input_shape=(128, 128, 3))
        model.compile(optimizer=Adam(1e-4),
                      loss="binary_crossentropy",
                      metrics=["accuracy"])

        history = model.fit(tr_ds, validation_data=val_ds,
                            epochs=EPOCHS, verbose=1)
        plot_loss(history, fold_idx)

        y_true, y_pred, y_scores = [], [], []
        for imgs, lbls in val_ds:
            preds = model.predict(imgs)[:, 0]
            y_true.extend(lbls.numpy().astype(int).tolist())
            y_pred.extend((preds > 0.5).astype(int).tolist())
            y_scores.extend(preds.tolist())

        metrics = ModelMetrics(
            y_true=y_true, y_pred=y_pred, y_scores=y_scores
        ).get_all_metrics()

        metrics["train_acc"] = float(history.history["accuracy"][-1])
        metrics["val_acc"]   = float(metrics.get("accuracy", metrics.get("acc", 0.0)))

        fold_metrics.append(metrics)
        print(f"Fold {fold_idx} metrics:")
        for k, v in metrics.items():
            if k == "confusion_matrix":
                print(f"{k}:")
                print(np.array(v))
            else:
                print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

        model.save(os.path.join(FOLD_DIR, f"baseline_fold_{fold_idx}.keras"))

    # CV SUMMARY + OVER-FITTING CHECK
    print("\n~~~~~~~ K-fold summary ~~~~~~~")
    for k in ["val_acc", "auc", "eer", "f1"]:
        if k in fold_metrics[0]:
            scores = np.array([m[k] for m in fold_metrics])
            print(f"{k.upper():7s}: {scores.mean():.4f} ± {scores.std():.4f}")

    train_accs = np.array([m["train_acc"] for m in fold_metrics])
    val_accs   = np.array([m["val_acc"]   for m in fold_metrics])
    gap     = train_accs.mean() - val_accs.mean()
    std_val = val_accs.std()

    print("\n~~~~~~~ Over-fitting check ~~~~~~~")
    print(f"Train-val acc gap : {gap:.3f}")
    print(f"Val-acc  std-dev  : {std_val:.3f}")
    if gap > 0.10 or std_val > 0.05:
        print("Model is likely over-fitting.")
    else:
        print("No strong signs of over-fitting.")

    # Final model on full 80%
    print("\n~~~~ Training final model on full training split ~~~~")
    final_model = build_baseline_model(input_shape=(128, 128, 3))
    final_model.compile(optimizer=Adam(1e-4),
                        loss="binary_crossentropy",
                        metrics=["accuracy"])

    full_train_ds = make_dataset(train_paths, train_labels,
                                 BATCH_SIZE, shuffle=True, augment=True)
    final_model.fit(full_train_ds, epochs=EPOCHS, verbose=1)

    # Hold-out Evaluation
    test_ds = make_dataset(test_paths, test_labels,
                           BATCH_SIZE, shuffle=False, augment=False)
    y_true, y_pred, y_scores = [], [], []
    for imgs, lbls in test_ds:
        preds = final_model.predict(imgs)[:, 0]
        y_true.extend(lbls.numpy().astype(int).tolist())
        y_pred.extend((preds > 0.5).astype(int).tolist())
        y_scores.extend(preds.tolist())

    test_metrics = ModelMetrics(
        y_true=y_true, y_pred=y_pred, y_scores=y_scores
    ).get_all_metrics()

    print("\n~~~~ Hold-out Test Metrics ~~~~")
    for k, v in test_metrics.items():
        if k == "confusion_matrix":
            print(f"{k}:")
            print(np.array(v))
        else:
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # Save final weights only (no metrics/summary CSVs)
    final_weights = os.path.join(MODEL_DIR, "baseline_model.keras")
    final_model.save(final_weights)
    print(f"\nSaved deployable weights → {final_weights}")

if __name__ == "_main_":
    run_experiment()