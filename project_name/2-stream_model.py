import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model

def conv_block(x, filters, use_dropout=False, dropout_rate=0.3, name=None):
    shortcut = layers.Conv2D(filters, 1, padding='same')(x)

    # main branch
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    if use_dropout:
        x = layers.Dropout(dropout_rate)(x)

    # residual add (same spatial size)
    x = layers.Add()([x, shortcut])

    # now down-sample both together
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    return x

def build_two_stream_model(input_shape=(128, 128, 3), num_classes=2):
    # Inputs
    input_a = Input(shape=input_shape, name='face_crop_input')
    input_b = Input(shape=input_shape, name='reprojected_input')

    # --- First Stream ---
    x1 = input_a
    for i, filters in enumerate([32, 64, 128, 256, 256]):
        use_dropout = (i == 2 or i == 4)
        x1 = conv_block(x1, filters, use_dropout=use_dropout, name=f'stream1_block{i+1}')

    # --- Second Stream ---
    x2 = input_b
    for i, filters in enumerate([32, 64, 128, 256, 256]):
        use_dropout = (i == 2 or i == 4)
        x2 = conv_block(x2, filters, use_dropout=use_dropout, name=f'stream2_block{i+1}')

    # Flatten and concatenate
    x1 = layers.Flatten()(x1)
    x2 = layers.Flatten()(x2)
    concatenated = layers.Concatenate()([x1, x2])

    # 512 → 128 bottleneck
    x = layers.Dense(512)(concatenated)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)

    # Output layer
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[input_a, input_b], outputs=output, name='TwoStreamDeepfakeDetector')
    return model

import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model
import matplotlib.pyplot as plt
from metrics import ModelMetrics
import numpy as np
import os

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

original_2d_path = 'project_name/data/preprocessed_dataset/preprocessed_eye_align/Train'
reprojection_path = 'project_name/data/3DRecon/Depth//train'

# Load file paths
original_ds = tf.keras.preprocessing.image_dataset_from_directory(
    original_2d_path,
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=None,  # Load single images
    shuffle=False
)

reproj_ds = tf.keras.preprocessing.image_dataset_from_directory(
    reprojection_path,
    label_mode=None,
    image_size=IMG_SIZE,
    batch_size=None,
    shuffle=False
)

# Zip datasets together
paired_ds = tf.data.Dataset.zip(((original_ds.map(lambda x, y: x), reproj_ds), original_ds.map(lambda x, y: y)))

# Preprocessing
def preprocess(inputs, label):
    original, reproj = inputs                # unpack the two images
    original = tf.image.convert_image_dtype(original, tf.float32) / 255.0
    reproj   = tf.image.convert_image_dtype(reproj,   tf.float32) / 255.0
    return (original, reproj), label

def extract_image(x, y): return x

def extract_label(x, y): return y

# Separate images and labels from original_ds
original_imgs = original_ds.map(extract_image)
original_labels = original_ds.map(extract_label)

# Combine datasets: ((original_img, reproj_img), label)
paired_ds = tf.data.Dataset.zip(((original_imgs, reproj_ds), original_labels))

# Apply preprocessing
paired_ds = paired_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
paired_ds = paired_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# Repeat the same for validation
val_original_path = 'project_name/data/preprocessed_dataset/preprocessed_eye_align/Validation'
val_reproj_path = 'project_name/data/3DRecon/Depth/validation'

val_original_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_original_path,
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=None,
    shuffle=False
)

val_reproj_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_reproj_path,
    label_mode=None,
    image_size=IMG_SIZE,
    batch_size=None,
    shuffle=False
)
val_paired_ds = tf.data.Dataset.zip(((val_original_ds.map(lambda x, y: x), val_reproj_ds), val_original_ds.map(lambda x, y: y)))
# Separate validation images and labels
val_original_imgs = val_original_ds.map(extract_image)
val_labels = val_original_ds.map(extract_label)

# Combine: ((original_img, reproj_img), label)
val_paired_ds = tf.data.Dataset.zip(((val_original_imgs, val_reproj_ds), val_labels))

# Apply the same preprocessing
val_paired_ds = val_paired_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
val_paired_ds = val_paired_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


model = build_two_stream_model(input_shape=(128, 128, 3))

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    paired_ds,
    validation_data=val_paired_ds,
    epochs=3
)

def plot_loss(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Two-Stream Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_loss(history)
y_true = []
y_pred = []
y_scores = []

for (img1_batch, img2_batch), labels in val_paired_ds:
    preds = model.predict([img1_batch, img2_batch])

    true = np.argmax(labels.numpy(), axis=1)
    pred = np.argmax(preds, axis=1)
    score = preds[:, 1]  # Confidence for class 1

    y_true.extend(true)
    y_pred.extend(pred)
    y_scores.extend(score)

metrics = ModelMetrics(y_true, y_pred, y_scores)
metrics.print_metrics()
results = metrics.get_all_metrics()
