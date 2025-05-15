import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from metrics import ModelMetrics
import numpy as np

### This is a very basic baseline model, still needs to be tuned

def build_baseline_model(input_shape=(128, 128, 3), num_classes=2):
    model = models.Sequential()

    filters = [16, 32, 64, 128]
    for f in filters:
        model.add(layers.Conv2D(f, (3, 3), padding='same', input_shape=input_shape if f == 16 else None))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# Set image size and batch size
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Load training dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'project_name/baby_dataset/preprocessed_eye_align/Train',
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)
# Load validation dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'project_name/baby_dataset/preprocessed_eye_align/Validation',
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Apply data augmentation and normalization
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./128),
    tf.keras.layers.RandomFlip('horizontal'),
])

# Apply only to training dataset
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
val_ds = val_ds.map(lambda x, y: (x / 255.0, y))  # just normalize

model = build_baseline_model(input_shape=(128, 128, 3))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

## EVALUATION
y_true = []
y_pred = []
y_scores = []

for images, labels in val_ds:
    preds = model.predict(images)

    true_labels = np.argmax(labels.numpy(), axis=1)
    predicted_labels = np.argmax(preds, axis=1)

    # For binary classification: collect probability of class 1
    # preds is shape (batch_size, 2), so we take column 1
    positive_class_scores = preds[:, 1]

    y_true.extend(true_labels)
    y_pred.extend(predicted_labels)
    y_scores.extend(positive_class_scores)

metrics = ModelMetrics(
    y_true=y_true,
    y_pred=y_pred,
    y_scores=y_scores 
)
metrics.print_metrics()

# Optional: Get dictionary of results
results = metrics.get_all_metrics()