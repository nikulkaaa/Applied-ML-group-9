import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from metrics import ModelMetrics
import numpy as np
import matplotlib.pyplot as plt


### This is a very basic baseline model, still needs to be tuned

def build_baseline_model(input_shape=(128, 128, 3), num_classes=2):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    filters = [16, 32, 64, 128]
    for i, f in enumerate(filters):
        x = layers.Conv2D(f, (3, 3), padding='same', name=f'conv_{f}')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((2, 2))(x)

    # Save the output before flattening
    conv_output = x  # this is the last conv output

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    # Save last conv layer name
    model.last_conv_layer_name = 'conv_128'
    model.last_conv_output = conv_output  # optional for clarity
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
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])


# Apply only to training dataset
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
val_ds = val_ds.map(lambda x, y: (x / 255.0, y))  # just normalize

model = build_baseline_model(input_shape=(128, 128, 3))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30
)
def plot_loss(history):
    # Plot training & validation loss values
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_loss(history)


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



def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Create a model that maps the input image to the activations of the last conv layer
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Compute gradients of the top predicted class with respect to the feature map
    grads = tape.gradient(class_channel, conv_outputs)

    # Mean intensity of the gradients over each feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in feature map by the gradient importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
import cv2

def display_gradcam(image, heatmap, alpha=0.4):
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    # Apply colormap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose heatmap on image
    superimposed_img = heatmap * alpha + image
    superimposed_img = np.uint8(superimposed_img)

    plt.figure(figsize=(6, 6))
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.title("Grad-CAM Overlay")
    plt.show()

# Get a batch of images and labels
# for images, labels in val_ds.take(1):
num_images = 5
count = 0

for images, labels in val_ds:
    for i in range(images.shape[0]):
        image = images[i].numpy()
        input_image = np.expand_dims(image, axis=0) / 255.0
        last_conv_layer_name = 'conv_128'
        # Compute gradcam heatmap
        heatmap = make_gradcam_heatmap(input_image, model, last_conv_layer_name)

        display_gradcam((image * 255).astype(np.uint8), heatmap)

        count += 1
        if count >= num_images:
            break
    if count >= num_images:
        break
    # image = images[0].numpy()
    # input_image = np.expand_dims(image, axis=0)

    # # Make sure image is normalized like training
    # input_image = input_image / 255.0

    # last_conv_layer_name = 'conv_128'


    # # Compute Grad-CAM heatmap
    # heatmap = make_gradcam_heatmap(input_image, model, last_conv_layer_name)

    # # Display result
    # display_gradcam((image * 255).astype(np.uint8), heatmap)
    break
