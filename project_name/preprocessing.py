import os
import cv2
import numpy as np
from retinaface import RetinaFace
from tensorflow.keras.preprocessing import image


def detect_face(image_path, margin=10, img_size=(128, 128)):
    """
    Detects and crops faces from the input image using RetinaFace.

    Args:
        image_path (str): Path to the image file.
        margin (int): Margin to add around the detected face.
        img_size (tuple): Size to resize the cropped face to.

    Returns:
        np.ndarray or None: Cropped and resized face, or None if no face is found.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image not found: {image_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detector = RetinaFace()
    results = detector.predict(img_rgb)

    if not results:
        print(f"No faces detected in {image_path}")
        return None

    face = results[0]

    x1, y1, x2, y2 = face['x1'], face['y1'], face['x2'], face['y2']

    # Apply margin safely
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(img_rgb.shape[1], x2 + margin)
    y2 = min(img_rgb.shape[0], y2 + margin)

    cropped_face = img_rgb[y1:y2, x1:x2]
    cropped_face_resized = cv2.resize(cropped_face, img_size)

    return cropped_face_resized

def process_images(input_folder, output_folder, label, img_size=(256, 256), margin=10):
    """
    Processes all images in the input folder, detects faces, crops and resizes them, and saves them to the output folder.
    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where processed images will be saved.
        label (int): The label for the images (0 for real, 1 for fake).
        img_size (tuple): Target image size for resizing (default 256x256).
        margin (int): Margin to add around the detected face (default 10).
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop over each image in the input folder
    for filename in os.listdir(input_folder):
        # Construct the full path to the image
        img_path = os.path.join(input_folder, filename)
        
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Detect face and crop the image
            cropped_face = detect_face(img_path, margin, img_size)
            
            if cropped_face is not None:
                # Save the cropped face image to the output folder
                output_img_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_img_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))  # Convert back to BGR for saving
                print(f"Processed {filename} with label {label}")

def preprocess_data():
    """
    Preprocesses the data for training, validation, and testing by detecting faces and saving them to respective folders.
    """
    print(f"------------------DIRECTORY HERE: {os.getcwd()}")
    # Paths to the input data (folders with images)
    train_folder_real = 'project_name/baby_dataset/Train/Real/'
    train_folder_fake = 'project_name/baby_dataset/Train/Fake/'
    validation_folder_real = 'project_name/baby_dataset/Validation/Real/'
    validation_folder_fake = 'project_name/baby_dataset/Validation/Fake/'
    test_folder_real = 'project_name/baby_dataset/Test/Real/'
    test_folder_fake = 'project_name/baby_dataset/Test/Fake/'
    
    # Paths to the output data (folders to save cropped faces)
    output_train_folder_fake = 'project_name/baby_dataset/preprocessed/Train/Fake/'
    output_validation_folder_fake = 'project_name/baby_dataset/preprocessed/Validation/Fake/'
    output_test_folder_fake = 'project_name/baby_dataset/preprocessed/Test/Fake/'
    output_train_folder_real = 'project_name/baby_dataset/preprocessed/Train/Fake/Real/'
    output_validation_folder_real = 'project_name/baby_dataset/preprocessed/Validation/Real/'
    output_test_folder_real = 'project_name/baby_dataset/preprocessed/Test/Real/'
    
    # Process the images in each folder with their respective labels
    # Label 0 for 'real' images, 1 for 'fake' images
    process_images(train_folder_fake, output_train_folder_fake, label=1)
    process_images(train_folder_real, output_train_folder_real, label=0)
    process_images(train_folder_fake, output_train_folder_fake, label=1)
    process_images(validation_folder_real, output_validation_folder_real, label=0)
    process_images(validation_folder_fake, output_validation_folder_fake, label=1)
    process_images(test_folder_real, output_test_folder_real, label=0)
    process_images(test_folder_fake, output_test_folder_fake, label=1)

if __name__ == '__main__':
    preprocess_data()
