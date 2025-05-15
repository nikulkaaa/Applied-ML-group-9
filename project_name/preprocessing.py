import os
import cv2
import dlib
import numpy as np
from retinaface import RetinaFace
from tensorflow.keras.preprocessing import image

# initialize the Dlib face detector and landmark predictor 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

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
    
    # Use RetinaFace for initial face detection
    detector_retina = RetinaFace()
    results = detector_retina.predict(img_rgb)

    if not results:
        print(f"No faces detected in {image_path}")
        return None

    face = results[0]

    x1, y1, x2, y2 = face['x1'], face['y1'], face['x2'], face['y2']

    # Apply margin
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(img_rgb.shape[1], x2 + margin)
    y2 = min(img_rgb.shape[0], y2 + margin)

    # Crop the face
    cropped_face = img_rgb[y1:y2, x1:x2]
    
    # Align the face using Dlib's 68-point landmarks
    aligned_face = align_face(cropped_face)

    # Resize the aligned face to the right size
    aligned_face_resized = cv2.resize(aligned_face, img_size)

    return aligned_face_resized

def align_face(face_img):
    """
    Align the face based on the eye locations using Dlib's 68-point facial landmarks.
    
    Args:
        face_img (np.ndarray): Cropped face image.
    
    Returns:
        np.ndarray: Aligned face image.
    """
    # Convert the image to grayscale for Dlib
    gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
    
    # Detect faces in the image
    faces = detector(gray)
    if len(faces) == 0:
        print("No face detected for alignment.")
        return face_img

    # Assume the first face is the target
    shape = predictor(gray, faces[0])

    # Get the coordinates of the eyes (left eye and right eye)
    left_eye = (shape.part(36).x, shape.part(36).y)
    right_eye = (shape.part(45).x, shape.part(45).y)
    
    # calculate the center of the eyes
    eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

    # Calculate the angle of rotation to align the eyes horizontally
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Get the rotation matrix and rotate the image to align the eyes
    matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    aligned_face = cv2.warpAffine(face_img, matrix, (face_img.shape[1], face_img.shape[0]))

    return aligned_face

def process_images(input_folder, output_folder, label, img_size=(128, 128), margin=10):
    """
    Processes all images in the input folder, detects faces, crops, aligns, and resizes them, 
    and saves them to the output folder.
    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where processed images will be saved.
        label (int): The label for the images (0 for real, 1 for fake).
        img_size (tuple): Target image size for resizing (default 1298x128).
        margin (int): Margin to add around the detected face (default 10).
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        # Construct the full path to the image
        img_path = os.path.join(input_folder, filename)
        
        # Loop over each image in the input folder
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Detect face, align, and crop the image
            cropped_face = detect_face(img_path, margin, img_size)
            
            if cropped_face is not None:
                # Save the cropped face image to the output folder
                output_img_path = os.path.join(output_folder, filename)
                # Convert back to BGR for saving
                cv2.imwrite(output_img_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
                print(f"Processed {filename} with label {label}")
            else:
                print(f"Skipped {filename}: No face detected.")

def preprocess_data():
    """
    Preprocesses the data for training, validation, and testing by detecting faces and saving them to respective folders.
    """
    # Paths to the input data (folders with images)
    # train_folder_real = 'project_name/data/mini_dataset/Train/Real/'
    # train_folder_fake = 'project_name/data/mini_dataset/Train/Fake/'
    validation_folder_real = 'project_name/data/mini_dataset/Validation/Real/'
    validation_folder_fake = 'project_name/data/mini_dataset/Validation/Fake/'
    # test_folder_real = 'project_name/data/mini_dataset/Test/Real/'
    # test_folder_fake = 'project_name/data/mini_dataset/Test/Fake/'
    
    # Paths to the output data (folders to save cropped faces)
    # output_train_folder_fake = 'project_name/baby_dataset/preprocessed_eye_align/Train/Fake/'
    output_validation_folder_fake = 'project_name/baby_dataset/preprocessed_eye_align/Validation/Fake/'
    # output_test_folder_fake = 'project_name/baby_dataset/preprocessed_eye_align/Test/Fake/'
    # output_train_folder_real = 'project_name/baby_dataset/preprocessed_eye_align/Train/Real/'
    output_validation_folder_real = 'project_name/baby_dataset/preprocessed_eye_align/Validation/Real/'
    # output_test_folder_real = 'project_name/baby_dataset/preprocessed_eye_align/Test/Real/'
    
    # Process the images in each folder with their respective labels
    # process_images(train_folder_real, output_train_folder_real, label=0)
    # process_images(train_folder_fake, output_train_folder_fake, label=1)
    process_images(validation_folder_real, output_validation_folder_real, label=0)
    process_images(validation_folder_fake, output_validation_folder_fake, label=1)
    # process_images(test_folder_real, output_test_folder_real, label=0)
    # process_images(test_folder_fake, output_test_folder_fake, label=1)

if __name__ == '__main__':
    preprocess_data()
