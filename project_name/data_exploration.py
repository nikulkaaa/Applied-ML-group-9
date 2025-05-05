import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from insightface.app import FaceAnalysis
from preprocessing import detect_face

### We will tune this when we have all data from the sub-dataset preprocessed

# Initialize the face analyzer model
app = FaceAnalysis()
 # Load the model on the CPU → if GPU then we change it to 1
app.prepare(ctx_id=0)

def calculate_detection_rate(input_folder):
    """Calculate the detection rate."""
    total_images = 0
    detected_faces = 0
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            total_images += 1
            if detect_face(img_path):
                detected_faces += 1
    return detected_faces / total_images * 100

def calculate_bounding_box_size(input_folder):
    """Calculate the bounding-box size relative to the image size."""
    face_areas = []
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            cropped_face, (x1, y1, x2, y2) = detect_face(img_path)
            face_area = (x2 - x1) * (y2 - y1)
            image_area = cropped_face.shape[0] * cropped_face.shape[1]
            face_areas.append(face_area / image_area)
    
    return np.mean(face_areas), np.std(face_areas), np.percentile(face_areas, 25), np.percentile(face_areas, 75)

def calculate_sharpness(input_folder):
    """Calculate sharpness using the variance of the Laplacian."""
    sharpness_values = []
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            variance_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_values.append(variance_laplacian)
    
    return np.mean(sharpness_values), np.std(sharpness_values)

def calculate_brightness_contrast(input_folder):
    """Calculate the brightness and contrast of the images."""
    brightness_values = []
    contrast_values = []
    
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            brightness = np.mean(img_rgb)
            contrast = np.std(img_rgb)
            brightness_values.append(brightness)
            contrast_values.append(contrast)

    return np.mean(brightness_values), np.std(brightness_values), np.mean(contrast_values), np.std(contrast_values)

def estimate_age_gender_ethnicity(input_folder):
    """Estimate age, gender, and ethnicity for each face in the folder."""
    age_results = []
    gender_results = []
    ethnicity_results = []

    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(img_path)
            faces = app.get(img)  # Detect faces and their attributes
            
            for face in faces:
                # Extract attributes
                gender = face.gender
                age = face.age
                ethnicity = face.race

                # Store results
                gender_results.append(gender)
                age_results.append(age)
                ethnicity_results.append(ethnicity)

    # Plotting the age distribution using a boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=age_results)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.show()

    # Return the distributions for possible further analysis
    return np.mean(age_results), np.std(age_results), gender_results, ethnicity_results

def process_images_for_exploration(input_folder):
    """Process the images and compute the metrics."""
    detection_rate = calculate_detection_rate(input_folder)
    print(f"Detection Rate: {detection_rate}%")
    
    mean_bbox, std_bbox, p25_bbox, p75_bbox = calculate_bounding_box_size(input_folder)
    print(f"Bounding Box Size (Mean/STD/25th/75th Percentile): {mean_bbox}/{std_bbox}/{p25_bbox}/{p75_bbox}")
    
    mean_sharpness, std_sharpness = calculate_sharpness(input_folder)
    print(f"Sharpness (Mean/STD of Laplacian Variance): {mean_sharpness}/{std_sharpness}")
    
    mean_brightness, std_brightness, mean_contrast, std_contrast = calculate_brightness_contrast(input_folder)
    print(f"Brightness (Mean/STD): {mean_brightness}/{std_brightness}")
    print(f"Contrast (Mean/STD): {mean_contrast}/{std_contrast}")
    
    # Estimating Age, Gender, and Ethnicity
    estimate_age_gender_ethnicity(input_folder)

def preprocess_data_exploration():
    """Preprocess and explore the data for training, validation, and testing."""
    # Paths to the input data (folders with images)
    train_folder_real = 'project_name/data/Dataset/Train/Real/'
    train_folder_fake = 'project_name/data/Dataset/Train/Fake/'
    validation_folder_real = 'project_name/data/Dataset/Validation/Real/'
    validation_folder_fake = 'project_name/data/Dataset/Validation/Fake/'
    test_folder_real = 'project_name/data/Dataset/Test/Real/'
    test_folder_fake = 'project_name/data/Dataset/Test/Fake/'
    
    # Perform exploration on the dataset
    print("Exploring Real Train Data")
    process_images_for_exploration(train_folder_real)
    print("Exploring Fake Train Data")
    process_images_for_exploration(train_folder_fake)
    print("Exploring Real Validation Data")
    process_images_for_exploration(validation_folder_real)
    print("Exploring Fake Validation Data")
    process_images_for_exploration(validation_folder_fake)
    print("Exploring Real Test Data")
    process_images_for_exploration(test_folder_real)
    print("Exploring Fake Test Data")
    process_images_for_exploration(test_folder_fake)

if __name__ == '__main__':
    preprocess_data_exploration()
