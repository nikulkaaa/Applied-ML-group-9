"""
Preprocessing file.
Face detected with RetinaFace then eye-aligned done (dlib 68 landmarks) and
background-masked by the face convex-hull.
"""

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
    Detects a face with RetinaFace, crops + aligns it, then masks out
    everything outside the convex hull of the 68‐point landmarks.
    Finally resizes to img_size.

    Returns:
        np.ndarray or None: The 128×128 masked RGB face.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image not found: {image_path}")
        return None

    # Convert BGR → RGB for RetinaFace
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Run RetinaFace to get bounding box
    detector_retina = RetinaFace()
    results = detector_retina.predict(img_rgb)
    if not results:
        print(f"No faces detected in {image_path}")
        return None

    # Take the first (biggest) face
    face = results[0]
    x1, y1, x2, y2 = face['x1'], face['y1'], face['x2'], face['y2']

    # Apply margin
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(img_rgb.shape[1], x2 + margin)
    y2 = min(img_rgb.shape[0], y2 + margin)

    # Crop to that expanded box
    cropped_face = img_rgb[y1:y2, x1:x2]
    if cropped_face.size == 0:
        print(f"Crop was empty for {image_path}")
        return None

    # Align via Dlib’s landmarks (rotating so eyes are horizontal)
    aligned_face = align_face(cropped_face)

    # Re‐run dlib landmarks on the aligned face so we can build a convex hull mask of the exact face shape
    gray = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2GRAY)
    dets = detector(gray)
    if len(dets) == 0:
        # If alignment lost the face skip masking and just resize the aligned crop
        masked_face = aligned_face.copy()
    else:
        # We assume the first detected face is the one we want
        shape = predictor(gray, dets[0])
        # Collect the 68 (x,y) points
        landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.int32)

        # Build a convex hull around all 68 points
        hull = cv2.convexHull(landmarks)

        # Create a mask of the same height/width as aligned_face
        h, w = aligned_face.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Fill the convex hull polygon with “1” (white) on the mask
        cv2.fillConvexPoly(mask, hull, 1)

        # Convert single‐channel mask → 3‐channel so we can elementwise‐multiply
        mask_3ch = np.stack([mask, mask, mask], axis=-1)

        # Apply the mask: all outside‐hull pixels become 0 (black)
        masked_face = aligned_face * mask_3ch
    
    # Resize the masked face to (128×128)
    try:
        final_face = cv2.resize(masked_face, img_size, interpolation=cv2.INTER_AREA)
    except Exception as e:
        print(f"Resize failed for {image_path}: {e}")
        return None

    return final_face


def align_face(face_img):
    """
    Rotates face_img so that the eyes lie on a horizontal line.
    Returns the rotated face_img. If no landmarks are found, returns original.
    """
    gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
    dets = detector(gray)
    if len(dets) == 0:
        # no face skip alignment
        return face_img

    shape = predictor(gray, dets[0])
    left_eye = (shape.part(36).x, shape.part(36).y)
    right_eye = (shape.part(45).x, shape.part(45).y)

    # compute center of eyes
    eye_center = ((left_eye[0] + right_eye[0]) // 2,
                  (left_eye[1] + right_eye[1]) // 2)

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # rotation matrix / warp
    M = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
    aligned = cv2.warpAffine(face_img, M, (face_img.shape[1], face_img.shape[0]),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return aligned

def process_images(input_folder, output_folder, label, img_size=(128, 128), margin=10):
    """
    Processes all images in input_folder by:
      1. Detecting + aligning + masking out background.
      2. Resizing to img_size.
      3. Saving the result as a JPEG into output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if not (filename.lower().endswith(".jpg") or filename.lower().endswith(".png")):
            continue

        img_path = os.path.join(input_folder, filename)
        masked_face = detect_face(img_path, margin=margin, img_size=img_size)
        if masked_face is None:
            print(f"Skipped {filename}: No face detected.")
            continue

        # Convert RGB→BGR and write out
        out_path = os.path.join(output_folder, filename)
        bgr = cv2.cvtColor(masked_face, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, bgr)
        print(f"Processed {filename} with label {label}")

def preprocess_data():
    """
    Preprocesses the data for training, validation, and testing by detecting faces and saving them to respective folders.
    """
    # Paths to the input data
    real_origin = 'data/merged_raw/real'
    fake_origin = 'data/merged_raw/fake'
    
    # Paths to the output data
    real_goal = 'data/preprocessed_dataset/preprocessed_no_background/real'
    fake_goal = 'data/preprocessed_dataset/preprocessed_no_background/fake'
    
    # Process the images in each folder
    process_images(real_origin, real_goal, label=1)
    process_images(fake_origin, fake_goal, label=0)

if __name__ == '__main__':
    preprocess_data()
