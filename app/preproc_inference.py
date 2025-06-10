"""
Inference script for the preprocessing.
Outputs the preprocessed image.
"""

import os
import sys
import cv2
import dlib
import numpy as np
from pathlib import Path
from retinaface import RetinaFace

detector   = dlib.get_frontal_face_detector()
predictor  = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
IMG_EXTS   = (".jpg", ".jpeg", ".png")

def align_face(face_img):
    """Rotate face_img so the eyes lie on a horizontal line. Returns the
    rotated image (RGB). If no landmarks are found it simply returns the
    original image unchanged. """
    gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
    dets = detector(gray)
    if len(dets) == 0:
        return face_img

    shape = predictor(gray, dets[0])
    left_eye  = (shape.part(36).x, shape.part(36).y)
    right_eye = (shape.part(45).x, shape.part(45).y)

    eye_center = ((left_eye[0] + right_eye[0]) // 2,
                  (left_eye[1] + right_eye[1]) // 2)

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    rotated = cv2.warpAffine(face_img, M, (face_img.shape[1], face_img.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def apply_face_mask(aligned_face):
    """Return aligned_face with everything outside its convex-hull landmark
    removed. If landmarks aren't found we return the original
    aligned_face unchanged."""
    gray = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2GRAY)
    dets = detector(gray)
    if len(dets) == 0:
        return aligned_face

    shape = predictor(gray, dets[0])
    landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.int32)
    hull      = cv2.convexHull(landmarks)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    mask_3c = np.dstack([mask] * 3)

    return aligned_face * mask_3c

def detect_face(image_path: str, margin: int = 10, img_size=(128, 128)):
    """
    Detect, align, mask & resize a single face.
    Returns an RGB uint8 numpy array of shape (*img_size*,*img_size*,3) or
    None if no face is detected.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Could not read {image_path}", file=sys.stderr)
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # RetinaFace bounding box
    try:
        faces = RetinaFace.predict(img_rgb)
    except Exception:
        faces = []

    if faces:
        face = faces[0]
        x1, y1, x2, y2 = face["x1"], face["y1"], face["x2"], face["y2"]
    else:
        dets = detector(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY))
        if not dets:
            print(f"No faces detected in {image_path}", file=sys.stderr)
            return None
        d = dets[0]
        x1, y1 = d.left(),  d.top()
        x2, y2 = d.right(), d.bottom()

    # Expand box by margin
    h, w = img_rgb.shape[:2]
    x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
    x2, y2 = min(w, x2 + margin), min(h, y2 + margin)
    crop = img_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        print(f"Empty crop for {image_path}", file=sys.stderr)
        return None

    # Align so eyes are horizontal
    aligned = align_face(crop)

    # Mask out everything outside the convex hull of landmarks
    masked = apply_face_mask(aligned)

    # Resize to output size
    try:
        final = cv2.resize(masked, img_size, interpolation=cv2.INTER_AREA)
    except Exception as e:
        print(f"Resize failed for {image_path}: {e}", file=sys.stderr)
        return None

    return final

def process_single(image_path, margin=10, img_size=(128, 128)):
    face = detect_face(image_path, margin, img_size)
    if face is None:
        print(f"Failed to preprocess {image_path}", file=sys.stderr)
        sys.exit(2) 

    base      = os.path.splitext(image_path)[0]
    out_dir   = f"{base}_preprocessed"
    os.makedirs(out_dir, exist_ok=True)

    save_path = os.path.join(out_dir, f"{Path(base).name}_preprocessed.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
    print(f"Saved {save_path}")


def process_folder(folder, margin=10, img_size=(128, 128)):
    out_dir   = os.path.join(folder, "preprocessed")
    os.makedirs(out_dir, exist_ok=True)
    saved_cnt = 0

    for fn in os.listdir(folder):
        if not fn.lower().endswith(IMG_EXTS):
            continue
        path = os.path.join(folder, fn)
        face = detect_face(path, margin, img_size)
        if face is not None:
            dst = os.path.join(out_dir, fn)
            cv2.imwrite(dst, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            print(f"Preprocessed {fn} → {dst}")
            saved_cnt += 1
        else:
            print(f"Skipping {fn}: no face", file=sys.stderr)

    if saved_cnt == 0:
        print("No faces detected in any images.", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Face‑detect, align, mask background & resize images for inference"
    )
    p.add_argument("path", help="Image file or directory")
    p.add_argument("--margin", type=int, default=10)
    p.add_argument("--size",   type=int, nargs=2, default=[128, 128], metavar=("W", "H"))
    args = p.parse_args()

    size = tuple(args.size)
    if os.path.isfile(args.path):
        process_single(args.path, args.margin, size)
    elif os.path.isdir(args.path):
        process_folder(args.path, args.margin, size)
    else:
        print(f"Path not found: {args.path}", file=sys.stderr)
        sys.exit(1)
