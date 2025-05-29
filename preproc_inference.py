import os
import sys
import cv2
import dlib
import numpy as np
from retinaface import RetinaFace

# initialisation
detector   = dlib.get_frontal_face_detector()
predictor  = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
IMG_EXTS   = (".jpg", ".jpeg", ".png")

def detect_face(image_path, margin=10, img_size=(128, 128)):
    """
    Detect, align and resize a single face.
    Returns RGB numpy array or None if no face.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read {image_path}", file=sys.stderr)
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # RetinaFace first
    try:
        faces = RetinaFace.predict(img_rgb)
    except Exception:
        faces = []

    if faces:
        face = faces[0]
        x1, y1, x2, y2 = face["x1"], face["y1"], face["x2"], face["y2"]
    else:
        # fallback to dlib
        dets = detector(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY))
        if not dets:
            print(f"No faces detected in {image_path}", file=sys.stderr)
            return None
        d      = dets[0]
        x1, y1 = d.left(), d.top()
        x2, y2 = d.right(), d.bottom()

    # crop with margin
    h, w, _ = img_rgb.shape
    x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
    x2, y2 = min(w, x2 + margin), min(h, y2 + margin)
    crop   = img_rgb[y1:y2, x1:x2]

    # resize
    return cv2.resize(crop, img_size)

def process_single(image_path, margin=10, img_size=(128,128)):
    face = detect_face(image_path, margin, img_size)
    if face is None:
        print(f"Failed to preprocess {image_path}", file=sys.stderr)
        sys.exit(2)                          # <<< special code for “no face”
    base, _   = os.path.splitext(image_path)
    out_dir   = f"{base}_preprocessed"
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(
        out_dir, f"{os.path.basename(base)}_preprocessed.jpg"
    )
    cv2.imwrite(save_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
    print(f"Saved {save_path}")

def process_folder(folder, margin=10, img_size=(128,128)):
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
        description="Face-detect, align & resize images for inference")
    p.add_argument("path", help="Image file or directory")
    p.add_argument("--margin", type=int, default=10)
    p.add_argument("--size",   type=int, nargs=2, default=[128,128])
    args = p.parse_args()

    size = tuple(args.size)
    if os.path.isfile(args.path):
        process_single(args.path, args.margin, size)
    elif os.path.isdir(args.path):
        process_folder(args.path, args.margin, size)
    else:
        print(f"Path not found: {args.path}", file=sys.stderr)
        sys.exit(1)
