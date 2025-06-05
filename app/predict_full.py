#!/usr/bin/env python
"""
python app/predict_full.py <preproc_dir> <deca_dir>

Args
    preproc_dir - folder containing the 2-D face crop (exactly one image expected)
    deca_dir - folder produced by DECA (contains depth_*, normals_*, orig_rendered_* files)

Outputs (a JSON dict)
    image_is_real - boolean prediction
    confidence - model confidence in [0,1]
    saliency - relative path to the saved Grad-CAM-like overlay (or null if unavailable)
    rendered_3d_image - relative path to the DECA rendered image (or null)
    depth_map_image - relative path to the DECA depth map (or null)
    normals_map_image - relative path to the DECA normals map (or null)

"""

from __future__ import annotations

import sys
import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torchvision import transforms
from PIL import Image

# Config
MODEL_ROOT = Path("checkpoints") # expects checkpoints/fold_{i}/best.pt
IMG_SIZE = 224 
IMG_EXTS = (".png", ".jpg", ".jpeg")
UPLOADS_ROOT = "uploads"
DEVICE = torch.device("cpu") # or "cuda:0" for gpu

Path(UPLOADS_ROOT).mkdir(parents=True, exist_ok=True)

def error_exit(msg: str, code: int = 2):
    """Print a JSON error object and exit with given code."""
    print(json.dumps({"error": msg}))
    sys.exit(code)


def first_file_containing(dirpath: Path, substring: str) -> Path | None:
    """Recursively search dirpath for the first image file whose name contains `substring`."""
    for p in dirpath.rglob(f"*{substring}*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            return p
    return None


def load_and_resize_pil(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    return transforms.ToTensor()(img)


def load_depth_gray(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(f"Cannot read depth image: {path}")
    arr = cv2.resize(arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    arr = arr.astype(np.float32)
    return arr / (arr.max() + 1e-6)


def overlay_heatmap(orig: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = orig.astype(np.float32) * (1 - alpha) + heatmap.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def save_saliency(orig: np.ndarray, saliency_map: np.ndarray, out_base: Path) -> str:
    """
    Save a blended (Grad-CAM) overlay at out_base.png, returning its path relative to UPLOADS_ROOT.
    out_base is a Path like `uploads/<filename>_preprocessed/<filename>_saliency`.
    """
    out_path = out_base.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay_heatmap(orig, saliency_map), cv2.COLOR_RGB2BGR))
    uploads_root_str = str(UPLOADS_ROOT)
    try:
        rel = out_path.relative_to(uploads_root_str)
        return f"{uploads_root_str}/{rel}".replace(os.sep, "/")
    except ValueError:
        return str(out_path).replace(os.sep, "/")

def compute_saliency(rgb_batch: torch.Tensor, err_batch: torch.Tensor, models: list[torch.nn.Module]) -> np.ndarray:
    """Return a normalized 2-D saliency map (H×W float in [0,1])."""
    rgb_batch = rgb_batch.clone().detach().requires_grad_(True)
    accumulated = torch.zeros_like(rgb_batch)

    for m in models:
        m.zero_grad(set_to_none=True)
        out = m(rgb_batch, err_batch)
        if out.numel() != 1:
            out = out.view(-1)[0]
        out.backward(retain_graph=True)
        accumulated += rgb_batch.grad.detach().abs()
        rgb_batch.grad.zero_()

    saliency = accumulated.mean(dim=0)
    saliency = saliency.mean(dim=0)
    saliency = saliency / (saliency.max() + 1e-6)
    return saliency.cpu().numpy()

def load_fold_model(fold_dir: Path):
    ckpt = fold_dir / "best.pt"
    if not ckpt.is_file():
        error_exit(f"Checkpoint not found: {ckpt}", code=1)

    try:
        model = torch.jit.load(str(ckpt), map_location=DEVICE)
    except RuntimeError as e:
        try:
            maybe_dict = torch.load(str(ckpt), map_location="cpu")
            if isinstance(maybe_dict, dict):
                error_exit(
                    f"Checkpoint in {fold_dir.name} appears to be a state_dict, not TorchScript.\n"
                    "Please re-export this fold as TorchScript.",
                    code=2,
                )
            else:
                error_exit(f"Unable to interpret checkpoint at {ckpt}: {e}", code=1)
        except Exception:
            error_exit(f"Error loading checkpoint [{ckpt}]: {e}", code=1)

    model.eval()
    return model


def ensemble_predict(rgb_batch: torch.Tensor, err_batch: torch.Tensor, models: list[torch.nn.Module]) -> float:
    with torch.no_grad():
        preds = []
        for m in models:
            out = m(rgb_batch, err_batch)
            if out.numel() != 1:
                out = out.view(-1)[0]
            preds.append(torch.sigmoid(out))
        return float(torch.stack(preds).mean().item())

def main(preproc_dir: Path, deca_dir: Path):
    if not preproc_dir.is_dir():
        error_exit(f"Preproc dir not found: {preproc_dir}", code=2)
    if not deca_dir.is_dir():
        error_exit(f"DECA dir not found: {deca_dir}", code=2)

    face_crop: Path | None = None
    for ext in IMG_EXTS:
        candidates = list(preproc_dir.glob(f"*{ext}"))
        if candidates:
            face_crop = sorted(candidates)[0]
            break
    if face_crop is None:
        error_exit(f"No image found in preproc_dir: {preproc_dir}", code=2)

    try:
        pil_img = Image.open(face_crop).convert("RGB")
    except Exception as e:
        error_exit(f"Cannot open image {face_crop}: {e}", code=1)

    pil_resized = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    orig_np = np.array(pil_resized, dtype=np.uint8)

    tfm = transforms.ToTensor()
    rgb_batch = tfm(pil_resized).unsqueeze(0).to(DEVICE)

    depth_path = first_file_containing(deca_dir, "depth_")
    norm_path  = first_file_containing(deca_dir, "normals_")
    rend_path  = first_file_containing(deca_dir, "orig_rendered_")

    if depth_path is None:
        error_exit(f"No depth_* found under {deca_dir}", code=2)
    if norm_path is None:
        error_exit(f"No normals_* found under {deca_dir}", code=2)
    if rend_path is None:
        error_exit(f"No orig_rendered_* found under {deca_dir}", code=2)

    depth_gray = load_depth_gray(depth_path)
    depth_t = torch.from_numpy(depth_gray).unsqueeze(0)

    def cv2_to_tensor(path: Path) -> torch.Tensor:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(path)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        arr = cv2.resize(arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        arr = arr.astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    rend_np_full = cv2_to_tensor(rend_path)
    norm_np_full = cv2_to_tensor(norm_path)

    orig_tensor_full = tfm(pil_resized)
    rgb_e = torch.abs(orig_tensor_full - rend_np_full).mean(dim=0, keepdim=True)

    depth_batched = depth_t.unsqueeze(0)
    pad = 2
    local = F.avg_pool2d(depth_batched, kernel_size=5, stride=1, padding=pad)[0]
    dep_e = torch.abs(depth_t - local)

    normals_batched = norm_np_full.unsqueeze(0)
    neigh = F.avg_pool2d(normals_batched, kernel_size=5, stride=1, padding=pad)[0]
    neigh = F.normalize(neigh, dim=0, eps=1e-6)
    dot = (norm_np_full * neigh).sum(dim=0).clamp(-1.0, 1.0)
    ang = torch.acos(dot)
    norm_e = ang.unsqueeze(0) / np.pi

    err_stack = torch.cat([rgb_e, dep_e, norm_e], dim=0).unsqueeze(0).to(DEVICE)

    if not MODEL_ROOT.is_dir():
        error_exit(f"Checkpoint root not found: {MODEL_ROOT}", code=1)

    fold_dirs = sorted(
        [d for d in MODEL_ROOT.iterdir() if d.is_dir() and d.name.startswith("fold_")],
        key=lambda p: int(p.name.split("_")[-1]),
    )
    if not fold_dirs:
        error_exit(f"No fold_* checkpoints under {MODEL_ROOT}", code=1)

    models = [load_fold_model(f) for f in fold_dirs]

    prob_real = ensemble_predict(rgb_batch, err_stack, models)
    label = "real" if prob_real >= 0.5 else "fake"
    confidence = prob_real if label == "real" else (1.0 - prob_real)

    saliency_rel: str | None = None
    try:
        saliency_map = compute_saliency(rgb_batch, err_stack, models)
        out_base = preproc_dir / f"{face_crop.stem}_saliency"
        saliency_rel = save_saliency(orig_np, saliency_map, out_base)
    except Exception as e:
        saliency_rel = None
        print(f"Warning: saliency generation failed → {e}", file=sys.stderr)

    def format_path_for_json(file_path: Path | None) -> str | None:
        if file_path is None or not file_path.exists():
            return None
        return str(file_path).replace(os.sep, "/")

    rendered_3d_rel = format_path_for_json(rend_path)
    depth_map_rel = format_path_for_json(depth_path)
    normals_map_rel = format_path_for_json(norm_path)

    result = {
        "image_is_real": label == "real",
        "confidence": confidence,
        "saliency": saliency_rel,
        "rendered_3d_image": rendered_3d_rel,
        "depth_map_image": depth_map_rel,
        "normals_map_image": normals_map_rel,
    }
    print(json.dumps(result))
    sys.stdout.flush()
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full-model inference (2-D + 3-D) with saliency")
    parser.add_argument("preproc_dir", help="Directory with the 2-D face crop (one image expected)")
    parser.add_argument("deca_dir",    help="Directory containing DECA outputs (depth_*, normals_*, orig_rendered_*)")
    args = parser.parse_args()

    try:
        main(Path(args.preproc_dir), Path(args.deca_dir))
    except SystemExit as e:
        sys.exit(e.code)
    except Exception as e:
        print(json.dumps({"error": f"Internal error: {e}"}))
        sys.exit(1)