from __future__ import annotations
"""
Full-model inference script.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import cv2

MODEL_ROOT = Path("checkpoints")
IMG_SIZE = 224
IMG_EXTS = (".png", ".jpg", ".jpeg")
UPLOADS_ROOT = Path("uploads")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UPLOADS_ROOT.mkdir(parents=True, exist_ok=True)

def error_exit(msg: str, code: int = 2):
    """Emit JSON error and terminate with ``code`` (default 2)."""
    print(json.dumps({"error": msg}))
    sys.exit(code)


def first_file_containing(dirpath: Path, substring: str) -> Path | None:
    for p in dirpath.rglob(f"*{substring}*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            return p
    return None


def load_tensor_img(path: Path, rgb: bool = True) -> torch.Tensor:
    flag = cv2.IMREAD_COLOR if rgb else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f"Could not load: {path}")
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    if not rgb:
        img = img[..., None]
    return torch.from_numpy(img).permute(2, 0, 1)


# Error map functions
def rgb_diff(orig: torch.Tensor, rend: torch.Tensor) -> torch.Tensor:
    return torch.abs(orig - rend).mean(0, keepdim=True)

def depth_inconsistency(depth: torch.Tensor, k: int = 5) -> torch.Tensor:
    pad = k // 2
    # pool per-channel
    local = F.avg_pool2d(depth.unsqueeze(0), k, stride=1, padding=pad)[0]
    dep_e = torch.abs(depth - local)
    # now average channels exactly as training
    if dep_e.shape[0] > 1:
        dep_e = dep_e.mean(0, keepdim=True)
    return dep_e


def normal_angle_error(normals: torch.Tensor, k: int = 5) -> torch.Tensor:
    pad = k // 2
    neigh = F.avg_pool2d(normals.unsqueeze(0), k, stride=1, padding=pad)[0]
    neigh = F.normalize(neigh, dim=0, eps=1e-6)
    ang = torch.acos(torch.clamp((normals * neigh).sum(0), -1.0, 1.0))
    return ang.unsqueeze(0) / np.pi

def compute_saliency(
    rgb: torch.Tensor,
    err: torch.Tensor,
    model: torch.nn.Module,
    cls_idx: int
) -> np.ndarray:
    rgb_ = rgb.clone().detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(rgb_, err)
    logits[0, cls_idx].backward()
    sal = rgb_.grad.detach().abs().mean(1)[0]
    sal = sal / (sal.max() + 1e-6)
    return sal.cpu().numpy()


def save_saliency(orig: np.ndarray, smap: np.ndarray, out_base: Path) -> str:
    """Overlay heat-map on top of orig image and then save it."""
    out_path = out_base.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap = cv2.resize(smap, (orig.shape[1], orig.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = orig.astype(np.float32) * 0.6 + heatmap.astype(np.float32) * 0.4
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    return str(out_path.resolve())

def _load_single_model(pt: Path):
    return torch.jit.load(str(pt), map_location=DEVICE)

def load_models() -> List[torch.nn.Module]:
    models: List[torch.nn.Module] = []
    if not MODEL_ROOT.is_dir():
        error_exit(f"Checkpoint directory not found: {MODEL_ROOT}")
    for fold_dir in sorted(MODEL_ROOT.glob("fold_*")):
        pt = fold_dir / "best.pt"
        if pt.is_file():
            m = _load_single_model(pt).to(DEVICE)
            m.eval()
            models.append(m)
    if not models:
        error_exit(f"No models found under {MODEL_ROOT}")
    return models

def predict(models: List[torch.nn.Module], rgb: torch.Tensor, err: torch.Tensor) -> Tuple[float, float]:
    """Return the probabilties of fake or real averaged over all models (folds)."""
    with torch.no_grad():
        ps = [F.softmax(m(rgb, err), dim=1) for m in models]
        prob = torch.stack(ps).mean(0)[0]
        return float(prob[0]), float(prob[1])

def main(preproc_dir: Path, deca_dir: Path):
    face_crop = None
    for ext in IMG_EXTS:
        cand = list(preproc_dir.glob(f"*{ext}"))
        if cand:
            face_crop = sorted(cand)[0]
            break
    if face_crop is None:
        error_exit(f"No image found in preproc_dir: {preproc_dir}")

    depth_p = first_file_containing(deca_dir, "depth_")
    norm_p  = first_file_containing(deca_dir, "normals_")
    rend_p  = first_file_containing(deca_dir, "orig_rendered_")
    if None in (depth_p, norm_p, rend_p):
        error_exit("Missing one of: depth_*, normals_*, orig_rendered_* in the DECA output")

    rgb_orig = load_tensor_img(face_crop, rgb=True)
    depth_orig = load_tensor_img(depth_p, rgb=False)
    normals_orig = load_tensor_img(norm_p, rgb=True)
    rend_orig = load_tensor_img(rend_p, rgb=True)

    H, W = rgb_orig.shape[1], rgb_orig.shape[2]

    def _resize(t: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            t.unsqueeze(0), size=(H, W),
            mode="bilinear", align_corners=False
        )[0]

    depth_r = _resize(depth_orig)
    normals_r = _resize(normals_orig)
    rend_r = _resize(rend_orig)

    # Compute error maps at original‐crop resolution
    rgb_e = rgb_diff(rgb_orig, rend_r)
    dep_e = depth_inconsistency(depth_r)
    norm_e = normal_angle_error(normals_r)

    # Build the error stack
    err_stack = torch.cat([rgb_e, dep_e, norm_e], dim=0)

    rgb_input = rgb_orig.unsqueeze(0).to(DEVICE)
    err_input = err_stack.unsqueeze(0).to(DEVICE)

    # Ensemble prediction
    models = load_models()
    prob_fake, prob_real = predict(models, rgb_input, err_input)
    is_fake    = prob_fake >= prob_real
    confidence = prob_fake if is_fake else prob_real

    # Saliency
    saliency_path = None
    try:
        orig_np = (rgb_orig.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        cls_idx = 0 if is_fake else 1
        sal = compute_saliency(rgb_input, err_input, models[0], cls_idx)
        out_base = preproc_dir / f"{face_crop.stem}_saliency"
        saliency_path = save_saliency(orig_np, sal, out_base)
    except Exception as e:
        print(f"Warning: Saliency failed: {e}", file=sys.stderr)

    # JSON output
    def _abs(p): return str(Path(p).resolve()) if p else None
    result = {
        "image_is_real":     not is_fake,
        "confidence":        confidence,
        "saliency":          _abs(saliency_path),
        "rendered_3d_image": _abs(rend_p),
        "depth_map_image":   _abs(depth_p),
        "normals_map_image": _abs(norm_p),}
    print(json.dumps(result))
    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full-model inference (2D + 3D) with saliency and fold-ensemble")
    parser.add_argument("preproc_dir", help="Directory with the 2D face crop")
    parser.add_argument("deca_dir", help="Directory containing DECA outputs (depth_, normals_, orig_rendered_)")
    args = parser.parse_args()
    try:
        main(Path(args.preproc_dir), Path(args.deca_dir))
    except SystemExit as e:
        sys.exit(e.code)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(json.dumps({"error": f"Internal error: {e}"}))
        sys.exit(1)
