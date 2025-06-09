import os
import sys
import argparse
import cv2
from tqdm import tqdm
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif'}


def collect_image_paths(root: str, recursive: bool = True):
    """
    Walk root (if it's a directory) and return a sorted list of all image files
    with extensions in IMG_EXT.  If root is a single file, return [root].
    """
    root = os.path.abspath(root)
    if os.path.isfile(root) and os.path.splitext(root)[1].lower() in IMG_EXT:
        return [root]

    hits = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMG_EXT:
                hits.append(os.path.join(dirpath, f))
    return sorted(hits)


def ensure_dir(path: str):
    """Create directory path if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def save_img(img: np.ndarray, out_folder: str, filename: str):
    """
    Save imgas JPEG under "out_folder/filename".
    """
    out_path = os.path.join(out_folder, filename)
    cv2.imwrite(out_path, img)


def depth_to_normal(depth_img: np.ndarray) -> np.ndarray:
    """
    Fallback Sobel-based normal estimation from a single-channel depth image.
    Returns a uint8 normal map.
    """
    dx = cv2.Sobel(depth_img, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(depth_img, cv2.CV_32F, 0, 1, ksize=3)
    nz = np.ones_like(depth_img, dtype=np.float32)
    nx, ny = -dx, -dy
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    nx /= norm
    ny /= norm
    nz /= norm
    normal = np.stack([nx, ny, nz], axis=-1) 
    normal = ((normal + 1.0) * 0.5 * 255.0).astype(np.uint8)
    return normal


def main(args):
    device = args.device
    input_root = args.inputpath
    root_out = args.savefolder

    all_paths = collect_image_paths(input_root, recursive=not args.no_recursive)
    if not all_paths:
        raise RuntimeError(f"No images found in {input_root!r}")

    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex
    deca = DECA(config=deca_cfg, device=device)

    out_depth    = ensure_dir(os.path.join(root_out, 'Depth'))
    out_normals  = ensure_dir(os.path.join(root_out, 'Normals'))
    out_rendered = ensure_dir(os.path.join(root_out, 'Rendered'))
    out_orig     = ensure_dir(os.path.join(root_out, 'OriginalRendered'))

    for img_path in tqdm(all_paths, desc="processing"):
        basename = os.path.splitext(os.path.basename(img_path))[0]

        # Load & preprocess
        td = datasets.TestData(img_path, iscrop=args.iscrop,
                                 face_detector=args.detector, sample_step=1)
        data = td[0]
        images = data['image'].to(device)[None, ...]

        with torch.no_grad():
            codedict = deca.encode(images)
            opdict, visdict = deca.decode(codedict)

            orig_vis = None
            if args.render_orig:
                tform = torch.inverse(data['tform'][None]).transpose(1, 2).to(device)
                orig_img = data['original_image'][None].to(device)
                _, orig_vis = deca.decode(codedict, render_orig=True,
                                          original_image=orig_img, tform=tform)
        if args.saveDepth:
            depth_t = deca.render.render_depth(opdict['trans_verts']).repeat(1, 3, 1, 1)
            depth_img = util.tensor2image(depth_t[0])  # H×W×3 uint8
            save_img(depth_img, out_depth, f'depth_{basename}.jpg')

        normals = None
        method = None

        if hasattr(deca.render, 'render_normal'):
            try:
                normals_t = deca.render.render_normal(opdict['trans_verts']).repeat(1, 3, 1, 1)
                normals = util.tensor2image(normals_t[0])
                method = 'renderer'
            except Exception:
                normals = None

        if normals is None:
            for key in ('normals', 'normal_images', 'normals_images', 'detail_normal_images'):
                if key in visdict:
                    normals = util.tensor2image(visdict[key][0])
                    method = f'visdict[{key}]'
                    break

        if normals is None:
            gray = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            normals = depth_to_normal(gray)
            method = 'depth-grad'

        save_img(normals, out_normals, f'normals_{basename}.jpg')
        if args.verbose:
            print(f'[info] normals for {basename} via {method}')

        if 'rendered_images' in visdict:
            rendered_img = util.tensor2image(visdict['rendered_images'][0])
            save_img(rendered_img, out_rendered, f'rendered_{basename}.jpg')

        if args.render_orig and orig_vis and 'rendered_images' in orig_vis:
            orig_rend = util.tensor2image(orig_vis['rendered_images'][0])
            save_img(orig_rend, out_orig, f'orig_rendered_{basename}.jpg')

    print(f'✔ Done – results are in "{root_out}"')


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="DECA – simple reconstruction into flat folders"
    )
    p.add_argument('-i', '--inputpath', required=True, help='Image or folder of images')
    p.add_argument('-s', '--savefolder', default='deca_output',
                   help='Root folder to write Depth/, Normals/, Rendered/, OriginalRendered/')
    p.add_argument('--device', default='cuda',
                   help='Device for PyTorch (e.g., "cpu" or "cuda")')
    p.add_argument('--iscrop', default=True,
                   type=lambda x: x.lower() in {'true', '1'},
                   help='Whether to crop faces (True/False)')
    p.add_argument('--detector', default='fan',
                   help='Face detector for TestData (e.g., "fan" or "sfd")')
    p.add_argument('--no_recursive', action='store_true',
                   help='If set, do not recurse into subfolders of inputpath')
    p.add_argument('--rasterizer_type', default='standard',
                   help='DECA rasterizer type (e.g., "standard" or "pytorch3d")')
    p.add_argument('--render_orig', default=True,
                   type=lambda x: x.lower() in {'true', '1'},
                   help='Whether to render the original textured face')
    p.add_argument('--saveDepth', default=True,
                   type=lambda x: x.lower() in {'true', '1'},
                   help='Whether to save depth maps')
    p.add_argument('--useTex', default=False,
                   type=lambda x: x.lower() in {'true', '1'},
                   help='Whether to extract texture')
    p.add_argument('--extractTex', default=True,
                   type=lambda x: x.lower() in {'true', '1'},
                   help=argparse.SUPPRESS)
    p.add_argument('--verbose', action='store_true',
                   help='Print additional info about normal computation')
    args = p.parse_args()

    try:
        main(args)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
