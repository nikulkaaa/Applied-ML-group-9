# import threestudio
import os
import subprocess
import cv2
from preprocessing import process_images

import cv2                        # pip install opencv-python
import numpy as np                # pip install numpy
import torch                      # pip install torch torchvision
import torch.nn.functional as F
from scipy.spatial import Delaunay  # pip install scipy
import trimesh                    # pip install trimesh
from trimesh.remesh import subdivide  # pip install trimesh
from tqdm import tqdm             # pip install tqdm
import face_alignment             # pip install --upgrade face_alignment

midas_model = None
midas_transform = None

def _init_midas(device: torch.device):
    global midas_model, midas_transform
    if midas_model is None:
        midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        midas_model.to(device).eval()
        midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    return midas_model, midas_transform

def _estimate_depth_midas(bgr: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Runs MiDaS on the BGR image to produce a single‐channel
    uint8 depth map in [0,255].
    """
    # Lazy‐load MiDaS model & transform
    model, transform = _init_midas(device)

    # Convert to RGB and run the MiDaS transform
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inp = transform(img_rgb).to(device)

    # Forward pass
    with torch.no_grad():
        pred = model(inp)

    # Interpolate to original size and drop all singleton dims to [H,W]
    pred = F.interpolate(
        pred.unsqueeze(1),
        size=bgr.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    # Normalize to [0,1] then scale to [0,255]
    d = pred.cpu().numpy()
    d = d - d.min()
    if np.ptp(d) > 0:
        d = d / np.ptp(d)
    return (d * 255).astype(np.uint8)


def _best_landmark_mode() -> "face_alignment.LandmarksType":
    L = face_alignment.LandmarksType
    if hasattr(L, 'THREE_D'):
        return L.THREE_D
    if hasattr(L, 'TWO_HALF_D'):
        logging.warning('3D landmarks not supported; using 2.5D instead.')
        return L.TWO_HALF_D
    logging.warning('2.5D landmarks not supported; using 2D instead.')
    return L.TWO_D


def _init_face_alignment(device_str: str) -> face_alignment.FaceAlignment:
    return face_alignment.FaceAlignment(
        _best_landmark_mode(),
        flip_input=False,
        device=device_str,
    )


def _depth_map(verts: np.ndarray, faces: np.ndarray, H: int, W: int) -> np.ndarray:
    z = verts[:, 2] if verts.shape[1] == 3 else np.zeros(len(verts))
    z_norm = z - z.min()
    if np.ptp(z_norm) > 0:
        z_norm /= np.ptp(z_norm)
    z_img = (z_norm * 255).astype(np.uint8)
    depth = np.zeros((H, W), dtype=np.uint8)
    for tri in faces:
        pts = np.round(verts[tri, :2]).astype(np.int32)
        pts = np.clip(pts, [0, 0], [W - 1, H - 1])
        cv2.fillConvexPoly(depth, pts, int(z_img[tri].mean()))
    return cv2.medianBlur(depth, 5)


def _normal_map(verts: np.ndarray, faces: np.ndarray, H: int, W: int) -> np.ndarray:
    mesh_tmp = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    normals = mesh_tmp.vertex_normals
    enc = ((normals + 1.0) * 0.5 * 255).astype(np.uint8)
    nm = np.zeros((H, W, 3), dtype=np.uint8)
    for tri in faces:
        pts = np.round(verts[tri, :2]).astype(np.int32)
        pts = np.clip(pts, [0, 0], [W - 1, H - 1])
        col = enc[tri].mean(axis=0).astype(np.uint8)
        cv2.fillConvexPoly(nm, pts, (int(col[0]), int(col[1]), int(col[2])))
    return cv2.GaussianBlur(nm, (5,5), 0)


def _sample_vertex_colors(verts: np.ndarray, img_rgb: np.ndarray) -> np.ndarray:
    H, W, _ = img_rgb.shape
    cols = []
    for x,y,*_ in verts:
        xi = np.clip(int(round(x)), 0, W-1)
        yi = np.clip(int(round(y)), 0, H-1)
        r,g,b = img_rgb[yi, xi]
        cols.append([r,g,b,255])
    return np.array(cols, dtype=np.uint8)


def _rasterise_reprojection(mesh: trimesh.Trimesh, H: int, W: int):
    verts, faces = mesh.vertices, mesh.faces
    # color
    vc = mesh.visual.vertex_colors[:, :3]
    col_img = np.zeros((H,W,3),dtype=np.uint8)
    # depth
    z = verts[:,2] if verts.shape[1]==3 else np.zeros(len(verts))
    z_n = z - z.min()
    if np.ptp(z_n)>0: z_n/=np.ptp(z_n)
    z_i = (z_n*255).astype(np.uint8)
    dep_img = np.zeros((H,W),dtype=np.uint8)
    # normals
    vn = trimesh.Trimesh(vertices=verts,faces=faces,process=False).vertex_normals
    enc_n = ((vn+1)*0.5*255).astype(np.uint8)
    norm_img = np.zeros((H,W,3),dtype=np.uint8)

    for tri in faces:
        pts = np.round(verts[tri,:2]).astype(np.int32)
        pts = np.clip(pts, [0,0],[W-1,H-1])
        # color
        rc,gc,bc = vc[tri].mean(axis=0).astype(np.uint8)
        cv2.fillConvexPoly(col_img, pts, (int(bc),int(gc),int(rc)))
        # depth
        cv2.fillConvexPoly(dep_img, pts, int(z_i[tri].mean()))
        # normals
        nx,ny,nz = enc_n[tri].mean(axis=0).astype(np.uint8)
        cv2.fillConvexPoly(norm_img, pts, (int(nx),int(ny),int(nz)))

    dep_img = cv2.medianBlur(dep_img,5)
    norm_img = cv2.GaussianBlur(norm_img,(5,5),0)
    return col_img, dep_img, norm_img


def _compute_error_maps(
    orig_bgr: np.ndarray,
    reproj_bgr: np.ndarray,
    orig_depth: np.ndarray,
    reproj_depth: np.ndarray,
    orig_normals: np.ndarray,
    reproj_normals: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pixel‐wise error maps:
      - RGB error: abs(original vs reprojected)
      - Depth error: abs(MiDaS_depth vs reprojected_depth)
      - Normal error: angle between original normals vs reproj normals
    Returns (err_rgb, err_depth, err_norm), each as uint8 images.
    """
    # RGB error
    err_rgb = cv2.absdiff(orig_bgr, reproj_bgr)

    # Depth error
    err_depth = cv2.absdiff(orig_depth, reproj_depth)

    # Normal‐angle error
    o = orig_normals.astype(np.float32)
    r = reproj_normals.astype(np.float32)

    ox = (o[:, :, 0] / 255.0) * 2 - 1
    oy = (o[:, :, 1] / 255.0) * 2 - 1
    oz = (o[:, :, 2] / 255.0) * 2 - 1

    rx = (r[:, :, 0] / 255.0) * 2 - 1
    ry = (r[:, :, 1] / 255.0) * 2 - 1
    rz = (r[:, :, 2] / 255.0) * 2 - 1

    dot = np.clip(ox * rx + oy * ry + oz * rz, -1, 1)
    angles = np.degrees(np.arccos(dot))
    err_norm = ((angles / 180.0) * 255).astype(np.uint8)

    return err_rgb, err_depth, err_norm



def _triangulate(pts2d: np.ndarray) -> np.ndarray:
    return Delaunay(pts2d).simplices


def reconstruct_single_image(
    img_path: Path,
    fa: face_alignment.FaceAlignment,
    device: torch.device,
    subdivisions: int = 0
):
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        logging.warning("Could not read %s – skipping", img_path)
        return None
    H,W = bgr.shape[:2]
    # 1) MiDaS “expected” depth we calcualted as ground truth
    depth_midas = _estimate_depth_midas(bgr, device)

    # face→verts
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    lids = fa.get_landmarks_from_image(rgb)
    if not lids:
        logging.warning("No face in %s – skipping", img_path)
        return None
    verts = lids[0]
    if verts.shape[1]==2:
        verts = np.hstack([verts, np.zeros((len(verts),1))])
    faces = _triangulate(verts[:,:2])

    # build & subdivide mesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    for _ in range(subdivisions):
        v2,f2 = subdivide(mesh.vertices, mesh.faces)
        mesh = trimesh.Trimesh(vertices=v2, faces=f2, process=False)

    # maps from mesh
    depth_rec   = _depth_map(mesh.vertices, mesh.faces, H, W)
    normal_map  = _normal_map(mesh.vertices, mesh.faces, H, W)
    vcols       = _sample_vertex_colors(mesh.vertices, rgb)
    mesh.visual.vertex_colors = vcols

    # reproject
    rc, rd, rn = _rasterise_reprojection(mesh, H, W)

    return mesh, depth_midas, depth_rec, normal_map, rc, rd, rn


def process_dataset(
    data_root: Path,
    output_root: Path,
    device_str: str,
    overwrite: bool = False,
    subdivisions: int = 0
):
    fa     = _init_face_alignment(device_str)
    device = torch.device(device_str if device_str=="cpu" or torch.cuda.is_available() else "cpu")

    paths = ([data_root] if data_root.is_file()
             else [*data_root.rglob("*.png"), *data_root.rglob("*.jpg"), *data_root.rglob("*.jpeg")])
    logging.info(f"Found {len(paths)} image(s) in {data_root}")

    for img_path in tqdm(paths, desc="Reconstructing faces"):
        rel = img_path.relative_to(data_root) if data_root.is_dir() else img_path.name
        out = output_root / rel.parent if data_root.is_dir() else output_root
        out.mkdir(parents=True, exist_ok=True)
        stem = img_path.stem

        files = {
            "midas" : out / f"{stem}_midas_depth.png",
            "obj"   : out / f"{stem}.obj",
            "ply"   : out / f"{stem}_colored.ply",
            "depth" : out / f"{stem}_depth.png",
            "normal": out / f"{stem}_normal.png",
            "rcol"  : out / f"{stem}_reproj_color.png",
            "rdep"  : out / f"{stem}_reproj_depth.png",
            "rnorm" : out / f"{stem}_reproj_normals.png",
            "e_rgb" : out / f"{stem}_error_rgb.png",
            "e_dep" : out / f"{stem}_error_depth.png",
            "e_nrm" : out / f"{stem}_error_normal.png",
        }
        if not overwrite and all(p.exists() for p in files.values()):
            continue

        res = reconstruct_single_image(img_path, fa, device, subdivisions)
        if res is None:
            continue
        mesh, d_midas, d_rec, nmap, rc, rd, rn = res

        # save reference & reconstructions
        cv2.imwrite(str(files["midas"]), d_midas)
        mesh.export(str(files["obj"]))
        mesh.export(str(files["ply"]))
        cv2.imwrite(str(files["depth"]), d_rec)
        cv2.imwrite(str(files["normal"]), nmap)
        cv2.imwrite(str(files["rcol"]), rc)
        cv2.imwrite(str(files["rdep"]), rd)
        cv2.imwrite(str(files["rnorm"]), rn)

        # compute & save errors
        orig_bgr = cv2.imread(str(img_path))
        orig_normals = cv2.imread(str(files["normal"]))  # load the rasterised normal map

        err_rgb, err_depth, err_norm = _compute_error_maps(
            orig_bgr,
            rc, # reproj_color
            d_midas, # MiDaS depth
            rd, # reproj_depth
            orig_normals, # original normal map
            rn # reproj_normals
        )

        cv2.imwrite(str(files["e_rgb"]), err_rgb)
        cv2.imwrite(str(files["e_dep"]), err_depth)
        cv2.imwrite(str(files["e_nrm"]), err_norm)


def parse_args():
    p = argparse.ArgumentParser(
        description="3D face reconstruction + MiDaS depth + reprojection + error maps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_path", type=Path, required=True)
    p.add_argument("--out_dir",  type=Path, default=Path("recon_output"))
    p.add_argument("--device",   choices=["cuda","cpu"], default="cuda")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--subdivide", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    process_dataset(
        data_root  = args.data_path,
        output_root= args.out_dir,
        device_str = args.device,
        overwrite  = args.overwrite,
        subdivisions = args.subdivide,
    )

if __name__ == "__main__":
    main()
