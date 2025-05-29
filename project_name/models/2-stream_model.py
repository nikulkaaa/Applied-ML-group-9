"""
2-stream Model (v1.0) ran on PYTHON 3.10.8

RUN FILE WITH THESE ARGS: --preproc-root project_name/data/preprocessed_dataset/preprocessed_eye_align --recon-root project_name/data/3DRecon
libraries to import: reqs_2stream.txt
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import warnings
import matplotlib.pyplot as plt
import itertools
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    accuracy_score)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# supressing warnings since they are annoying 
warnings.filterwarnings("ignore")


"""
Helper functions 
"""

def _pil_or_cv(path: Path) -> np.ndarray:
    """
    Read an image from a path if it exists!
    """
    buf = np.fromfile(str(path), dtype=np.uint8)
    arr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if arr is None: # file does not exist
        raise FileNotFoundError(path)
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr

def rgb_diff(orig: torch.Tensor, rend: torch.Tensor) -> torch.Tensor:
    """
    Calculate the mean absolute difference between two RGB images.
    """
    return torch.abs(orig - rend).mean(0, keepdim=True)

def depth_inconsistency(depth: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    Calculate the local depth inconsistency using a sliding window approach.
    """
    pad = k // 2
    local = F.avg_pool2d(depth.unsqueeze(0), k, stride = 1, padding=pad)[0]
    return torch.abs(depth - local)

def normal_angle_error(normals: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    Calculate the local normal angle difference using a sliding window approach.
    """
    pad = k // 2
    neigh = F.avg_pool2d(normals.unsqueeze(0), k, stride = 1, padding=pad)[0]
    neigh = F.normalize(neigh, dim=0, eps=1e-6)
    ang = torch.acos(torch.clamp((normals * neigh).sum(0), -1.0, 1.0))
    return ang.unsqueeze(0) / math.pi


"""
Functions that handle loading of the dataset and further
"""

class DeepFake3DDataset(Dataset):
    exts = (".png", ".jpg", ".jpeg")

    def __init__(
        self,
        preproc_root: Path,
        recon_root: Path,
        split: str,
        transform: Callable | None = None,
        cache_dir: Path | None = None) -> None:
        """
        Initialize the DeepFake3DDataset object.
        """

        super().__init__()
        self.transform = transform
        self.cache_dir = cache_dir

        split_map = {
            "train": ("Train", "train"),
            "val": ("Validation", "validation"),
            "test": ("Test", "test")}
        orig_split, recon_split = split_map[split.lower()]

        self.samples: List[Dict[str, Path | int]] = []
        for cls_idx, cls_name in enumerate(["Real", "Fake"]):
            orig_dir = preproc_root / orig_split / cls_name
            dirs = {
                "depth": recon_root / "Depth" / recon_split / cls_name,
                "norm": recon_root / "Normals" / recon_split / cls_name,
                "rend": recon_root / "OriginalRendered" / recon_split / cls_name}
            for orig_path in itertools.chain.from_iterable(
                orig_dir.rglob(f"*{e}") for e in self.exts
            ):
                stem = orig_path.stem
                def _match(dir_: Path) -> Path | None:
                    for e in self.exts:
                        p = dir_ / f"{stem}{e}"
                        if p.exists(): return p
                    m = re.search(r"(\d+)$", stem)
                    if m:
                        tok = m.group(1)
                        hits = list(dir_.glob(f"*{tok}*"))
                        if hits: return hits[0]
                    return None

                paths = {k: _match(d) for k, d in dirs.items()}
                if any(v is None for v in paths.values()):
                    continue
                self.samples.append({
                    "orig": orig_path,
                    "rend": paths["rend"],
                    "depth": paths["depth"],
                    "norm": paths["norm"],
                    "label": cls_idx})

    def __len__(self) -> int:
        """Returns the number of samples in the dataset"""
        return len(self.samples)

    def _load_tensor(self, path: Path, rgb: bool) -> torch.Tensor:
        """
        Loads an image from a file path and converts it to a tensor
        """
        img = _pil_or_cv(path)
        if rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)

    def _resize(self, t: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        """
        Resizes a tensor to the given height and width.
        """
        if t.shape[-2:] == hw: return t
        return F.interpolate(t.unsqueeze(0), size=hw, mode="bilinear", align_corners=False)[0]

    def _load_modalities(self, idx: int):
        """
        Loads all modalities for a given sample index and resizes them to the same
        height and width as the original image. The modalities are:

        - orig: Original image (RGB)
        - rend: Original rendered image (RGB)
        - depth: Depth map (Grayscale)
        - normals: Normal map (RGB)

        The last element of the returned tuple is the label of the sample
        """
        m = self.samples[idx]
        orig = self._load_tensor(m["orig"], True)
        rend = self._load_tensor(m["rend"], True)
        depth = self._load_tensor(m["depth"], False)
        normals = self._load_tensor(m["norm"], True)
        hw = orig.shape[-2:]
        rend = self._resize(rend,   hw)
        depth = self._resize(depth,  hw)
        normals = self._resize(normals,hw)
        return orig, rend, depth, normals, torch.tensor(m["label"], dtype=torch.long)

    @torch.no_grad()
    def _errors(self, orig, rend, depth, normals):
        """
        Computes the errors for the given modalities of a sample:

        - rgb_e: RGB difference between the original and rendered images
        - dep_e: Depth inconsistency between the original and rendered images
        - norm_e: Normal angle error between the original and rendered images

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The errors for the RGB, depth and normal modalities
        """
        rgb_e = rgb_diff(orig, rend)
        dep_e = depth_inconsistency(depth)
        if dep_e.shape[0] > 1:
            dep_e = dep_e.mean(0, keepdim = True)
        norm_e = normal_angle_error(normals)
        return rgb_e, dep_e, norm_e

    def __getitem__(self, idx: int):
        """
        Retrieves and processes the data for a given sample index.
        1. Checks if the data for the given index is cached
        2. Loads modalities and computes errors if not cached
        3. Applies transformations if specified
        4. Returns the processed data and label
        """
        cache_f = self.cache_dir / f"{idx}.npz" if self.cache_dir else None
        if cache_f and cache_f.exists():
            d = np.load(cache_f)
            orig, rend, depth, normals = (
                torch.from_numpy(d[k]) for k in ("orig", "rend", "depth", "normals"))
            rgb_e, dep_e, norm_e = (
                torch.from_numpy(d[k]) for k in ("rgb_e", "dep_e", "norm_e"))
            label = torch.tensor(int(d["label"]))
        else:
            orig, rend, depth, normals, label = self._load_modalities(idx)
            rgb_e, dep_e, norm_e = self._errors(orig, rend, depth, normals)
            if cache_f:
                cache_f.parent.mkdir(parents = True, exist_ok = True)
                np.savez_compressed(
                    cache_f,
                    orig=orig.numpy(),
                    rend=rend.numpy(),
                    depth=depth.numpy(),
                    normals=normals.numpy(),
                    rgb_e=rgb_e.numpy(),
                    dep_e=dep_e.numpy(),
                    norm_e=norm_e.numpy(),
                    label=label.item())
        if self.transform:
            depth_ch = depth.shape[0]
            stack = torch.cat([orig, rend, depth, normals, rgb_e, dep_e, norm_e], 0)
            stack = stack.numpy().transpose(1,2,0)
            stack = self.transform(image=stack)["image"].transpose(2,0,1)
            stack = torch.from_numpy(stack)
            secs = [3,3,depth_ch,3,1,1,1]
            if sum(secs) != stack.shape[0]:
                depth_ch = stack.shape[0] - 12
                secs = [3,3,depth_ch,3,1,1,1]
            orig, rend, depth, normals, rgb_e, dep_e, norm_e = torch.split(stack, secs)
        err_stack = torch.cat([rgb_e, dep_e, norm_e], 0)
        return orig, rend, err_stack, label


"""
Model
"""

class ConvBlock(nn.Module):
    def __init__(self, inp: int, out: int, drop: float):
        """
        Initializes a ConvBlock module.
        
        Args:
        - inp (int): Number of input channels.
        - out (int): Number of output channels.
        - drop (float): Dropout probability (0. to disable).
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, out, 3, padding = 1, bias = False),
            nn.BatchNorm2d(out), nn.ReLU(inplace = True),
            nn.Conv2d(out, out, 3, padding = 1, bias = False),
            nn.BatchNorm2d(out), nn.ReLU(inplace = True))
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(drop) if drop else nn.Identity()
    def forward(self, x):
        """
        Forward pass through the ConvBlock module.
        """
        return self.drop(self.pool(self.conv(x)))

class StreamCNN(nn.Module):
    def __init__(self, in_ch: int):
        """
        Initializes a StreamCNN module.

        Args:
        - in_ch (int): Number of input channels.
        """
        super().__init__()
        chans = [32,64,128,256,256]
        drops = [0.0, 0.0, 0.3, 0.0, 0.3]
        layers, c = [], in_ch
        for n,p in zip(chans,drops):
            layers.append(ConvBlock(c, n, p)); c=n
        self.blocks = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """
        Forward pass through the StreamCNN module.
        """
        return self.gap(self.blocks(x)).flatten(1)

class TwoStreamDetector(nn.Module):
    def __init__(self, errors_only: bool=False):
        """
        Initializes a TwoStreamDetector module.

        Args:
        - errors_only (bool, optional): If True, only use the error stream. Defaults to False.
        """
        super().__init__()
        self.errors_only = errors_only
        if not errors_only:
            self.rgb_stream = StreamCNN(3)
        self.err_stream = StreamCNN(3)
        self.head = nn.Sequential(
            nn.Linear(256 if errors_only else 512, 512, bias=False),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 128, bias = False), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(128, 2))
    def forward(self, rgb, err):
        """
        Forward pass through the TwoStreamDetector module.
        """
        feat = (self.err_stream(err) if self.errors_only
                else torch.cat([self.rgb_stream(rgb), self.err_stream(err)], 1))
        return self.head(feat)


"""
Training utilities
"""

def compute_eer(y_true, scores):
    """
    Computes the Equal Error Rate (EER) from given true labels and prediction scores.
    """
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    return float(np.nanmin(np.maximum(fpr, fnr)))

class Trainer:
    def __init__(self, model, opt, sched, device, save_dir: Path):
        """
        Initializes the Trainer object with the model, optimizer, scheduler, device, and save directory.
        """
        self.model, self.opt, self.sched, self.dev = model.to(device), opt, sched, device
        self.best_auc = 0.0
        self.dir = save_dir; self.dir.mkdir(exist_ok = True)
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "val_auc": [], "val_eer": [], "val_f1": []}

    def _iter(self, loader: DataLoader, train: bool):
        """
        Performs a single iteration over the given data loader, updating the model weights if in training mode,
        and computing the loss and probabilities for the current batch.
        """
        if train:  self.model.train()
        else:      self.model.eval()
        losses, probs, ys = [], [], []
        pbar = tqdm(loader, desc = "train" if train else "eval", leave = False)
        for rgb, _, err, y in pbar:
            rgb, err, y = rgb.to(self.dev), err.to(self.dev), y.to(self.dev)
            ctx = (torch.cuda.amp.autocast() if self.scaler and train else nullcontext())
            with ctx:
                logit = self.model(rgb,err)
                loss = F.cross_entropy(logit,y)
            if train:
                self.opt.zero_grad(set_to_none = True)
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.opt.step()
            losses.append(loss.item())
            probs.append(F.softmax(logit,1)[:,1].detach().cpu().numpy())
            ys.append(y.cpu().numpy())
            pbar.set_postfix(loss = f"{np.mean(losses):.4f}")
        return np.concatenate(probs), np.concatenate(ys), np.array(losses)

    def fit(self, tr_loader, vl_loader, epochs):
        """
        Fit the model to the given training data loader for the specified number of epochs.
        """
        for epoch in range(1, epochs+1):
            # train
            tr_p, tr_y, tr_l = self._iter(tr_loader, True)
            tr_acc = ((tr_p > 0.5).astype(int)==tr_y).mean()
            if self.sched:  self.sched.step()

            # val
            vl_p, vl_y, vl_l = self._iter(vl_loader, False)
            vl_acc = ((vl_p> 0.5).astype(int)==vl_y).mean()
            auc = roc_auc_score(vl_y, vl_p)
            eer = compute_eer(vl_y, vl_p)
            f1 = f1_score(vl_y, (vl_p > 0.5).astype(int))

            # log it
            self.history["train_loss"].append(tr_l.mean())
            self.history["train_acc"].append(tr_acc)
            self.history["val_loss"].append(vl_l.mean())
            self.history["val_acc"].append(vl_acc)
            self.history["val_auc"].append(auc)
            self.history["val_eer"].append(eer)
            self.history["val_f1"].append(f1)

            print(
                f"Epoch {epoch:02d}  "
                f"TRAIN: loss={tr_l.mean():.4f} acc={tr_acc:.4f}  "
                f"VALIDATION: loss={vl_l.mean():.4f} acc={vl_acc:.4f}  "
                f"AUC={auc:.4f} EER={eer:.4f} F1={f1:.4f}")

            if auc >self.best_auc:
                self.best_auc=auc
                torch.save(self.model.state_dict(), self.dir/"best.pt") # SAVE BEST MODEL!~
        return self.history


"""
Our main script (pipeline) and command-line interface so we can adjust epochs/batch/lr etc
"""

def parse_args():
    """
    Parses command-line arguments for configuring the training process.

    Returns:
        argparse.Namespace: A namespace containing parsed arguments including:
            - preproc_root (str): Path to the preprocessing root directory (required).
            - recon_root (str): Path to the reconstruction root directory (required).
            - epochs (int): Number of training epochs (default: 20).
            - batch (int): Batch size for training (default: 32).
            - lr (float): Learning rate for optimization (default: 1e-3).
            - cache_dir (str, optional): Directory for caching data.
            - save_dir (str): Directory to save checkpoints (default: "checkpoints").
    """
    p = argparse.ArgumentParser()
    p.add_argument("--preproc-root", required=True)
    p.add_argument("--recon-root",   required=True)
    p.add_argument("--epochs",   type=int, default=20)
    p.add_argument("--batch",    type=int, default=32)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--cache-dir")
    p.add_argument("--save-dir", default="checkpoints")
    return p.parse_args()

def build_loaders(args):
    """
    Builds and returns data loaders for training and validation datasets.
    """
    aug = A.Compose([
        A.HorizontalFlip(0.5),
        A.Rotate(limit=5, border_mode=cv2.BORDER_REFLECT_101, p=0.3),
        A.RandomBrightnessContrast(0.1,0.1,p=0.3)])
    cache = Path(args.cache_dir) if args.cache_dir else None
    tr_ds = DeepFake3DDataset(Path(args.preproc_root), Path(args.recon_root), "train", aug, cache)
    vl_ds = DeepFake3DDataset(Path(args.preproc_root), Path(args.recon_root), "val", None, cache)
    return (
        DataLoader(tr_ds, batch_size=args.batch, shuffle=True,  num_workers=4, pin_memory=True),
        DataLoader(vl_ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True))

def main():
    """
    Main script: Parses command-line arguments, builds and trains
    the 2-stream detector model, and evaluates it on the validation and test datasets!!
    """
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_loader, vl_loader = build_loaders(args)
    model = TwoStreamDetector(errors_only = False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay = 1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    trainer = Trainer(model, opt, sched, device, Path(args.save_dir))
    history = trainer.fit(tr_loader, vl_loader, args.epochs)

    # validation results & plots in save-dir ---
    print("\n=== Validation Results ===")
    vl_p, vl_y, _ = trainer._iter(vl_loader, False)
    vl_pred = (vl_p>0.5).astype(int)
    print(f"VALIDATION:   ACC={accuracy_score(vl_y, vl_pred):.4f}  "
          f"AUC={roc_auc_score(vl_y, vl_p):.4f}  "
          f"EER={compute_eer(vl_y, vl_p):.4f}  "
          f"F1={f1_score(vl_y, vl_pred):.4f}")

    epochs = list(range(1, args.epochs+1))

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.plot(epochs, history["val_loss"],   label="val loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.savefig(trainer.dir/"loss_vs_epoch.png")
    #plt.show()
    # uncomment plt.show() to have the plots pop up

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train acc")
    plt.plot(epochs, history["val_acc"],   label="val acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.savefig(trainer.dir/"acc_vs_epoch.png")
    #plt.show()

    cm = confusion_matrix(vl_y, vl_pred)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Validation Confusion Matrix"); plt.colorbar()
    classes=["Real","Fake"]; ticks=np.arange(2)
    plt.xticks(ticks,classes); plt.yticks(ticks,classes)
    for i,j in itertools.product(range(2),range(2)):
        plt.text(j,i,cm[i,j],ha="center",va="center")
    plt.savefig(trainer.dir/"val_confusion_matrix.png")
    #plt.show()

    # test evaluation 
    print("\n=== Test Results ===")
    cache = Path(args.cache_dir) if args.cache_dir else None
    test_ds = DeepFake3DDataset(Path(args.preproc_root), Path(args.recon_root), "test", None, cache)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)
    t_p, t_y, _ = trainer._iter(test_loader, False)
    t_pred = (t_p > 0.5).astype(int)
    print(f"Test  ACC={accuracy_score(t_y, t_pred):.4f}  "
          f"AUC={roc_auc_score(t_y, t_p):.4f}  "
          f"EER={compute_eer(t_y, t_p):.4f}  "
          f"F1={f1_score(t_y, t_pred):.4f}")

    cm_t = confusion_matrix(t_y, t_pred)
    plt.figure()
    plt.imshow(cm_t, interpolation = "nearest")
    plt.title("Test Confusion Matrix"); plt.colorbar()
    plt.xticks(ticks,classes); plt.yticks(ticks,classes)
    for i, j in itertools.product(range(2),range(2)):
        plt.text(j, i, cm_t[i, j], ha="center", va="center")
    plt.savefig(trainer.dir/"test_confusion_matrix.png")
    #plt.show()

if __name__ == "__main__":
    main()