from __future__ import annotations
import os
os.environ["ALBUMENTATIONS_DISABLE_DOMAIN_ADAPTATION"] = "1"  # skips qudida + sklearn
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1" # ignores another warning for clarity

import argparse
import itertools
import math
import matplotlib.pyplot as plt
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, List

import albumentations as A
import warnings
import optuna
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (f1_score, roc_auc_score, roc_curve,
    confusion_matrix, accuracy_score)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from grad_cam_saliency import compute_gradcam, show_gradcam_on_image
import pickle
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# Data helpers
def _pil_or_cv(path: Path) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    arr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(path)
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr

def rgb_diff(orig: torch.Tensor, rend: torch.Tensor) -> torch.Tensor:
    return torch.abs(orig - rend).mean(0, keepdim=True)

def depth_inconsistency(depth: torch.Tensor, k: int = 5) -> torch.Tensor:
    pad = k // 2
    local = F.avg_pool2d(depth.unsqueeze(0), k, stride=1, padding=pad)[0]
    return torch.abs(depth - local)

def normal_angle_error(normals: torch.Tensor, k: int = 5) -> torch.Tensor:
    pad = k // 2
    neigh = F.avg_pool2d(normals.unsqueeze(0), k, stride=1, padding=pad)[0]
    neigh = F.normalize(neigh, dim=0, eps=1e-6)
    ang = torch.acos(torch.clamp((normals * neigh).sum(0), -1.0, 1.0))
    return ang.unsqueeze(0) / math.pi

def build_aug() -> "albumentations.core.composition.Compose":
    return A.Compose([
        A.HorizontalFlip(0.5),
        A.Rotate(limit=5, border_mode=cv2.BORDER_REFLECT_101, p=0.3),
        A.RandomBrightnessContrast(0.1, 0.1, p=0.3)
    ])

def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)

# Dataset
class DeepFake3DFullDataset(Dataset):
    exts = (".png", ".jpg", ".jpeg")

    def __init__(
        self,
        preproc_root: Path | str,
        recon_root: Path | str,
        transform: Callable | None   = None,
        cache_dir: Path | str | None = None,
        indices: List[int] | None  = None) -> None:
        super().__init__()
        self.preproc_root = Path(preproc_root)
        self.recon_root = Path(recon_root)
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None

        idx_file = (self.cache_dir or Path(".")) / "dataset_index.pkl"

        def _load_cached() -> list | None:
            if not idx_file.exists():
                return None
            try:
                with idx_file.open("rb") as f:
                    data = pickle.load(f)
                return data if isinstance(data, list) and len(data) else None
            except (EOFError, pickle.UnpicklingError):
                return None

        self.samples = _load_cached()
        if self.samples is None:
            self.samples = []
            for cls_idx, cls_name in enumerate(["fake", "real"]):
                p_dir = self.preproc_root / cls_name
                d_dir = self.recon_root / "Depth" / cls_name
                n_dir = self.recon_root / "Normals" / cls_name
                r_dir = self.recon_root / "OriginalRendered" / cls_name

                for img_path in itertools.chain.from_iterable(
                        p_dir.rglob(f"*{e}") for e in self.exts):
                    stem = img_path.stem
                    d = self._find_by_prefix(d_dir, f"depth_{stem}")
                    n = self._find_by_prefix(n_dir, f"normals_{stem}")
                    r = self._find_by_prefix(r_dir, f"orig_rendered_{stem}")
                    if None in (d, n, r):
                        continue
                    self.samples.append(dict(orig=img_path,
                                             rend=r, depth=d, norm=n,
                                             label=cls_idx))

            print(f"Indexed {len(self.samples):,} samples")

            if idx_file.parent:
                idx_file.parent.mkdir(parents=True, exist_ok=True)
            with idx_file.open("wb") as f:
                pickle.dump(self.samples, f)

        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    def __len__(self) -> int:
        return len(self.samples)

    def _find_by_prefix(self, root_dir: Path, prefix: str) -> Path | None:
        for ext in self.exts:
            cand = root_dir / f"{prefix}{ext}"
            if cand.exists():
                return cand
        hits = sorted(root_dir.glob(f"{prefix}.*"))
        if hits:
            return hits[0]
        return None

    def _load_tensor(self, path: Path, rgb: bool) -> torch.Tensor:
        img = _pil_or_cv(path)
        if rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)

    def _resize(self, t: torch.Tensor, hw) -> torch.Tensor:
        if t.shape[-2:] == hw:
            return t
        return F.interpolate(t.unsqueeze(0), size=hw, mode="bilinear", align_corners=False)[0]

    @torch.no_grad()
    def _errors(self, orig, rend, depth, normals):
        rgb_e = rgb_diff(orig, rend)
        dep_e = depth_inconsistency(depth)
        if dep_e.shape[0] > 1:
            dep_e = dep_e.mean(0, keepdim=True)
        norm_e = normal_angle_error(normals)
        return rgb_e, dep_e, norm_e

    def __getitem__(self, idx: int):
        cache_f = self.cache_dir / f"{idx}.npz" if self.cache_dir else None
        if cache_f and cache_f.exists():
            d = np.load(cache_f)
            orig, rend, depth, normals = (
                torch.from_numpy(d[k]) for k in ("orig", "rend", "depth", "normals"))
            rgb_e, dep_e, norm_e = (
                torch.from_numpy(d[k]) for k in ("rgb_e", "dep_e", "norm_e"))
            label = torch.tensor(int(d["label"]))
        else:
            m = self.samples[idx]
            orig = self._load_tensor(m["orig"],  True)
            rend = self._load_tensor(m["rend"],  True)
            depth = self._load_tensor(m["depth"], False)
            normals = self._load_tensor(m["norm"],  True)
            hw = orig.shape[-2:]
            rend = self._resize(rend,   hw)
            depth = self._resize(depth,  hw)
            normals = self._resize(normals, hw)
            rgb_e, dep_e, norm_e = self._errors(orig, rend, depth, normals)
            label = torch.tensor(m["label"])

            if cache_f:
                cache_f.parent.mkdir(parents=True, exist_ok=True)
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
            stack = stack.numpy().transpose(1, 2, 0)
            stack = self.transform(image=stack)["image"].transpose(2, 0, 1)
            stack = torch.from_numpy(stack)
            secs = [3, 3, depth_ch, 3, 1, 1, 1]
            if sum(secs) != stack.shape[0]:
                depth_ch = stack.shape[0] - 12
                secs = [3, 3, depth_ch, 3, 1, 1, 1]
            orig, rend, depth, normals, rgb_e, dep_e, norm_e = torch.split(stack, secs)

        err_stack = torch.cat([rgb_e, dep_e, norm_e], 0)
        return orig, rend, err_stack, label

def build_kfold_loaders(
    preproc_root: str | Path,
    recon_root: str | Path,
    n_splits: int = 5,
    batch_size: int = 32,
    seed: int = 42,
    cache_dir: str | Path | None = None,
    num_workers: int | None = None):
    num_workers = num_workers or min(8, os.cpu_count())

    full_ds = DeepFake3DFullDataset(preproc_root, recon_root,
                                    transform=None, cache_dir=cache_dir)
    if len(full_ds) == 0:
        raise RuntimeError(
            f"No samples were indexed under:\n"
            f"  - {preproc_root}\n  - {recon_root}\n"
            "Check paths / file naming conventions.")

    labels = np.array([s["label"] for s in full_ds.samples])
    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    loaders = []
    for tr_idx, vl_idx in skf.split(np.zeros(len(labels)), labels):
        tr_ds = torch.utils.data.Subset(full_ds, tr_idx.tolist())
        vl_ds = torch.utils.data.Subset(full_ds, vl_idx.tolist())

        tr_ds.dataset.transform = build_aug()
        vl_ds.dataset.transform = None

        loaders.append((
            DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                       num_workers=2, pin_memory=True,
                       persistent_workers=True, prefetch_factor=2),
            DataLoader(vl_ds, batch_size=batch_size, shuffle=False,
                       num_workers=2, pin_memory=True,
                       persistent_workers=True, prefetch_factor=2)))
    return loaders

# Model (classic fixed architecture)
class ConvBlock(nn.Module):
    def __init__(self, inp: int, out: int, drop: float):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, out, 3, padding=1, bias=False),
            nn.BatchNorm2d(out), nn.ReLU(inplace=True),
            nn.Conv2d(out, out, 3, padding=1, bias=False),
            nn.BatchNorm2d(out), nn.ReLU(inplace=True))
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(drop) if drop else nn.Identity()
    def forward(self, x):
        return self.drop(self.pool(self.conv(x)))

class StreamCNN(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        chans = [32, 64, 128, 256, 256]
        drops = [0.0, 0.0, 0.3, 0.0, 0.3]
        layers, c = [], in_ch
        for n, p in zip(chans, drops):
            layers.append(ConvBlock(c, n, p))
            c = n
        self.blocks = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        return self.gap(self.blocks(x)).flatten(1)

class TwoStreamDetector(nn.Module):
    def __init__(self, errors_only: bool=False):
        super().__init__()
        self.errors_only = errors_only
        if not errors_only:
            self.rgb_stream = StreamCNN(3)
        self.err_stream = StreamCNN(3)
        in_feats = 256 if errors_only else 256*2
        self.head = nn.Sequential(
            nn.Linear(in_feats, 512, bias=False),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 128, bias=False), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(128, 2))
    def forward(self, rgb, err):
        feat = (self.err_stream(err) if self.errors_only
                else torch.cat([self.rgb_stream(rgb), self.err_stream(err)], 1))
        return self.head(feat)

# Training
def compute_eer(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    return float(np.nanmin(np.maximum(fpr, fnr)))

class Trainer:
    def __init__(self, model, opt, sched, device, save_dir: Path):
        self.model, self.opt, self.sched, self.dev = model.to(device), opt, sched, device
        self.best_auc = 0.0
        self.dir = save_dir; self.dir.mkdir(exist_ok=True)
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "val_auc": [], "val_eer": [], "val_f1": []}

    def _iter(self, loader: DataLoader, train: bool):
        if train: self.model.train()
        else: self.model.eval()
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

    def fit(self, tr_loader, vl_loader, epochs, patience=3):
        epochs_no_improve = 0
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            tr_p, tr_y, tr_l = self._iter(tr_loader, True)
            tr_acc = ((tr_p > 0.5).astype(int) == tr_y).mean()
            if self.sched:
                self.sched.step()

            vl_p, vl_y, vl_l = self._iter(vl_loader, False)
            vl_acc = ((vl_p > 0.5).astype(int) == vl_y).mean()
            auc = roc_auc_score(vl_y, vl_p)
            eer = compute_eer(vl_y, vl_p)
            f1 = f1_score(vl_y, (vl_p > 0.5).astype(int))
            val_loss = vl_l.mean()

            self.history["train_loss"].append(tr_l.mean())
            self.history["train_acc"].append(tr_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(vl_acc)
            self.history["val_auc"].append(auc)
            self.history["val_eer"].append(eer)
            self.history["val_f1"].append(f1)

            print(
                f"Epoch {epoch:02d}  "
                f"TRAIN: loss={tr_l.mean():.4f} acc={tr_acc:.4f}  "
                f"VAL: loss={val_loss:.4f} acc={vl_acc:.4f} "
                f"AUC={auc:.4f} EER={eer:.4f} F1={f1:.4f}")

            # save best‐AUC as TORCHSCRIPT
            if auc > self.best_auc:
                self.best_auc = auc
                ts_path = self.dir / "best.pt"
                dummy_rgb = torch.randn(1, 3, 224, 224).to(self.dev)
                dummy_err = torch.randn(1, 3, 224, 224).to(self.dev)
                self.model.eval()
                try:
                    ts_model = torch.jit.trace(self.model, (dummy_rgb, dummy_err))
                    torch.jit.save(ts_model, ts_path)
                except Exception as e:
                    print(f"TorchScript trace failed: {e}")
                    torch.save(self.model.state_dict(), ts_path)
                    print(f"Saved raw state_dict fallback at {ts_path}")

            # EARLYSTOPPING HERE
            if val_loss < best_val_loss - 1e-4:
                best_val_loss     = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered (since there was no val-loss improvement for {patience} epochs).")
                    break

        return self.history

# Main Pipeline 
def run_experiment(
    preproc_root,
    recon_root,
    batch,
    lr,
    weight_decay,
    epochs,
    patience,
    folds,
    seed,
    cache_dir=None,
    save_dir="checkpoints"):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_loaders = build_kfold_loaders(
        preproc_root, recon_root,
        n_splits=folds, batch_size=batch,
        seed=seed, cache_dir=cache_dir)
    fold_metrics: list[dict[str, float]] = []
    for fold, (tr_loader, vl_loader) in enumerate(fold_loaders, 1):
        model = TwoStreamDetector(errors_only=False)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        fold_dir = Path(save_dir) / f"fold_{fold}"
        trainer = Trainer(model, opt, sched, device, fold_dir)
        history = trainer.fit(tr_loader, vl_loader, epochs, patience)
        vl_p, vl_y, _ = trainer._iter(vl_loader, False)
        vl_pred = (vl_p > 0.5).astype(int)
        acc = accuracy_score(vl_y, vl_pred)
        auc = roc_auc_score(vl_y, vl_p)
        f1  = f1_score(vl_y, vl_pred)
        fold_metrics.append(dict(val_acc=acc, auc=auc, f1=f1))
    mean_auc = np.mean([m["auc"] for m in fold_metrics])
    return mean_auc


# Tuning with Optuna library to find the best hyperparams to run the model with
def run_hyperparameter_search(args):
    def objective(trial):
        print(f"\n~~~ [Optuna Trial {trial.number}] ~~~")
        lr = trial.suggest_loguniform('lr', 1e-6, 1e-2)
        batch = trial.suggest_categorical('batch', [16, 32, 64, 128])
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
        epochs = trial.suggest_int('epochs', 10, 30)
        patience = 3

        print(f"Params: lr={lr:.2e}, batch={batch}, weight_decay={weight_decay:.2e}, epochs={epochs}, patience={patience}")

        try:
            mean_auc = run_experiment(
                preproc_root=args.preproc_root,
                recon_root=args.recon_root,
                batch=batch,
                lr=lr,
                weight_decay=weight_decay,
                epochs=epochs,
                patience=patience,
                folds=2, # speed up tuning since anything more is unnecessary
                seed=args.seed,
            )
            print(f"[Trial {trial.number}] mean AUC: {mean_auc:.4f}")
        except Exception as e:
            print(f"[Trial {trial.number}] Exception: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
        return mean_auc

    print("~~~ Starting Optuna hyperparameter tuning... ~~~")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15, show_progress_bar=True)

    print("\n~~~ Tuning Complete ~~~")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    save_dir = args.save_dir or "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # we save best trial as JSON so we can use the hyperparams in the future
    best_result = {
        "best_value": trial.value,
        "best_params": trial.params}
    out_json = os.path.join(save_dir, "optuna_best_trial.json")
    with open(out_json, "w") as f:
        json.dump(best_result, f, indent=2)
    print(f"Saved best Optuna (hyperparmeter tuning) trial to {out_json}")

    # saving all the trials in a .csv file 
    df = study.trials_dataframe()
    out_csv = os.path.join(save_dir, "optuna_all_trials.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved all Optuna (hyperparmeter tuning) trials to {out_csv}")

def full_run_with_results(
    preproc_root,
    recon_root,
    batch,
    lr,
    weight_decay,
    epochs,
    patience,
    folds,
    seed,
    cache_dir=None,
    save_dir="checkpoints"):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_loaders = build_kfold_loaders(
        preproc_root, recon_root,
        n_splits=folds, batch_size=batch,
        seed=seed, cache_dir=cache_dir)
    fold_metrics: list[dict[str, float]] = []
    for fold, (tr_loader, vl_loader) in enumerate(fold_loaders, 1):
        print(f"\n~~~~~~~ Fold {fold}/{folds} ~~~~~~~")
        model = TwoStreamDetector(errors_only=False)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        fold_dir = Path(save_dir) / f"fold_{fold}"
        trainer = Trainer(model, opt, sched, device, fold_dir)
        history = trainer.fit(tr_loader, vl_loader, epochs, patience)
        epochs_list = list(range(1, len(history["train_loss"]) + 1))
        # Loss plots
        plt.figure()
        plt.plot(epochs_list, history["train_loss"], label="train loss")
        plt.plot(epochs_list, history["val_loss"],   label="val loss")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
        plt.title(f"Fold {fold} Loss over Epochs")
        plt.savefig(fold_dir / f"loss_vs_epoch.png"); plt.close()
        # Acc plots
        plt.figure()
        plt.plot(epochs_list, history["train_acc"], label="train acc")
        plt.plot(epochs_list, history["val_acc"],   label="val acc")
        plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend()
        plt.title(f"Fold {fold} Accuracy over Epochs")
        plt.savefig(fold_dir / f"acc_vs_epoch.png"); plt.close()
        # Metrics
        print("\n~ Validation results ~")
        vl_p, vl_y, _ = trainer._iter(vl_loader, False)
        vl_pred = (vl_p > 0.5).astype(int)
        acc = accuracy_score(vl_y, vl_pred)
        auc = roc_auc_score(vl_y, vl_p)
        eer = compute_eer(vl_y, vl_p)
        f1  = f1_score(vl_y, vl_pred)
        print(f"ACC={acc:.4f}  AUC={auc:.4f}  EER={eer:.4f}  F1={f1:.4f}")
        train_final_acc = history["train_acc"][-1]
        fold_metrics.append(dict(
            train_acc = train_final_acc,
            val_acc = acc,
            auc = auc,
            eer = eer,
            f1 = f1))
        # Confusion matrix
        cm = confusion_matrix(vl_y, vl_pred)
        plt.figure()
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"Fold {fold} Validation Confusion Matrix"); plt.colorbar()
        classes = ["Real", "Fake"]; ticks = np.arange(2)
        plt.xticks(ticks, classes); plt.yticks(ticks, classes)
        for i, j in itertools.product(range(2), range(2)):
            plt.text(j, i, cm[i, j], ha="center", va="center")
        plt.savefig(fold_dir / f"val_confusion_matrix.png"); plt.close()
        # GradCAMs
        model.eval()
        gc_dir = fold_dir / "gradcam"; gc_dir.mkdir(exist_ok=True)
        saved = 0
        with torch.no_grad():
            for rgb, _, err, _ in vl_loader:
                rgb_gpu, err_gpu = rgb.to(device), err.to(device)
                cam = compute_gradcam(model, (rgb_gpu[:1], err_gpu[:1]), stream="err")
                show_gradcam_on_image(rgb[0].cpu(), cam)
                plt.axis("off")
                if saved == 0:
                    plt.title(f"Fold {fold} GradCAM Example")
                plt.savefig(gc_dir / f"val_gradcam_{saved+1}.png",
                            bbox_inches="tight", pad_inches=0)
                plt.close()
                saved += 1
                if saved == 5:
                    break
    # K-fold summary
    print("\n~~~~~~~ K-fold summary ~~~~~~~")
    for k in ["val_acc", "auc", "eer", "f1"]:
        scores = [m[k] for m in fold_metrics]
        print(f"{k.upper():7s}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    train_accs = np.array([m["train_acc"] for m in fold_metrics])
    val_accs   = np.array([m["val_acc"]  for m in fold_metrics])
    std_val = val_accs.std()
    gap     = train_accs.mean() - val_accs.mean()
    print("\n~~~~~~~ Over-fitting check ~~~~~~~")
    print(f"Train-val acc gap : {gap:.3f}")
    print(f"Val-acc  std-dev  : {std_val:.3f}")
    if gap > 0.10 or std_val > 0.05:
        print("Model is likely overfitting.")
    else:
        print("No strong signs of overfitting.")
    return fold_metrics


def load_best_optuna_params_if_available(args):
    """
    If optuna_best_trial.json exists in save_dir and the user did not override
    relevant params, load best params from the file. Otherwise, use CLI args.
    (Only classic hparams! No arch tuning.)
    """
    best_json = os.path.join(args.save_dir, "optuna_best_trial.json")
    if os.path.exists(best_json):
        with open(best_json) as f:
            best = json.load(f).get("best_params", {})
        dummy = argparse.Namespace(
            epochs=2,
            batch=32,
            lr=1e-3,
            weight_decay=1e-4,
            patience=3
        )
        # We only will override if the user left the args values at its default
        for key in ("epochs", "batch", "lr", "weight_decay", "patience"):
            if key in best and getattr(args, key) == getattr(dummy, key):
                setattr(args, key, best[key])
        print(f"\n[INFO] Loaded best Optuna trial from: {best_json}")
        print(f"[INFO] Overriding CLI args (if not passed explicitly) with: {best}")
    return args

def main():
    parser = argparse.ArgumentParser(
        description="Two-stream model training script")

    parser.add_argument(
        "--preproc-root",
        required=True,
        help="Path to preprocessed data root")
    parser.add_argument(
        "--recon-root",
        required=True,
        help="Path to 3DRecon data root")
    parser.add_argument(
        "--epochs", type=int, default=2,
        help="Number of training epochs (default: 2)")
    parser.add_argument(
        "--batch", type=int, default=32,
        help="Batch size (default: 32)")
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4,
        help="Weight decay (default: 1e-4)")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional cache directory")
    parser.add_argument(
        "--save-dir", default="checkpoints",
        help="Where to save checkpoints (default: checkpoints)")
    parser.add_argument(
        "--folds", type=int, default=5,
        help="Number of folds (default: 5)")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)")
    parser.add_argument(
        "--patience", type=int, default=3,
        help="Early-stopping patience in epochs (default: 3)")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning with Optuna")

    # Parse arguments:
    args = parser.parse_args()

    # If --tune is set then we run the hyperparameter tuning
    if args.tune:
        run_hyperparameter_search(args)
        return

    # Otherwise, try to load best Optuna trial (if hyperparam tuning was run before)
    args = load_best_optuna_params_if_available(args)

    full_run_with_results(
        preproc_root=args.preproc_root,
        recon_root=args.recon_root,
        batch=args.batch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        folds=args.folds,
        seed=args.seed,
        cache_dir=args.cache_dir,
        save_dir=args.save_dir)

if __name__ == "__main__":
    main()