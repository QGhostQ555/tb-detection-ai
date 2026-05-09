import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from train import AttentionLungUNet, LungUNet, select_device, set_seed


def list_segmentation_pairs(segmentation_dir: str) -> List[Tuple[str, str, str]]:
    cxr_dir = os.path.join(segmentation_dir, "CXR_png")
    left_dir = os.path.join(segmentation_dir, "ManualMask", "leftMask")
    right_dir = os.path.join(segmentation_dir, "ManualMask", "rightMask")
    if not os.path.isdir(cxr_dir):
        raise FileNotFoundError(f"No existe: {cxr_dir}")
    if not os.path.isdir(left_dir) or not os.path.isdir(right_dir):
        raise FileNotFoundError("Faltan ManualMask/leftMask o ManualMask/rightMask.")

    pairs: List[Tuple[str, str, str]] = []
    for name in sorted(os.listdir(cxr_dir)):
        if not name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            continue
        img_path = os.path.join(cxr_dir, name)
        left_path = os.path.join(left_dir, name)
        right_path = os.path.join(right_dir, name)
        if os.path.isfile(left_path) and os.path.isfile(right_path):
            pairs.append((img_path, left_path, right_path))
    if not pairs:
        raise RuntimeError("No se encontraron pares CXR + mascaras manuales.")
    return pairs


class LungMaskDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[str, str, str]],
        img_size: int,
        augment: bool,
        clahe_clip_limit: float = 2.0,
        gamma: float = 1.1,
    ):
        self.pairs = pairs
        self.img_size = img_size
        self.augment = augment
        self.clahe = cv2.createCLAHE(
            clipLimit=max(0.1, float(clahe_clip_limit)),
            tileGridSize=(8, 8),
        )
        self.gamma = max(0.1, float(gamma))
        inv = 1.0 / self.gamma
        self.gamma_lut = np.array(
            [(i / 255.0) ** inv * 255.0 for i in range(256)],
            dtype=np.uint8,
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, left_path, right_path = self.pairs[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        if img is None or left is None or right is None:
            raise FileNotFoundError(f"No se pudo leer par de segmentacion: {img_path}")

        mask = cv2.bitwise_or((left > 0).astype(np.uint8) * 255,
                              (right > 0).astype(np.uint8) * 255)
        # Close small holes in manual masks after merge.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # CLAHE + gamma before augmentation to stabilize contrast across studies.
        img = self.clahe.apply(img)
        img = cv2.LUT(img, self.gamma_lut)

        if self.augment:
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
            angle = random.uniform(-8.0, 8.0)
            scale = random.uniform(0.95, 1.05)
            tx = random.uniform(-0.03, 0.03) * self.img_size
            ty = random.uniform(-0.03, 0.03) * self.img_size
            center = (self.img_size / 2.0, self.img_size / 2.0)
            mat = cv2.getRotationMatrix2D(center, angle, scale)
            mat[0, 2] += tx
            mat[1, 2] += ty
            img = cv2.warpAffine(
                img, mat, (self.img_size, self.img_size),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
            )
            mask = cv2.warpAffine(
                mask, mat, (self.img_size, self.img_size),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            if random.random() < 0.30:
                alpha = random.uniform(0.90, 1.12)
                beta = random.uniform(-10.0, 10.0)
                img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        img = img.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)
        return (
            torch.from_numpy(img).unsqueeze(0),
            torch.from_numpy(mask).unsqueeze(0),
        )


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = float(np.clip(bce_weight, 0.0, 1.0))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = torch.sum(probs * targets, dims)
        denom = torch.sum(probs + targets, dims)
        dice = 1.0 - torch.mean((2.0 * inter + 1.0) / (denom + 1.0))
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice


def mask_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> Dict[str, float]:
    preds = (torch.sigmoid(logits) >= threshold).float()
    targets = (targets >= 0.5).float()
    dims = (1, 2, 3)
    inter = torch.sum(preds * targets, dims)
    pred_sum = torch.sum(preds, dims)
    target_sum = torch.sum(targets, dims)
    union = pred_sum + target_sum - inter
    dice = torch.mean((2.0 * inter + 1.0) / (pred_sum + target_sum + 1.0))
    iou = torch.mean((inter + 1.0) / (union + 1.0))
    return {"dice": float(dice.item()), "iou": float(iou.item())}


def evaluate(model, loader, device, criterion, threshold: float) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    dices, ious = [], []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            total_loss += criterion(logits, masks).item()
            m = mask_metrics(logits, masks, threshold)
            dices.append(m["dice"])
            ious.append(m["iou"])
    return {
        "loss": total_loss / max(1, len(loader)),
        "dice": float(np.mean(dices)) if dices else 0.0,
        "iou": float(np.mean(ious)) if ious else 0.0,
    }


def find_best_threshold(model, loader, device, thresholds: np.ndarray) -> float:
    model.eval()
    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            probs = torch.sigmoid(model(images)).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(masks.numpy())
    if not all_probs:
        return 0.50
    probs = np.concatenate(all_probs, axis=0)
    targets = (np.concatenate(all_targets, axis=0) >= 0.5).astype(np.float32)
    best_thr = 0.50
    best_dice = -1.0
    for thr in thresholds:
        preds = (probs >= thr).astype(np.float32)
        inter = np.sum(preds * targets, axis=(1, 2, 3))
        denom = np.sum(preds + targets, axis=(1, 2, 3))
        dice = np.mean((2.0 * inter + 1.0) / (denom + 1.0))
        if float(dice) > best_dice:
            best_dice = float(dice)
            best_thr = float(thr)
    return best_thr


def parse_args() -> argparse.Namespace:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_segmentation_dir = os.path.abspath(os.path.join(base_dir, "..", "segmentacion"))
    default_model_dir = os.path.abspath(os.path.join(base_dir, "..", "models"))
    parser = argparse.ArgumentParser(description="Entrena U-Net pulmonar con mascaras manuales.")
    parser.add_argument("--segmentation-dir", type=str, default=default_segmentation_dir)
    parser.add_argument("--model-dir", type=str, default=default_model_dir)
    parser.add_argument("--output-name", type=str, default="lung_attention_unet_best.pt")
    parser.add_argument("--architecture", type=str, default="attention_unet",
                        choices=["attention_unet", "unet"])
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=1.1)
    parser.add_argument("--auto-threshold", action="store_true",
                        help="Buscar umbral de mascara optimo en validacion por Dice.")
    parser.add_argument("--min-threshold", type=float, default=0.35)
    parser.add_argument("--max-threshold", type=float, default=0.70)
    parser.add_argument("--threshold-steps", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not (0.05 <= args.val_ratio <= 0.5):
        raise ValueError("--val-ratio debe estar entre 0.05 y 0.5")

    pairs = list_segmentation_pairs(args.segmentation_dir)
    generator = torch.Generator().manual_seed(args.seed)
    val_len = max(1, int(round(len(pairs) * args.val_ratio)))
    train_len = len(pairs) - val_len
    train_pairs, val_pairs = random_split(pairs, [train_len, val_len], generator=generator)

    train_ds = LungMaskDataset(
        list(train_pairs), args.img_size, augment=True,
        clahe_clip_limit=args.clahe_clip_limit, gamma=args.gamma,
    )
    val_ds = LungMaskDataset(
        list(val_pairs), args.img_size, augment=False,
        clahe_clip_limit=args.clahe_clip_limit, gamma=args.gamma,
    )

    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    device, device_name = select_device()
    print(f"Dispositivo: {device_name} ({device})")
    print(f"Pares segmentacion: train={len(train_ds)}, val={len(val_ds)}")

    model_cls = AttentionLungUNet if args.architecture == "attention_unet" else LungUNet
    model = model_cls(base_channels=args.base_channels).to(device)
    criterion = DiceBCELoss(bce_weight=0.45)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3,
    )

    if args.auto_threshold:
        if args.threshold_steps < 3:
            raise ValueError("--threshold-steps debe ser >= 3")
        if not (0.05 <= args.min_threshold < args.max_threshold <= 0.95):
            raise ValueError("Rango de threshold invalido.")

    best_dice = -1.0
    best_state = None
    best_threshold = float(args.threshold)
    wait = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))
        eval_threshold = float(args.threshold)
        if args.auto_threshold:
            threshold_grid = np.linspace(args.min_threshold, args.max_threshold, args.threshold_steps)
            eval_threshold = find_best_threshold(model, val_loader, device, threshold_grid)
        val_metrics = evaluate(model, val_loader, device, criterion, eval_threshold)
        scheduler.step(val_metrics["dice"])
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | dice={val_metrics['dice']:.4f} | "
            f"iou={val_metrics['iou']:.4f} | thr={eval_threshold:.3f}"
        )
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            best_threshold = eval_threshold
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print("Early stopping activado.")
                break

    if best_state is None:
        raise RuntimeError("No se obtuvo un checkpoint valido de U-Net.")

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, args.output_name)
    torch.save(
        {
            "model_state_dict": best_state,
            "architecture": args.architecture,
            "img_size": args.img_size,
            "threshold": float(best_threshold),
            "base_channels": args.base_channels,
            "segmentation_dir": args.segmentation_dir,
            "clahe_clip_limit": args.clahe_clip_limit,
            "gamma": args.gamma,
        },
        model_path,
    )
    metrics_path = os.path.join(
        args.model_dir, os.path.splitext(args.output_name)[0] + "_metrics.json"
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_dice": best_dice,
                "architecture": args.architecture,
                "img_size": args.img_size,
                "threshold": float(best_threshold),
                "threshold_auto": bool(args.auto_threshold),
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "segmentation_dir": args.segmentation_dir,
                "clahe_clip_limit": args.clahe_clip_limit,
                "gamma": args.gamma,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nU-Net guardado en: {model_path}")
    print(f"Metricas guardadas en: {metrics_path}")


if __name__ == "__main__":
    main()
