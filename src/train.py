import argparse
import json
import os
import random
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> Tuple[torch.device, str]:
    if torch.cuda.is_available():
        return torch.device("cuda"), "CUDA"

    try:
        import torch_directml  # type: ignore

        return torch_directml.device(), "DirectML"
    except Exception:
        return torch.device("cpu"), "CPU"


class CLAHEEnhancer:
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: int = 8):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))

    def __call__(self, img: Image.Image) -> Image.Image:
        gray = img.convert("L")
        arr = np.array(gray, dtype=np.uint8)
        out = self.clahe.apply(arr)
        return Image.fromarray(out, mode="L")


class GammaEnhancer:
    def __init__(self, gamma: float = 1.1):
        self.gamma = max(0.1, gamma)
        self.inv_gamma = 1.0 / self.gamma
        lut = np.array([(i / 255.0) ** self.inv_gamma * 255 for i in np.arange(256)], dtype=np.uint8)
        self.lut = lut

    def __call__(self, img: Image.Image) -> Image.Image:
        gray = img.convert("L")
        arr = np.array(gray, dtype=np.uint8)
        out = cv2.LUT(arr, self.lut)
        return Image.fromarray(out, mode="L")


class UnsharpMaskEnhancer:
    def __init__(self, sigma: float = 1.0, amount: float = 1.0):
        self.sigma = max(0.1, sigma)
        self.amount = max(0.0, amount)

    def __call__(self, img: Image.Image) -> Image.Image:
        gray = img.convert("L")
        arr = np.array(gray, dtype=np.uint8)
        blur = cv2.GaussianBlur(arr, (0, 0), self.sigma)
        out = cv2.addWeighted(arr, 1.0 + self.amount, blur, -self.amount, 0)
        out = np.clip(out, 0, 255).astype(np.uint8)
        return Image.fromarray(out, mode="L")


class EnhanceCompose:
    def __init__(self, steps):
        self.steps = steps

    def __call__(self, img: Image.Image) -> Image.Image:
        out = img
        for step in self.steps:
            out = step(out)
        return out


def build_enhancer(
    mode: str,
    clahe_clip_limit: float,
    clahe_tile_grid: int,
    gamma: float,
    unsharp_sigma: float,
    unsharp_amount: float,
):
    if mode == "none":
        return None

    steps = [CLAHEEnhancer(clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_grid)]
    if mode == "clahe_gamma":
        steps.append(GammaEnhancer(gamma=gamma))
    elif mode == "clahe_unsharp":
        steps.append(UnsharpMaskEnhancer(sigma=unsharp_sigma, amount=unsharp_amount))
    return EnhanceCompose(steps)


def get_tb_index(class_to_idx: Dict[str, int]) -> int:
    for name, idx in class_to_idx.items():
        if name.upper() == "TB":
            return idx
    raise ValueError(f"No se encontro la clase TB en: {class_to_idx}")


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = (sensitivity + specificity) / 2.0
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")

    return {
        "auc": float(auc),
        "f1": float(f1),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float(bal_acc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_sensitivity: float,
    min_specificity: float,
    policy: str,
) -> Tuple[float, str]:
    thresholds = np.linspace(0.01, 0.99, 99)
    candidates = []
    fallback = []

    for t in thresholds:
        metrics = compute_binary_metrics(y_true, y_prob, float(t))
        row = {"threshold": float(t), **metrics}
        fallback.append(row)
        if metrics["sensitivity"] >= min_sensitivity and metrics["specificity"] >= min_specificity:
            candidates.append(row)

    if policy == "who_tpp":
        if candidates:
            best = max(candidates, key=lambda x: (x["balanced_accuracy"], x["f1"]))
            return (
                best["threshold"],
                f"threshold cumpliendo sensibilidad>={min_sensitivity:.2f} y especificidad>={min_specificity:.2f}",
            )
        sens_only = [r for r in fallback if r["sensitivity"] >= min_sensitivity]
        if sens_only:
            best = max(sens_only, key=lambda x: (x["specificity"], x["f1"]))
            return best["threshold"], "no se cumplio especificidad objetivo; se priorizo sensibilidad"
        best = max(fallback, key=lambda x: (x["balanced_accuracy"], x["f1"]))
        return best["threshold"], "no se cumplieron objetivos; se uso mayor balanced accuracy"

    if policy == "strict":
        sens_only = [r for r in fallback if r["sensitivity"] >= min_sensitivity]
        if sens_only:
            best = max(sens_only, key=lambda x: (x["specificity"], x["f1"]))
            return best["threshold"], "modo strict: maximizando especificidad con sensibilidad minima"
        best = max(fallback, key=lambda x: (x["specificity"], x["balanced_accuracy"]))
        return best["threshold"], "modo strict sin sensibilidad minima alcanzada: maximizando especificidad"

    # balanced
    best = max(fallback, key=lambda x: (x["balanced_accuracy"], x["f1"]))
    return best["threshold"], "modo balanced: mayor balanced accuracy"


def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    tb_index: int,
):
    model.eval()
    total_loss = 0.0
    y_true, y_prob = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels_device = labels.to(device)

            logits = model(images)  # [N, 2]
            loss = criterion(logits, labels_device)
            probs = torch.softmax(logits, dim=1)[:, tb_index].detach().cpu().numpy()

            total_loss += loss.item()
            y_prob.extend(probs.tolist())
            y_true.extend((labels.numpy() == tb_index).astype(np.int64).tolist())

    avg_loss = total_loss / max(1, len(loader))
    return avg_loss, np.asarray(y_true, dtype=np.int64), np.asarray(y_prob, dtype=np.float32)


def print_eval_block(split_name: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    metrics = compute_binary_metrics(y_true, y_prob, threshold)
    y_pred = (y_prob >= threshold).astype(int)

    print(f"\n=== Evaluacion {split_name} ===")
    print(f"Threshold: {threshold:.3f}")
    print(
        "AUC={auc:.4f} | Sensibilidad={sensitivity:.4f} | Especificidad={specificity:.4f} | "
        "F1={f1:.4f} | BalancedAcc={balanced_accuracy:.4f}".format(**metrics)
    )
    print("Matriz de confusion [[TN, FP], [FN, TP]]:")
    print(np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]))
    print("Reporte de clasificacion:")
    print(classification_report(y_true, y_pred, target_names=["NORMAL", "TB"], digits=4, zero_division=0))

    return metrics


def parse_args() -> argparse.Namespace:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.abspath(os.path.join(base_dir, "..", "data_prepared_mixed"))
    default_external_dir = os.path.abspath(os.path.join(base_dir, "..", "data2"))
    default_model_dir = os.path.abspath(os.path.join(base_dir, "..", "models"))

    parser = argparse.ArgumentParser(description="Entrenamiento TB con EfficientNet-B4")
    parser.add_argument("--data-dir", type=str, default=default_data_dir)
    parser.add_argument("--external-dir", type=str, default=default_external_dir)
    parser.add_argument("--model-dir", type=str, default=default_model_dir)
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=380)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-sensitivity", type=float, default=0.90)
    parser.add_argument("--min-specificity", type=float, default=0.70)
    parser.add_argument(
        "--threshold-policy",
        type=str,
        default="who_tpp",
        choices=["who_tpp", "strict", "balanced"],
        help="Politica para seleccionar umbral final",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--fast-amd", action="store_true", help="Activa configuracion rapida recomendada para AMD/DirectML")
    parser.add_argument(
        "--enhancement-mode",
        type=str,
        default="clahe",
        choices=["none", "clahe", "clahe_gamma", "clahe_unsharp"],
        help="Tecnica de mejora de imagen basada en CXR",
    )
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--clahe-tile-grid", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=1.1)
    parser.add_argument("--unsharp-sigma", type=float, default=1.0)
    parser.add_argument("--unsharp-amount", type=float, default=1.0)
    parser.add_argument("--hflip-prob", type=float, default=0.0)
    parser.add_argument("--rotation-deg", type=float, default=5.0)
    parser.add_argument(
        "--test-internal-subdir",
        type=str,
        default="test_1",
        help="Subcarpeta de test interno dentro de data-dir",
    )
    parser.add_argument(
        "--test-external-subdir",
        type=str,
        default="test_2",
        help="Subcarpeta de test externo dentro de data-dir",
    )
    parser.add_argument(
        "--val-external-subdir",
        type=str,
        default="val_2",
        help="Subcarpeta opcional de validacion externa dentro de data-dir",
    )
    parser.add_argument(
        "--val2-weight",
        type=float,
        default=0.5,
        help="Peso de val_2 en monitor de early stopping (0 a 1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.val2_weight < 0 or args.val2_weight > 1:
        raise ValueError("val2-weight debe estar entre 0 y 1")
    if args.hflip_prob < 0 or args.hflip_prob > 1:
        raise ValueError("hflip-prob debe estar entre 0 y 1")

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    val_external_dir = os.path.join(args.data_dir, args.val_external_subdir)
    test_internal_dir = os.path.join(args.data_dir, args.test_internal_subdir)
    legacy_test_internal_dir = os.path.join(args.data_dir, "test_internal")
    legacy_test_dir = os.path.join(args.data_dir, "test")
    if os.path.isdir(test_internal_dir):
        test_dir = test_internal_dir
    elif os.path.isdir(legacy_test_internal_dir):
        test_dir = legacy_test_internal_dir
    else:
        test_dir = legacy_test_dir

    test_external_dir = os.path.join(args.data_dir, args.test_external_subdir)
    legacy_test_external_dir = os.path.join(args.data_dir, "test_external")

    for path in [train_dir, val_dir, test_dir]:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"No existe el directorio requerido: {path}")

    os.makedirs(args.model_dir, exist_ok=True)
    device, device_name = select_device()
    print(f"Dispositivo seleccionado: {device_name} ({device})")

    if args.fast_amd and device_name == "DirectML":
        # Perfil recomendado para RX 5600 XT en Windows con DirectML.
        args.img_size = min(args.img_size, 300)
        args.batch_size = max(args.batch_size, 12)
        print(
            "Modo fast-amd activado: "
            f"img_size={args.img_size}, batch_size={args.batch_size}"
        )
        if args.num_workers == 0:
            print("Tip: prueba --num-workers 4 en tu VS Code si tu entorno lo permite.")

    enhancer = build_enhancer(
        mode=args.enhancement_mode,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid=args.clahe_tile_grid,
        gamma=args.gamma,
        unsharp_sigma=args.unsharp_sigma,
        unsharp_amount=args.unsharp_amount,
    )
    print(f"Modo enhancement: {args.enhancement_mode}")

    weights = EfficientNet_B4_Weights.DEFAULT
    mean = weights.transforms().mean
    std = weights.transforms().std

    train_steps = [
        transforms.Resize((args.img_size, args.img_size)),
        transforms.Grayscale(num_output_channels=1),
    ]
    eval_steps = [
        transforms.Resize((args.img_size, args.img_size)),
        transforms.Grayscale(num_output_channels=1),
    ]
    if enhancer is not None:
        train_steps.append(enhancer)
        eval_steps.append(enhancer)

    train_steps.extend(
        [
            transforms.RandomHorizontalFlip(p=args.hflip_prob),
            transforms.RandomRotation(degrees=args.rotation_deg),
            transforms.ColorJitter(brightness=0.10, contrast=0.10),
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_steps.extend(
        [
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_transform = transforms.Compose(train_steps)
    eval_transform = transforms.Compose(eval_steps)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)
    val_external_dataset = (
        datasets.ImageFolder(val_external_dir, transform=eval_transform)
        if os.path.isdir(val_external_dir) and len(os.listdir(val_external_dir)) > 0
        else None
    )

    external_dataset = None
    if os.path.isdir(test_external_dir) and len(os.listdir(test_external_dir)) > 0:
        external_dataset = datasets.ImageFolder(test_external_dir, transform=eval_transform)
        print(f"Evaluacion externa desde holdout preparado: {test_external_dir}")
    elif os.path.isdir(legacy_test_external_dir) and len(os.listdir(legacy_test_external_dir)) > 0:
        external_dataset = datasets.ImageFolder(legacy_test_external_dir, transform=eval_transform)
        print(f"Evaluacion externa desde holdout legacy: {legacy_test_external_dir}")
    elif os.path.isdir(args.external_dir) and len(os.listdir(args.external_dir)) > 0:
        external_dataset = datasets.ImageFolder(args.external_dir, transform=eval_transform)
        print(f"Evaluacion externa desde carpeta legacy: {args.external_dir}")

    class_to_idx = train_dataset.class_to_idx
    class_names = list(class_to_idx.keys())
    print(f"Clases detectadas en train: {class_to_idx}")
    if len(class_names) != 2:
        raise ValueError("Se esperaba clasificacion binaria (dos carpetas de clase).")

    tb_idx_train = get_tb_index(class_to_idx)
    tb_idx_val = get_tb_index(val_dataset.class_to_idx)
    tb_idx_val_external = get_tb_index(val_external_dataset.class_to_idx) if val_external_dataset is not None else None
    tb_idx_test = get_tb_index(test_dataset.class_to_idx)
    tb_idx_external = get_tb_index(external_dataset.class_to_idx) if external_dataset is not None else None

    # Balanceo via sampler para evitar ops de loss menos compatibles con DirectML.
    class_counts = np.bincount(np.asarray(train_dataset.targets), minlength=len(class_to_idx))
    sample_weights = [1.0 / max(1, class_counts[t]) for t in train_dataset.targets]
    train_sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    def build_loader_kwargs(nw: int) -> Dict:
        kw = {
            "num_workers": nw,
            "pin_memory": torch.cuda.is_available(),
        }
        if nw > 0:
            kw["persistent_workers"] = True
            kw["prefetch_factor"] = max(2, args.prefetch_factor)
        return kw

    def build_loaders(nw: int):
        kw = build_loader_kwargs(nw)
        t_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            **kw,
        )
        v_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            **kw,
        )
        te_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            **kw,
        )
        ve_loader = (
            DataLoader(
                val_external_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                **kw,
            )
            if val_external_dataset is not None
            else None
        )
        ex_loader = (
            DataLoader(
                external_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                **kw,
            )
            if external_dataset is not None
            else None
        )
        return t_loader, v_loader, te_loader, ve_loader, ex_loader

    train_loader, val_loader, test_loader, val_external_loader, external_loader = build_loaders(args.num_workers)

    pos_count = int(np.sum(np.asarray(train_dataset.targets) == tb_idx_train))
    neg_count = len(train_dataset.targets) - pos_count
    print(f"Conteo train -> NORMAL: {neg_count}, TB: {pos_count} (balanceo por WeightedRandomSampler)")

    try:
        model = efficientnet_b4(weights=weights)
        print("Pesos preentrenados de EfficientNet-B4 cargados correctamente.")
    except Exception as exc:
        print(f"Advertencia: no se pudieron cargar pesos preentrenados ({exc}).")
        print("Se inicializa EfficientNet-B4 sin preentrenamiento.")
        model = efficientnet_b4(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(class_to_idx))
    model = model.to(device)

    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr_head,
        weight_decay=args.weight_decay,
        foreach=False,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_auc = -1.0
    best_state = None
    wait = 0
    head_epochs = max(1, int(args.epochs * 0.35))

    for epoch in range(1, args.epochs + 1):
        if epoch == head_epochs + 1:
            for param in model.features[-3:].parameters():
                param.requires_grad = True

            optimizer = optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr_finetune,
                weight_decay=args.weight_decay,
                foreach=False,
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
            print(f"\nFine-tuning activado desde epoch {epoch}: se descongelaron bloques finales.\n")

        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            targets = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)  # [N, 2]
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / max(1, len(train_loader))
        val_loss, val_true, val_prob = run_inference(model, val_loader, device, criterion, tb_idx_val)
        val_auc = roc_auc_score(val_true, val_prob) if len(np.unique(val_true)) > 1 else 0.0
        val_auc_monitor = val_auc
        val2_loss = 0.0
        val2_auc = None
        if val_external_loader is not None and tb_idx_val_external is not None:
            val2_loss, val2_true, val2_prob = run_inference(
                model, val_external_loader, device, criterion, tb_idx_val_external
            )
            val2_auc = roc_auc_score(val2_true, val2_prob) if len(np.unique(val2_true)) > 1 else 0.0
            val_auc_monitor = (1.0 - args.val2_weight) * val_auc + args.val2_weight * val2_auc

        scheduler.step(val_auc_monitor)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_auc={val_auc:.4f}"
        )
        if val2_auc is not None:
            print(f"                val2_loss={val2_loss:.4f} | val2_auc={val2_auc:.4f} | monitor_auc={val_auc_monitor:.4f}")

        if val_auc_monitor > best_auc:
            best_auc = val_auc_monitor
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print("Early stopping activado por estancamiento en val_auc.")
                break

    if best_state is None:
        raise RuntimeError("No se pudo obtener estado del mejor modelo.")

    model.load_state_dict(best_state)
    model.to(device)

    _, val_true, val_prob = run_inference(model, val_loader, device, criterion, tb_idx_val)
    cal_true = [val_true]
    cal_prob = [val_prob]
    if val_external_loader is not None and tb_idx_val_external is not None:
        _, val2_true, val2_prob = run_inference(model, val_external_loader, device, criterion, tb_idx_val_external)
        cal_true.append(val2_true)
        cal_prob.append(val2_prob)

    cal_true_all = np.concatenate(cal_true)
    cal_prob_all = np.concatenate(cal_prob)
    threshold, threshold_note = find_best_threshold(
        cal_true_all,
        cal_prob_all,
        args.min_sensitivity,
        args.min_specificity,
        args.threshold_policy,
    )
    print(f"\nSeleccion de umbral: {threshold_note}. Threshold final = {threshold:.3f}")

    metrics_summary = {
        "threshold": threshold,
        "threshold_rule": threshold_note,
        "threshold_policy": args.threshold_policy,
        "min_sensitivity": args.min_sensitivity,
        "min_specificity": args.min_specificity,
        "enhancement_mode": args.enhancement_mode,
        "best_monitor_auc": float(best_auc),
        "best_val_auc": float(best_auc),
        "calibration_samples": int(len(cal_true_all)),
    }

    _, test_true, test_prob = run_inference(model, test_loader, device, criterion, tb_idx_test)
    metrics_summary["test_1"] = print_eval_block("TEST 1 (DATA1)", test_true, test_prob, threshold)

    if external_loader is not None and tb_idx_external is not None:
        _, ext_true, ext_prob = run_inference(model, external_loader, device, criterion, tb_idx_external)
        metrics_summary["test_2"] = print_eval_block("TEST 2 (DATA2)", ext_true, ext_prob, threshold)
    else:
        print("\nNo se encontro dataset externo, se omite evaluacion externa.")

    model_path = os.path.join(args.model_dir, "efficientnet_b4_tb_best.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx_train": class_to_idx,
            "tb_index_train": tb_idx_train,
            "img_size": args.img_size,
            "threshold": threshold,
            "mean": mean,
            "std": std,
        },
        model_path,
    )

    metrics_path = os.path.join(args.model_dir, "training_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

    print(f"\nModelo guardado en: {model_path}")
    print(f"Metricas guardadas en: {metrics_path}")


if __name__ == "__main__":
    main()
