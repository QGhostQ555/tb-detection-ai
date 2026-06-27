import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from ultralytics import YOLO


VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalua YOLO classification con metricas clinicas TB.")
    parser.add_argument("--model", type=Path, required=True, help="Ruta a best.pt de YOLO cls.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Dataset con train/val/test o val_1/test_1.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "val_1", "test", "test_1"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--output-json", type=Path, default=Path("models/yolo_cls_clinical_metrics.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("models/yolo_cls_predictions.csv"))
    return parser.parse_args()


def collect_images(data_dir: Path, split: str) -> List[Tuple[Path, str]]:
    split_dir = data_dir / split
    if not split_dir.exists() and split == "test":
        split_dir = data_dir / "test_1"
    if not split_dir.exists() and split == "val":
        split_dir = data_dir / "val_1"
    if not split_dir.exists():
        raise FileNotFoundError(f"No existe split: {split_dir}")

    samples: List[Tuple[Path, str]] = []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in VALID_EXT:
                samples.append((path, class_dir.name))
    if not samples:
        raise RuntimeError(f"No hay imagenes en: {split_dir}")
    return samples


def find_tb_index(names: Dict[int, str]) -> int:
    for idx, name in names.items():
        if str(name).upper() == "TB":
            return int(idx)
    raise ValueError(f"Clase TB no encontrada en modelo YOLO: {names}")


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    bal_acc = (sens + spec) / 2.0
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    return {
        "threshold": float(threshold),
        "auc": float(auc),
        "accuracy": float(acc),
        "precision": float(precision),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "f1": float(f1),
        "balanced_accuracy": float(bal_acc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main() -> None:
    args = parse_args()
    samples = collect_images(args.data_dir, args.split)
    model = YOLO(str(args.model))
    tb_idx = find_tb_index(model.names)

    rows = []
    y_true, y_prob = [], []
    for path, class_name in samples:
        result = model.predict(str(path), imgsz=args.imgsz, device=args.device, verbose=False)[0]
        probs = result.probs.data.detach().cpu().numpy()
        prob_tb = float(probs[tb_idx])
        target = 1 if class_name.upper() == "TB" else 0
        y_true.append(target)
        y_prob.append(prob_tb)
        rows.append({
            "path": str(path),
            "class": class_name,
            "target_tb": target,
            "prob_tb": prob_tb,
            "pred_tb": int(prob_tb >= args.threshold),
        })

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float32)
    metrics = compute_metrics(y_true_arr, y_prob_arr, args.threshold)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "class", "target_tb", "prob_tb", "pred_tb"])
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== YOLO CLS Clinical Metrics ===")
    print(f"Model: {args.model}")
    print(f"Split: {args.split} | samples={len(samples)} | threshold={args.threshold:.3f}")
    print(
        f"AUC={metrics['auc']:.4f} | Sens={metrics['sensitivity']:.4f} | "
        f"Spec={metrics['specificity']:.4f} | F1={metrics['f1']:.4f} | "
        f"BalAcc={metrics['balanced_accuracy']:.4f} | Acc={metrics['accuracy']:.4f}"
    )
    print("Matriz [[TN, FP], [FN, TP]]:")
    print(np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]))
    y_pred = (y_prob_arr >= args.threshold).astype(int)
    print(classification_report(y_true_arr, y_pred, target_names=["NORMAL", "TB"], digits=4, zero_division=0))
    print(f"JSON: {args.output_json}")
    print(f"CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
