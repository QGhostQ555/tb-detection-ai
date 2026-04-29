import argparse
import hashlib
import os
import random
import shutil
import stat
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from sklearn.model_selection import train_test_split

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Split de datasets TB con mezcla de dominios")
    parser.add_argument("--source-original", type=Path, default=base_dir / "data1")
    parser.add_argument("--source-external", type=Path, default=base_dir / "data2")
    parser.add_argument("--target", type=Path, default=base_dir / "data_prepared_mixed")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--external-holdout-ratio", type=float, default=0.30)
    parser.add_argument("--external-val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true", help="Limpia el target antes de copiar")
    return parser.parse_args()


def gather_samples(source: Path) -> Tuple[List[Path], List[int], Dict[str, int]]:
    if not source.exists():
        raise FileNotFoundError(f"No existe el dataset: {source}")

    class_dirs = [d for d in source.iterdir() if d.is_dir()]
    class_names = sorted(d.name for d in class_dirs)
    if len(class_names) != 2:
        raise ValueError(f"Se esperaban 2 clases en {source}, encontradas: {class_names}")

    class_to_idx = {name: i for i, name in enumerate(class_names)}
    files: List[Path] = []
    labels: List[int] = []

    for class_name in class_names:
        for p in (source / class_name).rglob("*"):
            if p.is_file() and p.suffix.lower() in VALID_EXT:
                files.append(p)
                labels.append(class_to_idx[class_name])

    if not files:
        raise ValueError(f"No se encontraron imagenes validas en {source}")

    return files, labels, class_to_idx


def stratified_split(
    files: Sequence[Path],
    labels: Sequence[int],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Path], List[int], List[Path], List[int], List[Path], List[int]]:
    temp_ratio = val_ratio + test_ratio
    x_train, x_temp, y_train, y_temp = train_test_split(
        list(files),
        list(labels),
        test_size=temp_ratio,
        stratify=list(labels),
        random_state=seed,
    )

    val_fraction_in_temp = val_ratio / temp_ratio
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=(1 - val_fraction_in_temp),
        stratify=y_temp,
        random_state=seed,
    )

    return x_train, y_train, x_val, y_val, x_test, y_test


def split_external_holdout(
    files: Sequence[Path], labels: Sequence[int], holdout_ratio: float, seed: int
) -> Tuple[List[Path], List[int], List[Path], List[int]]:
    x_keep, x_hold, y_keep, y_hold = train_test_split(
        list(files),
        list(labels),
        test_size=holdout_ratio,
        stratify=list(labels),
        random_state=seed,
    )
    return x_keep, y_keep, x_hold, y_hold


def ensure_dirs(target: Path, class_names: Sequence[str], clean: bool) -> None:
    def on_rm_error(func, path, _exc_info):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    split_names = ["train", "val", "val_1", "test_1", "test_2"]
    if clean and target.exists():
        for split in split_names + ["test", "test_internal", "test_external"]:
            split_path = target / split
            if split_path.exists():
                try:
                    shutil.rmtree(split_path, onerror=on_rm_error)
                except PermissionError:
                    print(f"Advertencia: no se pudo limpiar {split_path} por archivo bloqueado.")
                    print("Se continua sin limpiar ese subdirectorio.")

    for split in split_names:
        for class_name in class_names:
            (target / split / class_name).mkdir(parents=True, exist_ok=True)


def build_stable_name(src: Path, domain_tag: str) -> str:
    digest = hashlib.md5(str(src).encode("utf-8")).hexdigest()[:10]
    return f"{domain_tag}_{digest}_{src.name}"


def copy_files(
    files: Sequence[Path],
    labels: Sequence[int],
    split: str,
    target: Path,
    class_names: Sequence[str],
    domain_tag: str,
) -> None:
    for src, y in zip(files, labels):
        dst_name = build_stable_name(src, domain_tag)
        dst = target / split / class_names[y] / dst_name
        if dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)


def count_split(target: Path, split: str, class_names: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for class_name in class_names:
        out[class_name] = len([p for p in (target / split / class_name).glob("*") if p.is_file()])
    return out


def main() -> None:
    args = parse_args()

    if args.val_ratio <= 0 or args.test_ratio <= 0 or (args.val_ratio + args.test_ratio) >= 0.9:
        raise ValueError("Ratios invalidos: usa val_ratio/test_ratio > 0 y val+test < 0.9")
    if args.external_holdout_ratio <= 0 or args.external_holdout_ratio >= 0.95:
        raise ValueError("external_holdout_ratio invalido: usa valor en (0, 0.95)")
    if args.external_val_ratio < 0 or args.external_val_ratio >= 0.9:
        raise ValueError("external_val_ratio invalido: usa valor en [0, 0.9)")

    random.seed(args.seed)

    o_files, o_labels, o_map = gather_samples(args.source_original)
    e_files, e_labels, e_map = gather_samples(args.source_external)

    if sorted(o_map.keys()) != sorted(e_map.keys()):
        raise ValueError(
            f"Clases distintas entre data1 {sorted(o_map.keys())} y data2 {sorted(e_map.keys())}"
        )

    class_names = sorted(o_map.keys())

    o_train_x, o_train_y, o_val_x, o_val_y, o_test_x, o_test_y = stratified_split(
        o_files, o_labels, args.val_ratio, args.test_ratio, args.seed
    )
    e_keep_x, e_keep_y, e_hold_x, e_hold_y = split_external_holdout(
        e_files, e_labels, args.external_holdout_ratio, args.seed
    )
    if args.external_val_ratio > 0:
        e_train_x, e_val1_x, e_train_y, e_val1_y = train_test_split(
            e_keep_x,
            e_keep_y,
            test_size=args.external_val_ratio,
            stratify=e_keep_y,
            random_state=args.seed,
        )
    else:
        e_train_x, e_val1_x, e_train_y, e_val1_y = e_keep_x, [], e_keep_y, []

    ensure_dirs(args.target, class_names, clean=args.clean)
    copy_files(o_train_x, o_train_y, "train", args.target, class_names, "d1")
    copy_files(e_train_x, e_train_y, "train", args.target, class_names, "d2")
    copy_files(o_val_x, o_val_y, "val", args.target, class_names, "d1")
    copy_files(e_val1_x, e_val1_y, "val_1", args.target, class_names, "d2")
    copy_files(o_test_x, o_test_y, "test_1", args.target, class_names, "d1")
    copy_files(e_hold_x, e_hold_y, "test_2", args.target, class_names, "d2")

    print(f"Split completado en: {args.target}")
    print("\nResumen:")
    for split in ["train", "val", "val_1", "test_1", "test_2"]:
        c = count_split(args.target, split, class_names)
        print(f"- {split}: " + ", ".join([f"{k}={v}" for k, v in c.items()]))

    print("\nNota:")
    print("- train mezcla data1 + data2 (excepto val_1 y test_2 de data2)")
    print("- val proviene de data1")
    print("- val_1 proviene de data2 para calibracion de dominio")
    print("- test_1 proviene de data1")
    print("- test_2 proviene de data2")


if __name__ == "__main__":
    main()
