import argparse
import hashlib
import random
import shutil
import stat
from pathlib import Path
from typing import Dict, List, Tuple

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CLASS_NAMES = ["NORMAL", "TB"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Divide imágenes en train/val/test de forma balanceada por clase y opcionalmente por subcarpeta (dominio)."
    )
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directorio raíz que contiene las carpetas NORMAL/ y TB/ (y dentro opcionalmente subcarpetas de dominio)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directorio destino donde se crearán train/val/test con las mismas subcarpetas de clase")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                        help="Proporción para entrenamiento (0-1)")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Proporción para validación (0-1)")
    parser.add_argument("--test-ratio", type=float, default=0.15,
                        help="Proporción para prueba (0-1)")
    parser.add_argument("--balance-domains", action="store_true",
                        help="Si está activo, dentro de cada clase se intenta que cada subcarpeta (dominio) aporte la misma cantidad de muestras a cada split")
    parser.add_argument("--target-per-class", type=int, default=0,
                        help="Número fijo de muestras por clase (0 = usar todas disponibles)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria")
    parser.add_argument("--clean", action="store_true",
                        help="Elimina output-dir antes de copiar")
    return parser.parse_args()


def collect_samples(class_dir: Path, class_name: str) -> List[Tuple[Path, str]]:
    """
    Recorre class_dir y devuelve lista de (ruta_imagen, subcarpeta_relativa).
    La subcarpeta se usa como dominio (si hay subcarpetas internas, se toma la primera relativa).
    Ejemplo: class_dir/NORMAL/ -> dentro puede haber d1/ y d2/; devuelve (path, 'd1').
    Si no hay subcarpetas, dominio = ''.
    """
    samples = []
    for item in class_dir.iterdir():
        if item.is_dir():
            # subcarpeta = dominio
            domain = item.name
            for img in item.rglob("*"):
                if img.suffix.lower() in VALID_EXT:
                    samples.append((img, domain))
        elif item.is_file() and item.suffix.lower() in VALID_EXT:
            # archivos directamente en class_dir
            samples.append((item, "default"))
    return samples


def split_stratified_by_domain(samples_by_domain: Dict[str, List[Path]],
                               train_ratio, val_ratio, test_ratio, seed):
    """
    Dado un diccionario {dominio: lista de rutas}, divide cada dominio independientemente
    en train/val/test, luego concatena.
    Retorna dict con splits.
    """
    random.seed(seed)
    splits = {"train": [], "val": [], "test": []}
    for domain, paths in samples_by_domain.items():
        random.shuffle(paths)
        n = len(paths)
        n_train = int(round(train_ratio * n))
        n_val = int(round(val_ratio * n))
        # Ajuste por redondeo
        if n_train + n_val > n:
            n_train = n - n_val
        elif n_train + n_val < n:
            n_val = n - n_train
        n_test = n - n_train - n_val
        splits["train"].extend(paths[:n_train])
        splits["val"].extend(paths[n_train:n_train+n_val])
        splits["test"].extend(paths[n_train+n_val:])
    # Mezclar globalmente
    for k in splits:
        random.shuffle(splits[k])
    return splits


def select_balanced_by_domain(all_samples: List[Tuple[Path, str]], target_count: int, seed: int):
    """
    Selecciona target_count muestras balanceando la cantidad por dominio.
    Si balance_domains está activo, se asegura que cada dominio contribuya proporcionalmente.
    """
    if target_count <= 0 or target_count >= len(all_samples):
        return all_samples
    # Agrupar por dominio
    by_domain: Dict[str, List[Path]] = {}
    for path, domain in all_samples:
        by_domain.setdefault(domain, []).append(path)
    # Calcular cuántas por dominio (aproximadamente igual)
    num_domains = len(by_domain)
    base = target_count // num_domains
    remainder = target_count % num_domains
    selected = []
    for i, (domain, paths) in enumerate(sorted(by_domain.items())):
        quota = base + (1 if i < remainder else 0)
        # Si no hay suficientes en ese dominio, tomar todas y ajustar después
        take = min(quota, len(paths))
        selected.extend(random.Random(seed + i).sample(paths, take))
    # Si faltan muestras (porque algún dominio tenía menos), completar con cualquier dominio
    if len(selected) < target_count:
        remaining = [p for p in all_samples if p[0] not in selected]
        extra = random.Random(seed).sample(remaining, target_count - len(selected))
        selected.extend([p[0] for p in extra])
    return selected


def main():
    args = parse_args()
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("Las proporciones deben sumar 1")

    input_root = args.input_dir
    output_root = args.output_dir
    if not input_root.exists():
        raise FileNotFoundError(f"No existe {input_root}")

    # Verificar que existan las carpetas de clase
    for cls in CLASS_NAMES:
        class_dir = input_root / cls
        if not class_dir.exists():
            raise FileNotFoundError(f"Falta la carpeta {cls} en {input_root}")

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)

    # Recopilar muestras por clase
    class_samples: Dict[str, List[Tuple[Path, str]]] = {}
    for cls in CLASS_NAMES:
        class_dir = input_root / cls
        samples = collect_samples(class_dir, cls)
        if not samples:
            raise ValueError(f"No se encontraron imágenes en {class_dir}")
        class_samples[cls] = samples

    # Seleccionar número fijo por clase si se pidió
    target_per_class = args.target_per_class
    if target_per_class > 0:
        for cls in CLASS_NAMES:
            if len(class_samples[cls]) < target_per_class:
                print(f"Advertencia: {cls} solo tiene {len(class_samples[cls])} muestras, se usarán todas")
                continue
            # Seleccionar balanceando por dominio si se pide
            class_samples[cls] = select_balanced_by_domain(
                class_samples[cls], target_per_class, args.seed + hash(cls)
            )
            class_samples[cls] = [(p, d) for p, d in class_samples[cls]]

    # División por clase
    final_splits = {"train": [], "val": [], "test": []}
    for cls, samples in class_samples.items():
        # Agrupar por dominio dentro de esta clase
        by_domain: Dict[str, List[Path]] = {}
        for path, domain in samples:
            by_domain.setdefault(domain, []).append(path)
        if args.balance_domains:
            # Dividir estratificando por dominio
            splits = split_stratified_by_domain(by_domain,
                                                args.train_ratio, args.val_ratio, args.test_ratio,
                                                args.seed + hash(cls))
        else:
            # Mezclar todas las rutas y dividir proporcionalmente
            all_paths = [p for p, _ in samples]
            random.Random(args.seed + hash(cls)).shuffle(all_paths)
            n = len(all_paths)
            n_train = int(round(args.train_ratio * n))
            n_val = int(round(args.val_ratio * n))
            if n_train + n_val > n:
                n_train = n - n_val
            elif n_train + n_val < n:
                n_val = n - n_train
            n_test = n - n_train - n_val
            splits = {
                "train": all_paths[:n_train],
                "val": all_paths[n_train:n_train+n_val],
                "test": all_paths[n_train+n_val:]
            }
        # Almacenar junto con la clase
        for split_name, paths in splits.items():
            final_splits[split_name].extend((cls, p) for p in paths)

    # Copiar archivos a la estructura de salida
    for split_name in ["train", "val", "test"]:
        split_dir = output_root / split_name
        for cls in CLASS_NAMES:
            (split_dir / cls).mkdir(parents=True, exist_ok=True)
        for cls, path in final_splits[split_name]:
            dst = split_dir / cls / path.name
            # Evitar colisiones con nombres duplicados
            if dst.exists():
                new_name = f"{path.stem}_{hashlib.md5(str(path).encode()).hexdigest()[:8]}{path.suffix}"
                dst = split_dir / cls / new_name
            shutil.copy2(path, dst)

    # Mostrar estadísticas finales
    print(f"\nDivisión completada en: {output_root}")
    for split in ["train", "val", "test"]:
        split_dir = output_root / split
        stats = {}
        for cls in CLASS_NAMES:
            count = len(list((split_dir / cls).glob("*")))
            stats[cls] = count
        total = sum(stats.values())
        print(f"{split}: NORMAL={stats['NORMAL']}, TB={stats['TB']}, total={total}")

    print("\nOpciones usadas:")
    print(f"  - Balancear dominios: {args.balance_domains}")
    print(f"  - Muestras por clase: {args.target_per_class if args.target_per_class>0 else 'todas'}")
    print(f"  - Proporciones: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    print(f"  - Semilla: {args.seed}")


if __name__ == "__main__":
    main()