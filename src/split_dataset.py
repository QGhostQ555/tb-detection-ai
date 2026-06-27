import argparse
import copy
import hashlib
import os
import random
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CLASS_NAMES = ["NORMAL", "TB"]


@dataclass
class Sample:
    path: Path
    domain: str
    score: float
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Construye un split balanceado NORMAL/TB con estratificacion por dataset y opcional oversampling."
    )
    parser.add_argument("--source-dataset-1", "--source-original", dest="source_dataset_1",
                        type=Path, default=base_dir / "data1")
    parser.add_argument("--source-dataset-2", "--source-external", dest="source_dataset_2",
                        type=Path, default=base_dir / "data2")
    parser.add_argument("--target", type=Path, default=base_dir / "data_prepared_mixed")
    parser.add_argument("--target-per-class", type=int, default=0,
                        help="Cantidad por clase. 0 usa el mínimo disponible entre NORMAL y TB (a menos que se use --oversample).")
    parser.add_argument("--target-normal", type=int, default=0,
                        help="Cantidad NORMAL. 0 usa auto/equitativo.")
    parser.add_argument("--target-tb", type=int, default=0,
                        help="Cantidad TB. 0 usa auto/equitativo.")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--balance-domain", action="store_true",
                        help="Fuerza igual cantidad desde data1 y data2 dentro de cada clase.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true", help="Limpia target antes de copiar")
    parser.add_argument("--deduplicate", action="store_true",
                        help="Elimina duplicados por hash dentro de cada clase")
    parser.add_argument("--quality-filter", action="store_true",
                        help="Activa filtros OpenCV de calidad. Por defecto se omiten para dividir rapido.")
    parser.add_argument("--oversample", action="store_true",
                        help="Realiza oversampling de la clase minoritaria para igualar el tamaño de la clase mayoritaria (balance perfecto con más imágenes).")
    parser.add_argument("--min-size", type=int, default=128, help="Resolucion minima por lado")
    parser.add_argument("--min-sharpness", type=float, default=6.0,
                        help="Varianza de Laplaciano minima")
    parser.add_argument("--min-contrast", type=float, default=10.0,
                        help="Desviacion estandar de intensidad minima")
    parser.add_argument("--max-white-black-ratio", type=float, default=0.85,
                        help="Maximo porcentaje combinado de pixeles casi negros/blancos")
    return parser.parse_args()


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_image_quality(path: Path, args: argparse.Namespace) -> Tuple[bool, float, int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, -1.0, 0, 0

    h, w = img.shape[:2]
    if min(h, w) < args.min_size:
        return False, -1.0, w, h

    sharpness = float(cv2.Laplacian(img, cv2.CV_64F).var())
    if sharpness < args.min_sharpness:
        return False, -1.0, w, h

    contrast = float(np.std(img))
    if contrast < args.min_contrast:
        return False, -1.0, w, h

    black_ratio = float(np.mean(img <= 5))
    white_ratio = float(np.mean(img >= 250))
    if (black_ratio + white_ratio) > args.max_white_black_ratio:
        return False, -1.0, w, h

    score = (0.65 * sharpness) + (0.35 * contrast)
    return True, score, w, h


def gather_class_samples(
    source: Path,
    class_name: str,
    domain_tag: str,
    args: argparse.Namespace,
    seen_hashes: set | None,
) -> List[Sample]:
    class_dir = source / class_name
    if not class_dir.exists():
        return []

    samples: List[Sample] = []
    for p in class_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in VALID_EXT:
            continue
        digest = ""
        if seen_hashes is not None:
            try:
                digest = md5_file(p)
            except OSError:
                continue
            if digest in seen_hashes:
                continue

        if args.quality_filter:
            ok, score, w, h = validate_image_quality(p, args)
            if not ok:
                continue
        else:
            score = float(p.stat().st_size)
            w, h = 0, 0
        if seen_hashes is not None:
            seen_hashes.add(digest)
        samples.append(Sample(path=p, domain=domain_tag, score=score, width=w, height=h))
    return samples


def unique_by_path(samples: Sequence[Sample]) -> List[Sample]:
    seen = set()
    out: List[Sample] = []
    for s in samples:
        key = str(s.path)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def gather_for_class_with_fallback(
    class_name: str,
    args: argparse.Namespace,
    source_1: Path,
    source_2: Path,
) -> List[Sample]:
    seen_hashes = set() if args.deduplicate else None
    primary: List[Sample] = []
    primary += gather_class_samples(source_1, class_name, "d1", args, seen_hashes)
    primary += gather_class_samples(source_2, class_name, "d2", args, seen_hashes)
    primary = unique_by_path(primary)
    primary.sort(key=lambda x: x.score, reverse=True)

    if not args.quality_filter:
        return primary

    relaxed_args = copy.deepcopy(args)
    relaxed_args.min_size = min(64, args.min_size)
    relaxed_args.min_sharpness = 0.0
    relaxed_args.min_contrast = 0.0
    relaxed_args.max_white_black_ratio = 1.0

    relaxed_seen = set() if args.deduplicate else None
    relaxed: List[Sample] = []
    relaxed += gather_class_samples(source_1, class_name, "d1", relaxed_args, relaxed_seen)
    relaxed += gather_class_samples(source_2, class_name, "d2", relaxed_args, relaxed_seen)
    relaxed = unique_by_path(relaxed)
    relaxed.sort(key=lambda x: x.score, reverse=True)

    merged = unique_by_path(primary + relaxed)
    merged.sort(key=lambda x: x.score, reverse=True)
    return merged


def domain_counts(samples: Sequence[Sample]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for sample in samples:
        counts[sample.domain] = counts.get(sample.domain, 0) + 1
    return counts


def select_equitable_samples(
    samples: List[Sample],
    target_count: int,
    seed: int,
    balance_domain: bool,
) -> List[Sample]:
    if target_count <= 0:
        raise ValueError("target_count debe ser mayor que 0")
    if len(samples) < target_count:
        raise ValueError(
            f"No hay suficientes muestras: disponibles={len(samples)} solicitadas={target_count}"
        )

    by_domain: Dict[str, List[Sample]] = {}
    for sample in samples:
        by_domain.setdefault(sample.domain, []).append(sample)
    for domain_samples in by_domain.values():
        domain_samples.sort(key=lambda x: x.score, reverse=True)

    if not balance_domain or len(by_domain) <= 1:
        selected = sorted(samples, key=lambda x: x.score, reverse=True)[:target_count]
        rnd = random.Random(seed)
        rnd.shuffle(selected)
        return selected

    domains = sorted(by_domain)
    base_quota = target_count // len(domains)
    remainder = target_count % len(domains)
    selected: List[Sample] = []
    leftovers: List[Sample] = []

    for i, domain in enumerate(domains):
        quota = base_quota + (1 if i < remainder else 0)
        take = min(quota, len(by_domain[domain]))
        selected.extend(by_domain[domain][:take])
        leftovers.extend(by_domain[domain][take:])

    if len(selected) < target_count:
        leftovers.sort(key=lambda x: x.score, reverse=True)
        selected.extend(leftovers[:target_count - len(selected)])

    rnd = random.Random(seed)
    rnd.shuffle(selected)
    return selected[:target_count]


def oversample_class(samples: List[Sample], target_count: int, seed: int, balance_domain: bool) -> List[Sample]:
    """
    Realiza oversampling (con repetición) de las muestras disponibles hasta alcanzar target_count.
    Si balance_domain es True, intenta mantener la proporción original de dominios.
    """
    if len(samples) >= target_count:
        return select_equitable_samples(samples, target_count, seed, balance_domain)
    # Necesitamos duplicar
    if balance_domain:
        by_domain = {}
        for s in samples:
            by_domain.setdefault(s.domain, []).append(s)
        total = len(samples)
        result = []
        rnd = random.Random(seed)
        for domain, domain_samples in by_domain.items():
            needed = int(round(target_count * len(domain_samples) / total))
            # Ajustar por redondeo
            if needed < len(domain_samples):
                result.extend(rnd.sample(domain_samples, needed))
            else:
                result.extend(rnd.choices(domain_samples, k=needed))
        # Si por redondeo falta algún sample, añadir aleatoriamente de cualquier dominio
        if len(result) < target_count:
            remaining = target_count - len(result)
            all_samples = samples
            result.extend(rnd.choices(all_samples, k=remaining))
        rnd.shuffle(result)
        return result
    else:
        rnd = random.Random(seed)
        return rnd.choices(samples, k=target_count)


def split_one_class(samples: List[Sample], val_ratio: float, test_ratio: float, seed: int) -> Dict[str, List[Sample]]:
    if val_ratio <= 0 or test_ratio <= 0 or (val_ratio + test_ratio) >= 0.9:
        raise ValueError("Ratios invalidos: usa val_ratio/test_ratio > 0 y val+test < 0.9")

    out = {"train": [], "val_1": [], "test_1": []}
    by_domain: Dict[str, List[Sample]] = {}
    for sample in samples:
        by_domain.setdefault(sample.domain, []).append(sample)

    for offset, (domain, domain_samples) in enumerate(sorted(by_domain.items())):
        rnd = random.Random(seed + offset + sum(ord(c) for c in domain))
        shuffled = list(domain_samples)
        rnd.shuffle(shuffled)

        n = len(shuffled)
        n_val = int(round(n * val_ratio))
        n_test = int(round(n * test_ratio))
        n_train = n - n_val - n_test
        if n_train <= 0:
            raise ValueError("No hay muestras suficientes para train con esos ratios")

        out["val_1"].extend(shuffled[:n_val])
        out["test_1"].extend(shuffled[n_val:n_val + n_test])
        out["train"].extend(shuffled[n_val + n_test:])

    for split, split_samples in out.items():
        rnd = random.Random(seed + len(split))
        rnd.shuffle(split_samples)
    return out


def ensure_dirs(target: Path, class_names: Sequence[str], clean: bool) -> None:
    def on_rm_error(func, path, _exc_info):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    split_names = ["train", "val_1", "test_1"]
    if clean and target.exists():
        for split in split_names + ["val", "test", "val_2", "test_2", "test_internal", "test_external"]:
            split_path = target / split
            if split_path.exists():
                try:
                    shutil.rmtree(split_path, onerror=on_rm_error)
                except PermissionError:
                    print(f"Advertencia: no se pudo limpiar {split_path} por archivo bloqueado.")

    for split in split_names:
        for class_name in class_names:
            (target / split / class_name).mkdir(parents=True, exist_ok=True)


def build_stable_name(src: Path, domain_tag: str, idx: int = 0) -> str:
    """Si idx>0, se añade un sufijo para evitar colisiones en oversampling."""
    digest = hashlib.md5(str(src).encode("utf-8")).hexdigest()[:10]
    base = f"{domain_tag}_{digest}_{src.stem}"
    if idx > 0:
        base = f"{base}_copy{idx}"
    return base + src.suffix


def copy_split(samples: List[Sample], split: str, target: Path, class_name: str, copy_counter: Dict[str, int]) -> None:
    """
    Copia las muestras al destino. Para muestras que son duplicadas (oversampling),
    se usa un contador para generar nombres únicos.
    """
    for s in samples:
        path_key = str(s.path)
        count = copy_counter.get(path_key, 0)
        copy_counter[path_key] = count + 1
        dst_name = build_stable_name(s.path, s.domain, count)
        dst = target / split / class_name / dst_name
        if not dst.exists():
            shutil.copy2(s.path, dst)


def count_split(target: Path, split: str, class_names: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for class_name in class_names:
        out[class_name] = len([p for p in (target / split / class_name).glob("*") if p.is_file()])
    return out


def resolve_targets(args: argparse.Namespace, normal_count: int, tb_count: int) -> Tuple[int, int]:
    if args.target_per_class > 0:
        return args.target_per_class, args.target_per_class
    if args.target_normal > 0 or args.target_tb > 0:
        requested = [v for v in [args.target_normal, args.target_tb] if v > 0]
        target_normal = args.target_normal if args.target_normal > 0 else min(requested)
        target_tb = args.target_tb if args.target_tb > 0 else min(requested)
        return target_normal, target_tb
    balanced = min(normal_count, tb_count)
    return balanced, balanced


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    for source in [args.source_dataset_1, args.source_dataset_2]:
        if not source.exists():
            raise FileNotFoundError(f"No existe el dataset: {source}")
        for cls in CLASS_NAMES:
            if not (source / cls).exists():
                raise FileNotFoundError(f"Falta clase {cls} en {source}")

    normal_all = gather_for_class_with_fallback(
        "NORMAL", args, args.source_dataset_1, args.source_dataset_2
    )
    tb_all = gather_for_class_with_fallback(
        "TB", args, args.source_dataset_1, args.source_dataset_2
    )

    if args.oversample:
        max_count = max(len(normal_all), len(tb_all))
        target_normal = max_count
        target_tb = max_count
        print(f"Oversampling activado: se balancearán ambas clases a {max_count} muestras.")
    else:
        target_normal, target_tb = resolve_targets(args, len(normal_all), len(tb_all))

    if len(normal_all) < target_normal:
        print(f"Oversampling de NORMAL de {len(normal_all)} a {target_normal}")
        selected_normal = oversample_class(normal_all, target_normal, args.seed + 101, args.balance_domain)
    else:
        selected_normal = select_equitable_samples(normal_all, target_normal, args.seed + 101, args.balance_domain)

    if len(tb_all) < target_tb:
        print(f"Oversampling de TB de {len(tb_all)} a {target_tb}")
        selected_tb = oversample_class(tb_all, target_tb, args.seed + 202, args.balance_domain)
    else:
        selected_tb = select_equitable_samples(tb_all, target_tb, args.seed + 202, args.balance_domain)

    n_split = split_one_class(selected_normal, args.val_ratio, args.test_ratio, args.seed)
    t_split = split_one_class(selected_tb, args.val_ratio, args.test_ratio, args.seed)

    ensure_dirs(args.target, CLASS_NAMES, clean=args.clean)

    copy_counter = {}

    for split in ["train", "val_1", "test_1"]:
        copy_split(n_split[split], split, args.target, "NORMAL", copy_counter)
        copy_split(t_split[split], split, args.target, "TB", copy_counter)

    print(f"Split completado en: {args.target}")
    print("\nMuestras disponibles despues de filtros:")
    print(f"- NORMAL disponibles: {len(normal_all)} | dominios={domain_counts(normal_all)}")
    print(f"- TB disponibles: {len(tb_all)} | dominios={domain_counts(tb_all)}")

    print("\nSeleccion usada (despues de posible oversampling):")
    print(f"- NORMAL usadas: {len(selected_normal)} | dominios={domain_counts(selected_normal)}")
    print(f"- TB usadas: {len(selected_tb)} | dominios={domain_counts(selected_tb)}")

    print("\nResumen de splits:")
    for split in ["train", "val_1", "test_1"]:
        c = count_split(args.target, split, CLASS_NAMES)
        print(f"- {split}: NORMAL={c['NORMAL']}, TB={c['TB']}, total={c['NORMAL'] + c['TB']}")

    print("\nNota:")
    if args.deduplicate:
        print("- Se eliminaron duplicados por hash MD5 dentro de cada clase")
    else:
        print("- Duplicados no eliminados (activa --deduplicate si deseas filtrarlos)")
    if args.oversample:
        print("- Se realizó oversampling de la clase minoritaria para balancear las cantidades.")
    else:
        print("- Split balanceado por clase (tamaño limitado por la clase minoritaria).")
    print(f"Total objetivo usado: NORMAL={target_normal}, TB={target_tb}")


if __name__ == "__main__":
    main()