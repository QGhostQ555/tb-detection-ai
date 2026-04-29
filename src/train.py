import argparse
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4


# ---------------------------------------------------------------------------
# Reproducibilidad y dispositivo
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Enhancement — técnicas de mejora de imagen (paper Section 2.3)
# ---------------------------------------------------------------------------

class CLAHEEnhancer:
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: int = 8):
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_grid_size, tile_grid_size),
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert("L"), dtype=np.uint8)
        return Image.fromarray(self.clahe.apply(arr), mode="L")


class GammaEnhancer:
    """Gamma correction — mejor técnica según Rahman et al. 2020 (Table 3&5)."""

    def __init__(self, gamma: float = 1.1):
        self.gamma = max(0.1, gamma)
        inv = 1.0 / self.gamma
        self.lut = np.array(
            [(i / 255.0) ** inv * 255 for i in np.arange(256)], dtype=np.uint8
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert("L"), dtype=np.uint8)
        return Image.fromarray(cv2.LUT(arr, self.lut), mode="L")


class UnsharpMaskEnhancer:
    def __init__(self, sigma: float = 1.0, amount: float = 1.0):
        self.sigma = max(0.1, sigma)
        self.amount = max(0.0, amount)

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert("L"), dtype=np.uint8)
        blur = cv2.GaussianBlur(arr, (0, 0), self.sigma)
        out = cv2.addWeighted(arr, 1.0 + self.amount, blur, -self.amount, 0)
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="L")


class HistEqEnhancer:
    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert("L"), dtype=np.uint8)
        return Image.fromarray(cv2.equalizeHist(arr), mode="L")


class ComplementEnhancer:
    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert("L"), dtype=np.uint8)
        return Image.fromarray(255 - arr, mode="L")


def _apply_bcet(arr: np.ndarray, target_mean: float = 110.0) -> np.ndarray:
    x = arr.astype(np.float32)
    l, h, e = float(np.min(x)), float(np.max(x)), float(np.mean(x))
    s = float(np.mean(x ** 2))
    l_out, h_out, e_out = 0.0, 255.0, float(np.clip(target_mean, 1.0, 254.0))
    denom_b = 2.0 * (h * (e_out - l_out) - e * (h_out - l_out) + l * (h_out - e_out))
    if abs(denom_b) < 1e-8 or abs(h - l) < 1e-8:
        return arr.copy()
    b = (h**2 * (e_out - l_out) - s * (h_out - l_out) + l**2 * (h_out - e_out)) / denom_b
    denom_a = (h - l) * (h + l - 2.0 * b)
    if abs(denom_a) < 1e-8:
        return arr.copy()
    a = (h_out - l_out) / denom_a
    c = l_out - a * (l - b) ** 2
    return np.clip(a * (x - b) ** 2 + c, 0, 255).astype(np.uint8)


class BCETEnhancer:
    def __init__(self, target_mean: float = 110.0):
        self.target_mean = target_mean

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert("L"), dtype=np.uint8)
        return Image.fromarray(_apply_bcet(arr, self.target_mean), mode="L")


class EnhanceCompose:
    def __init__(self, steps):
        self.steps = steps

    def __call__(self, img: Image.Image) -> Image.Image:
        for step in self.steps:
            img = step(img)
        return img


def build_enhancer(
    mode: str,
    clahe_clip_limit: float,
    clahe_tile_grid: int,
    gamma: float,
    bcet_target_mean: float = 110.0,
    unsharp_sigma: float = 1.0,
    unsharp_amount: float = 1.0,
) -> Optional[EnhanceCompose]:
    if mode == "none":
        return None
    if mode == "histeq":
        return EnhanceCompose([HistEqEnhancer()])
    if mode == "clahe":
        return EnhanceCompose([CLAHEEnhancer(clahe_clip_limit, clahe_tile_grid)])
    if mode == "gamma":
        return EnhanceCompose([GammaEnhancer(gamma)])
    if mode == "complement":
        return EnhanceCompose([ComplementEnhancer()])
    if mode == "bcet":
        return EnhanceCompose([BCETEnhancer(bcet_target_mean)])
    if mode == "clahe_gamma":
        return EnhanceCompose(
            [
                CLAHEEnhancer(clahe_clip_limit, clahe_tile_grid),
                GammaEnhancer(gamma),
            ]
        )
    if mode == "clahe_unsharp":
        return EnhanceCompose(
            [
                CLAHEEnhancer(clahe_clip_limit, clahe_tile_grid),
                UnsharpMaskEnhancer(unsharp_sigma, unsharp_amount),
            ]
        )
    raise ValueError(f"Modo de enhancement no soportado: {mode}")


# ---------------------------------------------------------------------------
# Segmentación pulmonar heurística
# ---------------------------------------------------------------------------

def _largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(mask)
    out[labels == largest] = 255
    return out


def _estimate_body_mask(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = max(5, int(np.percentile(blur, 2)))
    body = (blur > thr).astype(np.uint8) * 255
    body = _largest_component(body)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, k, iterations=2)
    h, w = gray.shape
    roi = np.zeros_like(body)
    roi[int(0.06 * h):int(0.98 * h), int(0.06 * w):int(0.94 * w)] = 255
    return cv2.bitwise_and(body, roi)


def _make_fallback_lung_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (int(w * 0.34), int(h * 0.53)),
                (int(w * 0.19), int(h * 0.34)), 0, 0, 360, 255, -1)
    cv2.ellipse(mask, (int(w * 0.66), int(h * 0.53)),
                (int(w * 0.19), int(h * 0.34)), 0, 0, 360, 255, -1)
    return mask


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a > 0, b > 0).sum())
    union = float(np.logical_or(a > 0, b > 0).sum())
    return inter / union if union > 0 else 0.0


def _estimate_lung_mask(gray: np.ndarray) -> np.ndarray:
  
    h, w = gray.shape
    body = _estimate_body_mask(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    inv = cv2.GaussianBlur(cv2.bitwise_not(clahe), (5, 5), 0)
    _, dark = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    roi = np.zeros_like(gray, dtype=np.uint8)
    roi[int(0.08 * h):int(0.95 * h), int(0.10 * w):int(0.90 * w)] = 255
    candidate = cv2.bitwise_and(cv2.bitwise_and(dark, body), roi)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, k, iterations=1)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, k, iterations=3)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        candidate, connectivity=8
    )
    components = [
        (int(stats[i, cv2.CC_STAT_AREA]), float(centroids[i][0]), i)
        for i in range(1, num_labels)
        if (int(stats[i, cv2.CC_STAT_AREA]) >= int(0.015 * h * w)
            and 0.16 * h <= centroids[i][1] <= 0.92 * h
            and 0.10 * w <= centroids[i][0] <= 0.90 * w)
    ]
    keep = np.zeros_like(candidate)
    left = [c for c in components if c[1] < 0.5 * w]
    right = [c for c in components if c[1] >= 0.5 * w]
    if left and right:
        for _, _, idx in [max(left, key=lambda x: x[0]),
                          max(right, key=lambda x: x[0])]:
            keep[labels == idx] = 255
    else:
        for _, _, idx in sorted(components, reverse=True)[:2]:
            keep[labels == idx] = 255

    if cv2.countNonZero(keep) < int(0.08 * h * w):
        keep = _make_fallback_lung_mask(h, w)

    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(keep)
    cv2.drawContours(filled, contours, -1, 255, thickness=-1) if contours else None
    filled = filled if contours else keep
    filled = cv2.bitwise_and(filled, body)

    if cv2.countNonZero(filled) < int(0.08 * h * w):
        return _make_fallback_lung_mask(h, w)

    prior = cv2.bitwise_and(_make_fallback_lung_mask(h, w), body)
    if _mask_iou(filled, prior) < 0.18:
        return prior
    sf = filled.astype(np.float32) / 255.0
    pf = prior.astype(np.float32) / 255.0
    return ((0.68 * sf + 0.32 * pf) > 0.34).astype(np.uint8) * 255


class LungSegmentationEnhancer:
    """Oscurece el fondo fuera del pulmón, forzando al modelo a aprender de la ROI."""

    def __init__(self, outside_scale: float = 0.08):
        self.outside_scale = float(np.clip(outside_scale, 0.0, 1.0))

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert("L"), dtype=np.uint8)
        mask = _estimate_lung_mask(arr)
        mf = mask.astype(np.float32) / 255.0
        out = arr.astype(np.float32) * self.outside_scale
        out += arr.astype(np.float32) * mf * (1.0 - self.outside_scale)
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="L")


def build_lung_segmenter(mode: str, outside_scale: float) -> Optional[LungSegmentationEnhancer]:
    if mode == "none":
        return None
    if mode == "heuristic":
        return LungSegmentationEnhancer(outside_scale)
    raise ValueError(f"Modo de segmentación no soportado: {mode}")


# ---------------------------------------------------------------------------
# Score-CAM 
# ---------------------------------------------------------------------------

class ScoreCAM:

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        max_maps: int = 48,
        batch_size: int = 12,
        activation_quantile: float = 0.70,
    ):
        self.model = model
        self.max_maps = max(8, int(max_maps))
        self.batch_size = max(1, int(batch_size))
        self.activation_quantile = float(np.clip(activation_quantile, 0.0, 0.95))
        self.activations: Optional[torch.Tensor] = None
        self._hook = target_layer.register_forward_hook(self._save_activations)

    def _save_activations(self, _m, _i, output):
        self.activations = output

    def close(self):
        self._hook.remove()

    @staticmethod
    def _normalize_maps(maps: torch.Tensor) -> torch.Tensor:
        flat = maps.flatten(1)
        mn = flat.min(dim=1, keepdim=True).values.view(-1, 1, 1)
        mx = flat.max(dim=1, keepdim=True).values.view(-1, 1, 1)
        return (maps - mn) / (mx - mn + 1e-8)

    def compute(
        self,
        input_tensor: torch.Tensor,
        class_idx: int,
        region_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        with torch.no_grad():
            base_logits = self.model(input_tensor)
            if self.activations is None:
                raise RuntimeError("No se capturaron activaciones.")
            base_prob = torch.softmax(base_logits, dim=1)[:, class_idx]

            acts = torch.relu(self.activations[0])
            strength = acts.flatten(1).mean(dim=1)
            valid = torch.where(strength > 1e-8)[0]
            if valid.numel() == 0:
                return np.zeros(input_tensor.shape[-2:], dtype=np.float32)

            st = strength[valid]
            if st.numel() > 8:
                qv = torch.quantile(st, self.activation_quantile)
                f = valid[st >= qv]
                if f.numel() > 0:
                    valid = f

            k = min(self.max_maps, valid.numel())
            top = torch.topk(strength[valid], k=k, largest=True).indices
            selected = acts[valid[top]]

            up = F.interpolate(
                selected.unsqueeze(1), size=input_tensor.shape[-2:],
                mode="bilinear", align_corners=False,
            ).squeeze(1)
            up = self._normalize_maps(up)

            if region_mask is not None:
                msk = torch.from_numpy(
                    region_mask.astype(np.float32) / 255.0
                ).to(input_tensor.device)
                up = up * msk.unsqueeze(0)

            cam = torch.zeros_like(up[0])
            for start in range(0, up.shape[0], self.batch_size):
                mb = up[start:start + self.batch_size]
                probs = torch.softmax(
                    self.model(input_tensor * mb.unsqueeze(1)), dim=1
                )[:, class_idx]
                w = torch.relu(probs - base_prob)
                if w.sum() <= 1e-8:
                    w = probs
                cam += (w.view(-1, 1, 1) * mb).sum(dim=0)

            cam = torch.relu(cam)
            cam = cam / (cam.max() + 1e-8)
            return cam.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers de visualización
# ---------------------------------------------------------------------------

def _looks_like_collage(gray: np.ndarray) -> bool:
    h, w = gray.shape
    bright = (gray >= 245).astype(np.uint8)
    vert = np.mean(bright, axis=0)
    hori = np.mean(bright, axis=1)
    return (
        int(np.sum(vert > 0.95)) > max(10, int(0.04 * w))
        and int(np.sum(hori > 0.95)) > max(10, int(0.04 * h))
    )


def _prepare_gray_for_display(img_path: str, img_size: int, allow_collage: bool) -> np.ndarray:
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {img_path}")
    if not allow_collage and _looks_like_collage(gray):
        raise ValueError("La imagen parece un collage. Usa una radiografía individual.")
    return cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)


def _build_scorecam_mask(body_mask: np.ndarray, lung_mask: np.ndarray) -> np.ndarray:
    h, w = body_mask.shape
    roi = np.zeros_like(body_mask)
    roi[int(0.10 * h):int(0.98 * h), int(0.08 * w):int(0.92 * w)] = 255
    base = cv2.bitwise_and(body_mask, roi).astype(np.float32) / 255.0
    lung_soft = np.clip(
        cv2.GaussianBlur(lung_mask.astype(np.float32) / 255.0, (0, 0), 18.0),
        0.0, 1.0,
    )
    return np.clip((base * (0.25 + 0.75 * lung_soft)) * 255.0, 0, 255).astype(np.uint8)


def _normalize_cam(
    cam: np.ndarray,
    sigma: float,
    low_pct: float,
    high_pct: float,
    threshold: float,
) -> np.ndarray:
    cam = cam.astype(np.float32)
    if sigma > 0:
        cam = cv2.GaussianBlur(cam, (0, 0), sigmaX=sigma, sigmaY=sigma)
    lo = float(np.percentile(cam, np.clip(low_pct, 0, 95)))
    hi = float(np.percentile(cam, np.clip(high_pct, 5, 100)))
    if hi - lo > 1e-8:
        cam = (cam - lo) / (hi - lo)
    cam = np.clip(cam, 0, 1)
    cam = np.maximum(cam - float(np.clip(threshold, 0, 0.95)), 0)
    if cam.max() > 1e-8:
        cam /= cam.max()
    return cam


def _overlay_cam_on_gray(
    gray: np.ndarray,
    cam_map: np.ndarray,
    restrict_mask: Optional[np.ndarray] = None,
    sigma: float = 6.0,
    low_pct: float = 10.0,
    high_pct: float = 99.5,
    threshold: float = 0.08,
    alpha: float = 0.50,
) -> np.ndarray:
    h, w = gray.shape
    cam = cv2.resize(cam_map, (w, h), interpolation=cv2.INTER_CUBIC)
    if restrict_mask is not None:
        msk = cv2.resize(restrict_mask, (w, h), interpolation=cv2.INTER_NEAREST
                         ).astype(np.float32) / 255.0
        cam = cam * msk
    cam = _normalize_cam(cam, sigma, low_pct, high_pct, threshold)
    cam_u8 = np.clip(cam * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base, 1.0 - alpha, heatmap, alpha, 0.0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def _gray_to_model_tensor(
    gray: np.ndarray, mean, std, device: torch.device
) -> torch.Tensor:
    pil = Image.fromarray(gray, mode="L")
    t = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return t(pil).unsqueeze(0).to(device)


def _apply_enhancer_to_gray(gray: np.ndarray, enhancer) -> np.ndarray:
    pil = Image.fromarray(gray, mode="L")
    out = enhancer(pil) if enhancer is not None else pil
    return np.array(out.convert("L"), dtype=np.uint8)


def _find_first_image_in_class(split_dir: str, class_name: str) -> Optional[str]:
    cls_dir = os.path.join(split_dir, class_name)
    if not os.path.isdir(cls_dir):
        return None
    for name in sorted(os.listdir(cls_dir)):
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            return os.path.join(cls_dir, name)
    return None


def _resolve_reference_image(
    explicit_path: str, candidates: List[str]
) -> Optional[str]:
    if explicit_path:
        return explicit_path if os.path.isfile(explicit_path) else None
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


# ---------------------------------------------------------------------------
# generate_cam_grid_figure — función exportada para generate_cam_grid.py
# ---------------------------------------------------------------------------

def generate_cam_grid_figure(
    model: nn.Module,
    device: torch.device,
    tb_index: int,
    img_path: str,
    output_path: str,
    mean,
    std,
    args,
) -> None:
    """
    Genera la figura comparativa (paper Figure 10):
      Fila 1: CXR con cada técnica de enhancement
      Fila 2: Score-CAM sobre CXR plano  → ¿aprende del fondo?
      Fila 3: Score-CAM sobre CXR segmentado → ¿aprende del pulmón?

    Se exporta para que generate_cam_grid.py pueda usarla sin reentrenar.
    """
    import matplotlib.pyplot as plt

    modes = [
        ("none", "Original"),
        ("histeq", "Histeq"),
        ("clahe", "CLAHE"),
        ("gamma", "Gamma ★"),
        ("complement", "Complement"),
        ("bcet", "BCET"),
    ]

    base_gray = _prepare_gray_for_display(
        img_path=img_path,
        img_size=args.img_size,
        allow_collage=getattr(args, "allow_collage_image", False),
    )
    body_mask = _estimate_body_mask(base_gray)
    lung_mask = _estimate_lung_mask(base_gray)
    plain_mask = _build_scorecam_mask(body_mask, lung_mask)

    outside_scale = getattr(args, "segment_outside_scale", 0.08)
    layer_idx = int(np.clip(args.scorecam_layer, 0, len(model.features) - 1))

    score_cam = ScoreCAM(
        model=model,
        target_layer=model.features[layer_idx],
        max_maps=args.scorecam_max_maps,
        batch_size=args.scorecam_batch_size,
        activation_quantile=args.scorecam_activation_quantile,
    )

    fig, axes = plt.subplots(3, len(modes), figsize=(20, 9))
    plt.subplots_adjust(wspace=0.02, hspace=0.06)

    try:
        for col, (mode, title) in enumerate(modes):
            enhancer = build_enhancer(
                mode=mode,
                clahe_clip_limit=args.clahe_clip_limit,
                clahe_tile_grid=args.clahe_tile_grid,
                gamma=args.gamma,
                bcet_target_mean=getattr(args, "bcet_target_mean", 110.0),
                unsharp_sigma=args.unsharp_sigma,
                unsharp_amount=args.unsharp_amount,
            )
            enhanced_gray = _apply_enhancer_to_gray(base_gray, enhancer)

            # CXR segmentado: fondo oscurecido, modelo forzado a mirar pulmones
            mf = lung_mask.astype(np.float32) / 255.0
            seg_gray = np.clip(
                enhanced_gray * outside_scale
                + enhanced_gray * mf * (1.0 - outside_scale),
                0, 255,
            ).astype(np.uint8)

            # Score-CAM sobre CXR plano (¿aprende del fondo?)
            tensor_plain = _gray_to_model_tensor(enhanced_gray, mean, std, device)
            cam_plain = score_cam.compute(
                tensor_plain, class_idx=tb_index, region_mask=plain_mask
            )
            overlay_plain = _overlay_cam_on_gray(
                gray=enhanced_gray, cam_map=cam_plain, restrict_mask=plain_mask,
                sigma=args.cam_smooth_sigma,
                low_pct=args.cam_low_percentile,
                high_pct=args.cam_high_percentile,
                threshold=args.cam_threshold,
                alpha=args.cam_alpha,
            )

            # Score-CAM sobre CXR segmentado (¿aprende del pulmón?)
            tensor_seg = _gray_to_model_tensor(seg_gray, mean, std, device)
            cam_seg = score_cam.compute(
                tensor_seg, class_idx=tb_index, region_mask=lung_mask
            )
            overlay_seg = _overlay_cam_on_gray(
                gray=seg_gray, cam_map=cam_seg, restrict_mask=lung_mask,
                sigma=args.cam_smooth_sigma,
                low_pct=args.cam_low_percentile,
                high_pct=args.cam_high_percentile,
                threshold=args.cam_threshold,
                alpha=args.cam_alpha,
            )

            axes[0, col].imshow(enhanced_gray, cmap="gray")
            axes[0, col].set_title(title, fontsize=11, fontweight="bold")
            axes[0, col].axis("off")
            axes[1, col].imshow(overlay_plain)
            axes[1, col].axis("off")
            axes[2, col].imshow(overlay_seg)
            axes[2, col].axis("off")

        axes[0, 0].set_ylabel("CXR", fontsize=11)
        axes[1, 0].set_ylabel("Score-CAM\nPlain CXR", fontsize=10)
        axes[2, 0].set_ylabel("Score-CAM\nSegmented CXR", fontsize=10)

        fig.suptitle(
            "TB — Enhancement + Score-CAM \n"
            "(★ = mejor técnica según Rahman et al. 2020)",
            fontsize=12, y=0.99,
        )

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
    finally:
        score_cam.close()


# ---------------------------------------------------------------------------
# Métricas y threshold WHO-TPP
# ---------------------------------------------------------------------------

def get_tb_index(class_to_idx: Dict[str, int]) -> int:
    for name, idx in class_to_idx.items():
        if name.upper() == "TB":
            return idx
    raise ValueError(f"Clase TB no encontrada en: {class_to_idx}")


def compute_binary_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = (sens + spec) / 2.0
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "auc": float(auc), "f1": float(f1), "precision": float(prec),
        "sensitivity": float(sens), "specificity": float(spec),
        "balanced_accuracy": float(bal_acc),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_sensitivity: float,
    min_specificity: float,
    policy: str,
) -> Tuple[float, str]:
    thresholds = np.linspace(0.01, 0.99, 99)
    all_rows = [
        {"threshold": float(t), **compute_binary_metrics(y_true, y_prob, float(t))}
        for t in thresholds
    ]
    candidates = [r for r in all_rows
                  if r["sensitivity"] >= min_sensitivity
                  and r["specificity"] >= min_specificity]

    if policy == "who_tpp":
        if candidates:
            best = max(candidates, key=lambda x: (x["balanced_accuracy"], x["f1"]))
            return best["threshold"], (
                f"threshold cumpliendo sensibilidad>={min_sensitivity:.2f} "
                f"y especificidad>={min_specificity:.2f}"
            )
        sens_only = [r for r in all_rows if r["sensitivity"] >= min_sensitivity]
        if sens_only:
            best = max(sens_only, key=lambda x: (x["specificity"], x["f1"]))
            return best["threshold"], "no se cumplio especificidad objetivo; se priorizo sensibilidad"
        best = max(all_rows, key=lambda x: (x["balanced_accuracy"], x["f1"]))
        return best["threshold"], "no se cumplieron objetivos; se uso mayor balanced accuracy"

    if policy == "strict":
        sens_only = [r for r in all_rows if r["sensitivity"] >= min_sensitivity]
        if sens_only:
            best = max(sens_only, key=lambda x: (x["specificity"], x["f1"]))
            return best["threshold"], "modo strict: maximizando especificidad con sensibilidad minima"
        best = max(all_rows, key=lambda x: (x["specificity"], x["balanced_accuracy"]))
        return best["threshold"], "modo strict sin sensibilidad minima alcanzada"

    best = max(all_rows, key=lambda x: (x["balanced_accuracy"], x["f1"]))
    return best["threshold"], "modo balanced: mayor balanced accuracy"


def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    tb_index: int,
):
    model.eval()
    total_loss, y_true, y_prob = 0.0, [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels_d = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels_d).item()
            probs = torch.softmax(logits, dim=1)[:, tb_index].cpu().numpy()
            y_prob.extend(probs.tolist())
            y_true.extend((labels.numpy() == tb_index).astype(np.int64).tolist())
    avg_loss = total_loss / max(1, len(loader))
    return avg_loss, np.asarray(y_true, dtype=np.int64), np.asarray(y_prob, dtype=np.float32)


def print_eval_block(
    split_name: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> Dict[str, float]:
    metrics = compute_binary_metrics(y_true, y_prob, threshold)
    y_pred = (y_prob >= threshold).astype(int)
    print(f"\n=== Evaluacion {split_name} ===")
    print(f"Threshold: {threshold:.3f}")
    print(
        "AUC={auc:.4f} | Sensibilidad={sensitivity:.4f} | "
        "Especificidad={specificity:.4f} | F1={f1:.4f} | "
        "BalancedAcc={balanced_accuracy:.4f}".format(**metrics)
    )
    print("Matriz de confusion [[TN, FP], [FN, TP]]:")
    print(np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]))
    print("Reporte de clasificacion:")
    print(classification_report(y_true, y_pred, target_names=["NORMAL", "TB"],
                                digits=4, zero_division=0))
    return metrics


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

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
    parser.add_argument("--threshold-policy", type=str, default="who_tpp",
                        choices=["who_tpp", "strict", "balanced"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--fast-amd", action="store_true")
    parser.add_argument(
        "--enhancement-mode", type=str, default="clahe_gamma",
        choices=["none", "histeq", "clahe", "gamma", "complement",
                 "bcet", "clahe_gamma", "clahe_unsharp"],
    )
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--clahe-tile-grid", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=1.1)
    parser.add_argument("--bcet-target-mean", type=float, default=110.0)
    parser.add_argument("--unsharp-sigma", type=float, default=1.0)
    parser.add_argument("--unsharp-amount", type=float, default=1.0)
    parser.add_argument("--hflip-prob", type=float, default=0.0)
    parser.add_argument("--rotation-deg", type=float, default=5.0)
    parser.add_argument("--test-internal-subdir", type=str, default="test_1")
    parser.add_argument("--test-external-subdir", type=str, default="test_2")
    parser.add_argument("--val-external-subdir", type=str, default="val_2")
    parser.add_argument("--val2-weight", type=float, default=0.5)
    # Segmentación
    parser.add_argument("--lung-segmentation-mode", type=str, default="none",
                        choices=["none", "heuristic"])
    parser.add_argument("--lung-segmentation-outside-scale", type=float, default=0.08)
    # Score-CAM (activado con --make-cam-grid)
    parser.add_argument("--make-cam-grid", action="store_true",
                        help="Genera grilla Score-CAM al finalizar el entrenamiento")
    parser.add_argument("--scorecam-max-maps", type=int, default=32)
    parser.add_argument("--scorecam-batch-size", type=int, default=8)
    parser.add_argument("--scorecam-layer", type=int, default=8)
    parser.add_argument("--scorecam-activation-quantile", type=float, default=0.70)
    parser.add_argument("--cam-smooth-sigma", type=float, default=7.0)
    parser.add_argument("--cam-low-percentile", type=float, default=12.0)
    parser.add_argument("--cam-high-percentile", type=float, default=99.5)
    parser.add_argument("--cam-threshold", type=float, default=0.08)
    parser.add_argument("--cam-alpha", type=float, default=0.55)
    parser.add_argument("--segment-outside-scale", type=float, default=0.08)
    parser.add_argument("--cam-grid-image", type=str, default="")
    parser.add_argument("--cam-grid-output", type=str, default="")
    parser.add_argument("--allow-collage-image", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    for attr, lo, hi, label in [
        ("val2_weight", 0, 1, "val2-weight"),
        ("hflip_prob", 0, 1, "hflip-prob"),
        ("scorecam_activation_quantile", 0, 0.95, "scorecam-activation-quantile"),
        ("cam_alpha", 0.01, 0.99, "cam-alpha"),
    ]:
        val = getattr(args, attr)
        if not (lo <= val <= hi):
            raise ValueError(f"--{label} debe estar entre {lo} y {hi}")

    # Directorios
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val_1")
    legacy_val_dir = os.path.join(args.data_dir, "val")
    val_external_dir = os.path.join(args.data_dir, args.val_external_subdir)
    legacy_val_external_dir = os.path.join(args.data_dir, "val_1")
    test_internal_dir = os.path.join(args.data_dir, args.test_internal_subdir)
    test_dir = (
        test_internal_dir if os.path.isdir(test_internal_dir)
        else os.path.join(args.data_dir, "test_internal")
        if os.path.isdir(os.path.join(args.data_dir, "test_internal"))
        else os.path.join(args.data_dir, "test")
    )
    test_external_dir = os.path.join(args.data_dir, args.test_external_subdir)

    if not os.path.isdir(val_dir) and os.path.isdir(legacy_val_dir):
        val_dir = legacy_val_dir
        print(f"Usando validacion interna legacy: {legacy_val_dir}")

    for p in [train_dir, val_dir, test_dir]:
        if not os.path.isdir(p):
            raise FileNotFoundError(f"No existe el directorio: {p}")

    os.makedirs(args.model_dir, exist_ok=True)
    device, device_name = select_device()
    print(f"Dispositivo: {device_name} ({device})")

    if args.fast_amd and device_name == "DirectML":
        args.img_size = min(args.img_size, 300)
        args.batch_size = max(args.batch_size, 12)
        print(f"Modo fast-amd: img_size={args.img_size}, batch_size={args.batch_size}")

    enhancer = build_enhancer(
        mode=args.enhancement_mode,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid=args.clahe_tile_grid,
        gamma=args.gamma,
        bcet_target_mean=args.bcet_target_mean,
        unsharp_sigma=args.unsharp_sigma,
        unsharp_amount=args.unsharp_amount,
    )
    lung_segmenter = build_lung_segmenter(
        args.lung_segmentation_mode,
        args.lung_segmentation_outside_scale,
    )
    print(f"Enhancement: {args.enhancement_mode}")
    print(f"Segmentacion pulmonar: {args.lung_segmentation_mode}")

    weights = EfficientNet_B4_Weights.DEFAULT
    mean = weights.transforms().mean
    std = weights.transforms().std

    def _make_steps(is_train: bool):
        steps = [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.Grayscale(num_output_channels=1),
        ]
        if lung_segmenter is not None:
            steps.append(lung_segmenter)
        if enhancer is not None:
            steps.append(enhancer)
        if is_train:
            steps.extend([
                transforms.RandomHorizontalFlip(p=args.hflip_prob),
                transforms.RandomRotation(degrees=args.rotation_deg),
                transforms.ColorJitter(brightness=0.10, contrast=0.10),
            ])
        steps.extend([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        return transforms.Compose(steps)

    train_transform = _make_steps(is_train=True)
    eval_transform = _make_steps(is_train=False)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)
    val_external_dataset = None
    if os.path.isdir(val_external_dir) and os.listdir(val_external_dir):
        val_external_dataset = datasets.ImageFolder(val_external_dir, transform=eval_transform)
    elif (
        os.path.abspath(legacy_val_external_dir) != os.path.abspath(val_dir)
        and os.path.isdir(legacy_val_external_dir)
        and os.listdir(legacy_val_external_dir)
    ):
        val_external_dataset = datasets.ImageFolder(legacy_val_external_dir, transform=eval_transform)
        print(f"Usando validacion externa legacy: {legacy_val_external_dir}")
    external_dataset = None
    for ext_dir in [
        test_external_dir,
        os.path.join(args.data_dir, "test_external"),
        args.external_dir,
    ]:
        if os.path.isdir(ext_dir) and os.listdir(ext_dir):
            external_dataset = datasets.ImageFolder(ext_dir, transform=eval_transform)
            print(f"Evaluacion externa: {ext_dir}")
            break

    class_to_idx = train_dataset.class_to_idx
    if len(class_to_idx) != 2:
        raise ValueError("Se esperaba clasificacion binaria (2 clases).")
    print(f"Clases: {class_to_idx}")

    tb_idx_train = get_tb_index(class_to_idx)
    tb_idx_val = get_tb_index(val_dataset.class_to_idx)
    tb_idx_val_ext = (get_tb_index(val_external_dataset.class_to_idx)
                      if val_external_dataset else None)
    tb_idx_test = get_tb_index(test_dataset.class_to_idx)
    tb_idx_ext = (get_tb_index(external_dataset.class_to_idx)
                  if external_dataset else None)

    class_counts = np.bincount(np.asarray(train_dataset.targets), minlength=2)
    sample_weights = [1.0 / max(1, class_counts[t]) for t in train_dataset.targets]
    train_sampler = WeightedRandomSampler(
        torch.DoubleTensor(sample_weights), len(sample_weights), replacement=True,
    )

    def build_loader_kwargs(nw: int) -> Dict:
        kw = {"num_workers": nw, "pin_memory": torch.cuda.is_available()}
        if nw > 0:
            kw["persistent_workers"] = True
            kw["prefetch_factor"] = max(2, args.prefetch_factor)
        return kw

    def build_loaders(nw: int):
        kw = build_loader_kwargs(nw)
        return (
            DataLoader(train_dataset, args.batch_size, sampler=train_sampler, **kw),
            DataLoader(val_dataset, args.batch_size, shuffle=False, **kw),
            DataLoader(test_dataset, args.batch_size, shuffle=False, **kw),
            DataLoader(val_external_dataset, args.batch_size, shuffle=False, **kw)
            if val_external_dataset else None,
            DataLoader(external_dataset, args.batch_size, shuffle=False, **kw)
            if external_dataset else None,
        )

    (train_loader, val_loader, test_loader,
     val_external_loader, external_loader) = build_loaders(args.num_workers)

    pos = int(np.sum(np.asarray(train_dataset.targets) == tb_idx_train))
    neg = len(train_dataset.targets) - pos
    print(f"Train -> NORMAL: {neg}, TB: {pos}")

    try:
        model = efficientnet_b4(weights=weights)
        print("Pesos preentrenados cargados.")
    except Exception as exc:
        print(f"Sin pesos preentrenados ({exc}).")
        model = efficientnet_b4(weights=None)

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model = model.to(device)

    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr_head, weight_decay=args.weight_decay, foreach=False,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2,
    )

    best_auc, best_state, wait = -1.0, None, 0
    head_epochs = max(1, int(args.epochs * 0.35))

    for epoch in range(1, args.epochs + 1):
        if epoch == head_epochs + 1:
            for p in model.features[-3:].parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr_finetune, weight_decay=args.weight_decay, foreach=False,
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=2,
            )
            print(f"\nFine-tuning activado en epoch {epoch}.\n")

        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, targets = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / max(1, len(train_loader))
        val_loss, val_true, val_prob = run_inference(
            model, val_loader, device, criterion, tb_idx_val
        )
        val_auc = roc_auc_score(val_true, val_prob) if len(np.unique(val_true)) > 1 else 0.0
        monitor_auc = val_auc
        val2_auc = None

        if val_external_loader and tb_idx_val_ext is not None:
            _, v2t, v2p = run_inference(
                model, val_external_loader, device, criterion, tb_idx_val_ext
            )
            val2_auc = roc_auc_score(v2t, v2p) if len(np.unique(v2t)) > 1 else 0.0
            monitor_auc = (1 - args.val2_weight) * val_auc + args.val2_weight * val2_auc

        scheduler.step(monitor_auc)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_auc={val_auc:.4f}"
            + (f" | val2_auc={val2_auc:.4f} | monitor={monitor_auc:.4f}"
               if val2_auc is not None else "")
        )

        if monitor_auc > best_auc:
            best_auc = monitor_auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print("Early stopping activado.")
                break

    if best_state is None:
        raise RuntimeError("No se obtuvo estado del mejor modelo.")

    model.load_state_dict(best_state)
    model.to(device)

    # Selección de umbral WHO-TPP
    _, vt, vp = run_inference(model, val_loader, device, criterion, tb_idx_val)
    cal_true, cal_prob = [vt], [vp]
    if val_external_loader and tb_idx_val_ext is not None:
        _, v2t, v2p = run_inference(
            model, val_external_loader, device, criterion, tb_idx_val_ext
        )
        cal_true.append(v2t)
        cal_prob.append(v2p)

    cal_true_all = np.concatenate(cal_true)
    cal_prob_all = np.concatenate(cal_prob)
    threshold, thr_note = find_best_threshold(
        cal_true_all, cal_prob_all,
        args.min_sensitivity, args.min_specificity, args.threshold_policy,
    )
    print(f"\nUmbral: {thr_note}. Threshold = {threshold:.3f}")

    metrics_summary: Dict = {
        "threshold": threshold,
        "threshold_rule": thr_note,
        "threshold_policy": args.threshold_policy,
        "min_sensitivity": args.min_sensitivity,
        "min_specificity": args.min_specificity,
        "enhancement_mode": args.enhancement_mode,
        "lung_segmentation_mode": args.lung_segmentation_mode,
        "best_monitor_auc": float(best_auc),
        "best_val_auc": float(best_auc),
        "calibration_samples": int(len(cal_true_all)),
    }

    _, tt, tp_arr = run_inference(model, test_loader, device, criterion, tb_idx_test)
    metrics_summary["test_1"] = print_eval_block("TEST 1 (DATA1)", tt, tp_arr, threshold)

    if external_loader and tb_idx_ext is not None:
        _, et, ep = run_inference(model, external_loader, device, criterion, tb_idx_ext)
        metrics_summary["test_2"] = print_eval_block("TEST 2 (DATA2)", et, ep, threshold)
    else:
        print("\nNo se encontro dataset externo.")

    # Score-CAM al finalizar entrenamiento (si se solicita)
    if args.make_cam_grid:
        tb_class = next(n for n, i in class_to_idx.items() if i == tb_idx_train)
        ref_img = _resolve_reference_image(
            args.cam_grid_image,
            [
                _find_first_image_in_class(test_dir, tb_class) or "",
                _find_first_image_in_class(train_dir, tb_class) or "",
            ],
        )
        if ref_img:
            cam_out = args.cam_grid_output or os.path.join(
                args.model_dir, "tb_enhancement_cam_grid.png"
            )
            model.eval()
            try:
                generate_cam_grid_figure(
                    model=model, device=device, tb_index=tb_idx_train,
                    img_path=ref_img, output_path=cam_out,
                    mean=mean, std=std, args=args,
                )
                layer_idx = int(np.clip(args.scorecam_layer, 0, len(model.features) - 1))
                metrics_summary["cam_grid"] = {
                    "image_source": ref_img,
                    "image_output": cam_out,
                    "cam_method": "Score-CAM",
                    "scorecam_layer": layer_idx,
                }
                print(f"Grilla CAM guardada en: {cam_out}")
            except Exception as exc:
                print(f"No se pudo generar la grilla CAM: {exc}")
        else:
            print("No se encontro imagen de referencia para Score-CAM.")

    # Guardar modelo — incluye enhancement_mode para reproducibilidad
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
            "enhancement_mode": args.enhancement_mode,  # reproducibilidad
            "lung_segmentation_mode": args.lung_segmentation_mode,
            "lung_segmentation_outside_scale": args.lung_segmentation_outside_scale,
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
