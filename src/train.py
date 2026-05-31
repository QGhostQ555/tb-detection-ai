import argparse
import json
import logging
import os
import random
import hashlib
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
from torchvision.models import (
    efficientnet_b4, EfficientNet_B4_Weights,
    densenet169, DenseNet169_Weights,
)

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
        import torch_directml
        return torch_directml.device(), "DirectML"
    except Exception:
        return torch.device("cpu"), "CPU"

# ---------------------------------------------------------------------------
# Enhancement techniques (paper Section 2.3)
# ---------------------------------------------------------------------------
class CLAHEEnhancer:
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: int = 8):
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = int(tile_grid_size)
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(self.tile_grid_size, self.tile_grid_size),
        )
    def __getstate__(self):
        state = self.__dict__.copy()
        # cv2.CLAHE no es picklable en multiprocessing (Windows/Python 3.14).
        state["clahe"] = None
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(self.tile_grid_size, self.tile_grid_size),
        )
    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert("L"), dtype=np.uint8)
        return Image.fromarray(self.clahe.apply(arr), mode="L")

class GammaEnhancer:
    def __init__(self, gamma: float = 1.1):
        self.gamma = max(0.1, gamma)
        inv = 1.0 / self.gamma
        self.lut = np.array([(i / 255.0) ** inv * 255 for i in range(256)], dtype=np.uint8)
    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert("L"), dtype=np.uint8)
        return Image.fromarray(cv2.LUT(arr, self.lut), mode="L")

class ComplementEnhancer:
    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert("L"), dtype=np.uint8)
        return Image.fromarray(255 - arr, mode="L")

class HistEqEnhancer:
    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert("L"), dtype=np.uint8)
        return Image.fromarray(cv2.equalizeHist(arr), mode="L")

def apply_bcet(arr: np.ndarray, target_mean: float = 110.0) -> np.ndarray:
    x = arr.astype(np.float32)
    l, h, e = float(np.min(x)), float(np.max(x)), float(np.mean(x))
    s = float(np.mean(x ** 2))
    L, H, E = 0.0, 255.0, float(np.clip(target_mean, 1.0, 254.0))
    denom_b = 2.0 * (h * (E - L) - e * (H - L) + l * (H - E))
    if abs(denom_b) < 1e-8 or abs(h - l) < 1e-8:
        return arr.copy()
    b = (h**2 * (E - L) - s * (H - L) + l**2 * (H - E)) / denom_b
    denom_a = (h - l) * (h + l - 2.0 * b)
    if abs(denom_a) < 1e-8:
        return arr.copy()
    a = (H - L) / denom_a
    c = L - a * (l - b) ** 2
    y = a * (x - b) ** 2 + c
    return np.clip(y, 0, 255).astype(np.uint8)

class BCETEnhancer:
    def __init__(self, target_mean: float = 110.0):
        self.target_mean = target_mean
    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert("L"), dtype=np.uint8)
        return Image.fromarray(apply_bcet(arr, self.target_mean), mode="L")

class UnsharpMaskEnhancer:
    def __init__(self, sigma: float = 1.0, amount: float = 1.0):
        self.sigma = max(0.1, sigma)
        self.amount = max(0.0, amount)
    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert("L"), dtype=np.uint8)
        blur = cv2.GaussianBlur(arr, (0, 0), self.sigma)
        out = cv2.addWeighted(arr, 1.0 + self.amount, blur, -self.amount, 0)
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="L")

class TriChannelEnhancer:
    """RGB: gamma, CLAHE, complement – inspired by paper ref [7]."""
    def __init__(self, gamma: float = 1.1, clahe_clip: float = 2.0, clahe_grid: int = 8):
        self._gamma = GammaEnhancer(gamma)
        self._clahe = CLAHEEnhancer(clahe_clip, clahe_grid)
        self._comp = ComplementEnhancer()
    def __call__(self, img: Image.Image) -> Image.Image:
        r = np.array(self._gamma(img), dtype=np.uint8)
        g = np.array(self._clahe(img), dtype=np.uint8)
        b = np.array(self._comp(img), dtype=np.uint8)
        rgb = np.stack([r, g, b], axis=-1)
        return Image.fromarray(rgb, mode="RGB")

class EnhanceCompose:
    def __init__(self, steps):
        self.steps = steps
    def __call__(self, img: Image.Image) -> Image.Image:
        for step in self.steps:
            img = step(img)
        return img


class ToRGB:
    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode == "RGB":
            return img
        return img.convert("RGB")

def build_enhancer(mode: str, clahe_clip_limit: float, clahe_tile_grid: int,
                   gamma: float, bcet_target_mean: float = 110.0,
                   unsharp_sigma: float = 1.0, unsharp_amount: float = 1.0):
    if mode == "none": return None
    if mode == "histeq": return EnhanceCompose([HistEqEnhancer()])
    if mode == "clahe": return EnhanceCompose([CLAHEEnhancer(clahe_clip_limit, clahe_tile_grid)])
    if mode == "gamma": return EnhanceCompose([GammaEnhancer(gamma)])
    if mode == "complement": return EnhanceCompose([ComplementEnhancer()])
    if mode == "bcet": return EnhanceCompose([BCETEnhancer(bcet_target_mean)])
    if mode == "clahe_gamma": return EnhanceCompose([CLAHEEnhancer(clahe_clip_limit, clahe_tile_grid),
                                                     GammaEnhancer(gamma)])
    if mode == "clahe_unsharp": return EnhanceCompose([CLAHEEnhancer(clahe_clip_limit, clahe_tile_grid),
                                                       UnsharpMaskEnhancer(unsharp_sigma, unsharp_amount)])
    if mode == "tricanal": return TriChannelEnhancer(gamma, clahe_clip_limit, clahe_tile_grid)
    raise ValueError(f"Modo de enhancement no soportado: {mode}")

# ---------------------------------------------------------------------------
# Lung segmentation (heuristic + U-Net / Attention U-Net)
# ---------------------------------------------------------------------------
def _largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1: return mask
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
    inter = np.logical_and(a > 0, b > 0).sum()
    union = np.logical_or(a > 0, b > 0).sum()
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
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    components = [
        (int(stats[i, cv2.CC_STAT_AREA]), float(centroids[i][0]), i)
        for i in range(1, num_labels)
        if (stats[i, cv2.CC_STAT_AREA] >= 0.015 * h * w
            and 0.16 * h <= centroids[i][1] <= 0.92 * h
            and 0.10 * w <= centroids[i][0] <= 0.90 * w)
    ]
    keep = np.zeros_like(candidate)
    left = [c for c in components if c[1] < 0.5 * w]
    right = [c for c in components if c[1] >= 0.5 * w]
    if left and right:
        for _, _, idx in [max(left, key=lambda x: x[0]), max(right, key=lambda x: x[0])]:
            keep[labels == idx] = 255
    else:
        for _, _, idx in sorted(components, reverse=True)[:2]:
            keep[labels == idx] = 255
    if cv2.countNonZero(keep) < int(0.08 * h * w):
        keep = _make_fallback_lung_mask(h, w)
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(keep)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=-1)
    else:
        filled = keep
    filled = cv2.bitwise_and(filled, body)
    if cv2.countNonZero(filled) < int(0.08 * h * w):
        return _make_fallback_lung_mask(h, w)
    prior = cv2.bitwise_and(_make_fallback_lung_mask(h, w), body)
    if _mask_iou(filled, prior) < 0.18:
        return prior
    sf = filled.astype(np.float32) / 255.0
    pf = prior.astype(np.float32) / 255.0
    return ((0.68 * sf + 0.32 * pf) > 0.34).astype(np.uint8) * 255

# ----- U-Net models (only when checkpoint provided) -----
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = DoubleConv(in_c, out_c)
    def forward(self, x):
        return self.conv(F.max_pool2d(x, 2))

class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = DoubleConv(in_c, out_c)
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))

class LungUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        c = base_channels
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c*2)
        self.down2 = Down(c*2, c*4)
        self.down3 = Down(c*4, c*8)
        self.down4 = Down(c*8, c*16)
        self.up1 = Up(c*16 + c*8, c*8)
        self.up2 = Up(c*8 + c*4, c*4)
        self.up3 = Up(c*4 + c*2, c*2)
        self.up4 = Up(c*2 + c, c)
        self.outc = nn.Conv2d(c, out_channels, 1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
        return self.outc(x)

class AttentionGate(nn.Module):
    def __init__(self, gate_c, skip_c, inter_c):
        super().__init__()
        self.gate_proj = nn.Sequential(nn.Conv2d(gate_c, inter_c, 1), nn.BatchNorm2d(inter_c))
        self.skip_proj = nn.Sequential(nn.Conv2d(skip_c, inter_c, 1), nn.BatchNorm2d(inter_c))
        self.psi = nn.Sequential(nn.Conv2d(inter_c, 1, 1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, gate, skip):
        gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        att = self.psi(self.relu(self.gate_proj(gate) + self.skip_proj(skip)))
        return skip * att

class AttentionUp(nn.Module):
    def __init__(self, gate_c, skip_c, out_c):
        super().__init__()
        self.attention = AttentionGate(gate_c, skip_c, max(1, skip_c//2))
        self.conv = DoubleConv(gate_c + skip_c, out_c)
    def forward(self, x, skip):
        skip = self.attention(x, skip)
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))

class AttentionLungUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        c = base_channels
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c*2)
        self.down2 = Down(c*2, c*4)
        self.down3 = Down(c*4, c*8)
        self.down4 = Down(c*8, c*16)
        self.up1 = AttentionUp(c*16, c*8, c*8)
        self.up2 = AttentionUp(c*8, c*4, c*4)
        self.up3 = AttentionUp(c*4, c*2, c*2)
        self.up4 = AttentionUp(c*2, c, c)
        self.outc = nn.Conv2d(c, out_channels, 1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
        return self.outc(x)

def _clean_lung_mask(mask: np.ndarray, fallback_gray: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    mask = (mask > 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        cv2.drawContours(filled, contours, -1, 255, thickness=-1)
    else:
        filled = mask
    min_area = int(0.06 * h * w)
    max_area = int(0.70 * h * w)
    area = cv2.countNonZero(filled)
    if area < min_area or area > max_area:
        return _estimate_lung_mask(fallback_gray)
    return filled

class NeuralLungSegmentationEnhancer:
    def __init__(self, checkpoint_path: str, outside_scale=0.08, fallback="heuristic",
                 use_cache: bool = True, cache_max_items: int = 2500):
        self.checkpoint_path = checkpoint_path
        self.outside_scale = float(np.clip(outside_scale, 0.0, 1.0))
        self.fallback = fallback
        self.model = None
        self.input_size = 320
        self.threshold = 0.5
        self.use_cache = bool(use_cache)
        self.cache_max_items = max(0, int(cache_max_items))
        self._mask_cache = {}
    def _load(self):
        if self.model is not None: return
        if not self.checkpoint_path or not os.path.isfile(self.checkpoint_path):
            if self.fallback == "heuristic": return
            raise FileNotFoundError(f"No existe el checkpoint U-Net: {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        base_channels = int(ckpt.get("base_channels", 32))
        arch = str(ckpt.get("architecture", "unet")).lower()
        self.input_size = int(ckpt.get("img_size", self.input_size))
        self.threshold = float(ckpt.get("threshold", self.threshold))
        model_cls = AttentionLungUNet if arch in {"attention_unet", "attention-u-net"} else LungUNet
        model = model_cls(base_channels=base_channels)
        model.load_state_dict(state, strict=True)
        model.eval()
        self.model = model
    def predict_mask(self, gray: np.ndarray) -> np.ndarray:
        cache_key = None
        if self.use_cache and self.cache_max_items > 0:
            cache_key = hashlib.sha1(gray.tobytes()).hexdigest()
            cached = self._mask_cache.get(cache_key)
            if cached is not None:
                return cached
        self._load()
        if self.model is None:
            mask = _estimate_lung_mask(gray)
        else:
            h, w = gray.shape
            small = cv2.resize(gray, (self.input_size, self.input_size),
                               interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            x = torch.from_numpy(small).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                prob = torch.sigmoid(self.model(x))[0, 0].cpu().numpy()
            mask = (prob >= self.threshold).astype(np.uint8) * 255
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = _clean_lung_mask(mask, gray)
        if cache_key is not None:
            if len(self._mask_cache) >= self.cache_max_items:
                self._mask_cache.pop(next(iter(self._mask_cache)))
            self._mask_cache[cache_key] = mask
        return mask
    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert("L"), dtype=np.uint8)
        mask = self.predict_mask(arr)
        mf = mask.astype(np.float32) / 255.0
        out = arr.astype(np.float32) * self.outside_scale
        out += arr.astype(np.float32) * mf * (1.0 - self.outside_scale)
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="L")

class HeuristicLungSegmentationEnhancer:
    def __init__(self, outside_scale=0.08, use_cache: bool = True, cache_max_items: int = 2500):
        self.outside_scale = outside_scale
        self.use_cache = bool(use_cache)
        self.cache_max_items = max(0, int(cache_max_items))
        self._mask_cache = {}
    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert("L"), dtype=np.uint8)
        if self.use_cache and self.cache_max_items > 0:
            cache_key = hashlib.sha1(arr.tobytes()).hexdigest()
            mask = self._mask_cache.get(cache_key)
            if mask is None:
                mask = _estimate_lung_mask(arr)
                if len(self._mask_cache) >= self.cache_max_items:
                    self._mask_cache.pop(next(iter(self._mask_cache)))
                self._mask_cache[cache_key] = mask
        else:
            mask = _estimate_lung_mask(arr)
        mf = mask.astype(np.float32) / 255.0
        out = arr.astype(np.float32) * self.outside_scale
        out += arr.astype(np.float32) * mf * (1.0 - self.outside_scale)
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="L")

def build_lung_segmenter(mode: str, outside_scale: float, checkpoint_path: str = "",
                         use_cache: bool = True, cache_max_items: int = 2500):
    if mode == "none": return None
    if mode == "heuristic":
        return HeuristicLungSegmentationEnhancer(outside_scale, use_cache, cache_max_items)
    if mode in ("unet", "attention_unet"):
        return NeuralLungSegmentationEnhancer(checkpoint_path, outside_scale, "heuristic", use_cache, cache_max_items)
    raise ValueError(f"Modo de segmentación no soportado: {mode}")

# ---------------------------------------------------------------------------
# Apical weighting (specific to TB)
# ---------------------------------------------------------------------------
def _make_apical_weighted_mask(lung_mask: np.ndarray, apical_weight: float = 1.8) -> np.ndarray:
    h, w = lung_mask.shape
    weight_map = np.ones((h, w), dtype=np.float32)
    top_third = int(h * 0.33)
    weight_map[:top_third, :] = apical_weight
    mid_third = int(h * 0.66)
    gradient = np.linspace(apical_weight, 1.0, mid_third - top_third)
    weight_map[top_third:mid_third, :] = gradient[:, np.newaxis]
    lung_float = lung_mask.astype(np.float32) / 255.0
    weighted = lung_float * weight_map
    if weighted.max() > 1e-8:
        weighted = weighted / weighted.max() * 255.0
    return np.clip(weighted, 0, 255).astype(np.uint8)

# ---------------------------------------------------------------------------
# Score-CAM
# ---------------------------------------------------------------------------
class ScoreCAM:
    def __init__(self, model, target_layer, max_maps=48, batch_size=12, activation_quantile=0.70):
        self.model = model
        self.max_maps = max(8, int(max_maps))
        self.batch_size = max(1, int(batch_size))
        self.activation_quantile = float(np.clip(activation_quantile, 0.0, 0.95))
        self.activations = None
        self._hook = target_layer.register_forward_hook(self._save_activations)
    def _save_activations(self, _m, _i, output):
        self.activations = output
    def close(self):
        self._hook.remove()
    @staticmethod
    def _normalize_maps(maps):
        flat = maps.flatten(1)
        mn = flat.min(dim=1, keepdim=True).values.view(-1, 1, 1)
        mx = flat.max(dim=1, keepdim=True).values.view(-1, 1, 1)
        return (maps - mn) / (mx - mn + 1e-8)
    def compute(self, input_tensor, class_idx, region_mask=None):
        with torch.no_grad():
            base_logits = self.model(input_tensor)
            if self.activations is None: raise RuntimeError("Sin activaciones capturadas.")
            base_prob = torch.softmax(base_logits, dim=1)[:, class_idx]
            acts = torch.relu(self.activations[0])
            strength = acts.flatten(1).mean(dim=1)
            valid = torch.where(strength > 1e-8)[0]
            if valid.numel() == 0: return np.zeros(input_tensor.shape[-2:], dtype=np.float32)
            st = strength[valid]
            if st.numel() > 8:
                qv = torch.quantile(st, self.activation_quantile)
                filtered = valid[st >= qv]
                if filtered.numel() > 0: valid = filtered
            k = min(self.max_maps, valid.numel())
            top = torch.topk(strength[valid], k=k, largest=True).indices
            selected = acts[valid[top]]
            up = F.interpolate(selected.unsqueeze(1), size=input_tensor.shape[-2:],
                               mode="bilinear", align_corners=False).squeeze(1)
            up = self._normalize_maps(up)
            if region_mask is not None:
                msk = torch.from_numpy(region_mask.astype(np.float32) / 255.0).to(input_tensor.device)
                up = up * msk.unsqueeze(0)
            cam = torch.zeros_like(up[0])
            for start in range(0, up.shape[0], self.batch_size):
                mb = up[start:start+self.batch_size]
                probs = torch.softmax(self.model(input_tensor * mb.unsqueeze(1)), dim=1)[:, class_idx]
                w = torch.relu(probs - base_prob)
                if w.sum() <= 1e-8: w = probs
                cam += (w.view(-1, 1, 1) * mb).sum(dim=0)
            cam = torch.relu(cam)
            cam = cam / (cam.max() + 1e-8)
            return cam.cpu().numpy().astype(np.float32)

# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def _looks_like_collage(gray: np.ndarray) -> bool:
    h, w = gray.shape
    bright = (gray >= 245).astype(np.uint8)
    vert = np.mean(bright, axis=0); hori = np.mean(bright, axis=1)
    return (np.sum(vert > 0.95) > max(10, int(0.04 * w))
            and np.sum(hori > 0.95) > max(10, int(0.04 * h)))

def _prepare_gray(img_path: str, img_size: int, allow_collage: bool) -> np.ndarray:
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None: raise FileNotFoundError(f"No se pudo leer: {img_path}")
    if not allow_collage and _looks_like_collage(gray):
        raise ValueError("La imagen parece un collage. Usa --allow-collage-image para forzar.")
    return cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)

def _overlay_cam(gray, cam_map, restrict_mask=None, sigma=6.0, low_pct=10.0, high_pct=99.5,
                 threshold=0.08, alpha=0.50):
    h, w = gray.shape
    cam = cv2.resize(cam_map, (w, h), interpolation=cv2.INTER_CUBIC)
    if restrict_mask is not None:
        msk = cv2.resize(restrict_mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
        cam = cam * msk
    if sigma > 0: cam = cv2.GaussianBlur(cam, (0,0), sigma)
    lo, hi = np.percentile(cam, low_pct), np.percentile(cam, high_pct)
    if hi - lo > 1e-8: cam = (cam - lo) / (hi - lo)
    cam = np.clip(cam, 0, 1)
    cam = np.maximum(cam - threshold, 0)
    if cam.max() > 1e-8: cam /= cam.max()
    cam_u8 = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base, 1.0 - alpha, heatmap, alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

def _gray_to_tensor(gray, mean, std, device):
    pil = Image.fromarray(gray, mode="L")
    t = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return t(pil).unsqueeze(0).to(device)

def _apply_enhancer_to_gray(gray, enhancer):
    pil = Image.fromarray(gray, mode="L")
    out = enhancer(pil) if enhancer else pil
    return np.array(out.convert("L"), dtype=np.uint8)

def _find_first_image_in_class(split_dir, class_name):
    cls_dir = os.path.join(split_dir, class_name)
    if not os.path.isdir(cls_dir): return None
    for name in sorted(os.listdir(cls_dir)):
        if name.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
            return os.path.join(cls_dir, name)
    return None

def generate_cam_grid_figure(model, device, tb_index, img_path, output_path, mean, std, args):
    import matplotlib.pyplot as plt
    modes = [
        ("none", "Original"), ("gamma", "Gamma ★"), ("clahe", "CLAHE"),
        ("clahe_gamma", "CLAHE+Gamma"), ("complement", "Complement"), ("bcet", "BCET"),
    ]
    base_gray = _prepare_gray(img_path, args.img_size, getattr(args, "allow_collage_image", False))
    lung_mask = get_lung_mask(base_gray, args, img_path)
    apical_mask = _make_apical_weighted_mask(lung_mask, args.apical_weight)
    body_mask = _estimate_body_mask(base_gray)
    lung_soft = cv2.GaussianBlur(lung_mask.astype(np.float32)/255, (0,0), 18)
    plain_mask_u8 = np.clip((body_mask.astype(np.float32)/255)*(0.25+0.75*lung_soft)*255, 0, 255).astype(np.uint8)
    layer_idx = int(np.clip(args.scorecam_layer, 0, len(model.features)-1))
    score_cam = ScoreCAM(model, model.features[layer_idx],
                         max_maps=args.scorecam_max_maps, batch_size=args.scorecam_batch_size,
                         activation_quantile=args.scorecam_activation_quantile)
    n_rows = 4
    fig, axes = plt.subplots(n_rows, len(modes), figsize=(20, 12))
    plt.subplots_adjust(wspace=0.02, hspace=0.06)
    try:
        for col, (mode, title) in enumerate(modes):
            enhancer = build_enhancer(
                mode=mode, clahe_clip_limit=args.clahe_clip_limit,
                clahe_tile_grid=args.clahe_tile_grid, gamma=args.gamma,
                bcet_target_mean=getattr(args,"bcet_target_mean",110.0),
                unsharp_sigma=args.unsharp_sigma, unsharp_amount=args.unsharp_amount)
            enhanced_gray = _apply_enhancer_to_gray(base_gray, enhancer)
            mf = lung_mask.astype(np.float32)/255
            seg_gray = np.clip(enhanced_gray*args.segment_outside_scale
                               + enhanced_gray*mf*(1-args.segment_outside_scale), 0,255).astype(np.uint8)
            tensor_plain = _gray_to_tensor(enhanced_gray, mean, std, device)
            cam_plain = score_cam.compute(tensor_plain, tb_index, region_mask=plain_mask_u8)
            overlay_plain = _overlay_cam(enhanced_gray, cam_plain, restrict_mask=plain_mask_u8,
                                         sigma=args.cam_smooth_sigma, alpha=args.cam_alpha)
            tensor_seg = _gray_to_tensor(seg_gray, mean, std, device)
            cam_seg = score_cam.compute(tensor_seg, tb_index, region_mask=lung_mask)
            overlay_seg = _overlay_cam(seg_gray, cam_seg, restrict_mask=lung_mask,
                                       sigma=args.cam_smooth_sigma, alpha=args.cam_alpha)
            cam_apical = score_cam.compute(tensor_seg, tb_index, region_mask=apical_mask)
            overlay_apical = _overlay_cam(seg_gray, cam_apical, restrict_mask=apical_mask,
                                          sigma=args.cam_smooth_sigma, alpha=args.cam_alpha)
            axes[0,col].imshow(enhanced_gray, cmap="gray")
            axes[0,col].set_title(title, fontsize=11, fontweight="bold"); axes[0,col].axis("off")
            axes[1,col].imshow(overlay_plain); axes[1,col].axis("off")
            axes[2,col].imshow(overlay_seg); axes[2,col].axis("off")
            axes[3,col].imshow(overlay_apical); axes[3,col].axis("off")
        axes[0,0].set_ylabel("CXR", fontsize=11)
        axes[1,0].set_ylabel("Score-CAM\nPlain CXR", fontsize=10)
        axes[2,0].set_ylabel("Score-CAM\nSegmented", fontsize=10)
        axes[3,0].set_ylabel("Score-CAM\nApical TB", fontsize=10)
        fig.suptitle("TB Detection: Enhancement + Score-CAM (★=best per paper, row4=apical)",
                     fontsize=12, y=0.98)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
    finally:
        score_cam.close()

# ---------------------------------------------------------------------------
# Metrics and threshold (WHO-TPP compliant)
# ---------------------------------------------------------------------------
def get_tb_index(class_to_idx: Dict[str, int]) -> int:
    for name, idx in class_to_idx.items():
        if name.upper() == "TB": return idx
    raise ValueError(f"Clase TB no encontrada en: {class_to_idx}")

def compute_binary_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp/(tp+fn) if (tp+fn)>0 else 0.0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0.0
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = (sens+spec)/2.0
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true))>1 else float("nan")
    return {"auc": float(auc), "f1": float(f1), "precision": float(prec),
            "sensitivity": float(sens), "specificity": float(spec),
            "balanced_accuracy": float(bal_acc),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

def compute_clinical_score(metrics: Dict[str, float], min_sens: float, min_spec: float) -> float:
    # Penaliza fuerte si no cumple piso OMS, luego prioriza F1/BalAcc/AUC.
    sens_gap = max(0.0, min_sens - metrics["sensitivity"])
    spec_gap = max(0.0, min_spec - metrics["specificity"])
    penalty = 3.0 * sens_gap + 2.0 * spec_gap
    base = 0.35 * metrics["auc"] + 0.35 * metrics["f1"] + 0.30 * metrics["balanced_accuracy"]
    return float(base - penalty)

def find_best_threshold(y_true, y_prob, min_sens, min_spec, policy):
    thresholds = np.linspace(0.01, 0.99, 99)
    all_rows = [{"threshold": float(t), **compute_binary_metrics(y_true, y_prob, t)} for t in thresholds]
    candidates = [r for r in all_rows if r["sensitivity"]>=min_sens and r["specificity"]>=min_spec]
    if policy == "who_tpp":
        if candidates:
            best = max(candidates, key=lambda x: (x["balanced_accuracy"], x["f1"]))
            return best["threshold"], "WHO-TPP: sens & spec cumplidos"
        sens_only = [r for r in all_rows if r["sensitivity"]>=min_sens]
        if sens_only:
            best = max(sens_only, key=lambda x: (x["specificity"], x["f1"]))
            return best["threshold"], "WHO-TPP fallback: solo sensibilidad"
        best = max(all_rows, key=lambda x: (x["balanced_accuracy"], x["f1"]))
        return best["threshold"], "WHO-TPP fallback: balanced accuracy"
    elif policy == "strict":
        sens_only = [r for r in all_rows if r["sensitivity"]>=min_sens]
        if sens_only:
            best = max(sens_only, key=lambda x: (x["specificity"], x["f1"]))
            return best["threshold"], "strict: max specificity con sens mínima"
        best = max(all_rows, key=lambda x: (x["specificity"], x["balanced_accuracy"]))
        return best["threshold"], "strict fallback: max specificity"
    elif policy == "hybrid":
        if candidates:
            best = max(candidates, key=lambda x: (0.5 * x["f1"] + 0.5 * x["balanced_accuracy"], x["auc"]))
            return best["threshold"], "hybrid: OMS + max(F1,BalAcc)"
        best = max(all_rows, key=lambda x: (0.5 * x["f1"] + 0.5 * x["balanced_accuracy"], x["auc"]))
        return best["threshold"], "hybrid fallback: max(F1,BalAcc)"
    else:
        best = max(all_rows, key=lambda x: (x["balanced_accuracy"], x["f1"]))
        return best["threshold"], "balanced: mayor balanced accuracy"

def run_inference(model, loader, device, criterion, tb_index):
    model.eval()
    total_loss, y_true, y_prob = 0.0, [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels_d = labels.to(device, non_blocking=True)
            logits = model(images)
            total_loss += criterion(logits, labels_d).item()
            probs = torch.softmax(logits, dim=1)[:, tb_index].cpu().numpy()
            y_prob.extend(probs.tolist())
            y_true.extend((labels.numpy() == tb_index).astype(np.int64).tolist())
    avg_loss = total_loss / max(1, len(loader))
    return avg_loss, np.array(y_true, dtype=np.int64), np.array(y_prob, dtype=np.float32)

def print_eval_block(split_name, y_true, y_prob, threshold):
    metrics = compute_binary_metrics(y_true, y_prob, threshold)
    y_pred = (y_prob >= threshold).astype(int)
    print(f"\n=== Evaluación {split_name} ===")
    print(f"Threshold: {threshold:.3f}")
    print(f"AUC={metrics['auc']:.4f} | Sens={metrics['sensitivity']:.4f} | "
          f"Spec={metrics['specificity']:.4f} | F1={metrics['f1']:.4f} | "
          f"BalAcc={metrics['balanced_accuracy']:.4f}")
    print("Matriz de confusión [[TN, FP], [FN, TP]]:")
    print(np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]))
    print(classification_report(y_true, y_pred, target_names=["NORMAL", "TB"], digits=4, zero_division=0))
    return metrics

# ---------------------------------------------------------------------------
# Collage‑filtering Dataset wrapper
# ---------------------------------------------------------------------------
class FilteredImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, allow_collage=False):
        self.allow_collage = allow_collage
        super().__init__(root, transform=transform)
        if not self.allow_collage:
            self._filter_collages()
    def _filter_collages(self):
        valid_indices = []
        for idx, (path, _) in enumerate(self.samples):
            pil_img = Image.open(path).convert("L")
            gray = np.array(pil_img, dtype=np.uint8)
            if _looks_like_collage(gray):
                logging.warning(f"Collage image skipped: {path}")
            else:
                valid_indices.append(idx)
        if len(valid_indices) < len(self.samples):
            self.samples = [self.samples[i] for i in valid_indices]
            self.targets = [self.targets[i] for i in valid_indices]

# ---------------------------------------------------------------------------
# Transform pipeline builder
# ---------------------------------------------------------------------------
def build_transforms(args, enhancer, lung_segmenter, mean, std, is_train: bool):
    is_tricanal = (args.enhancement_mode == "tricanal")
    steps = []
    if not is_tricanal:
        steps.append(transforms.Grayscale(num_output_channels=1))
    if enhancer is not None:
        steps.append(enhancer)
    # Keep preprocessing order consistent with inference:
    # enhancement (e.g., CLAHE+Gamma) -> lung segmentation -> classification.
    if lung_segmenter is not None:
        steps.append(lung_segmenter)
    steps.append(transforms.Resize((args.img_size, args.img_size)))
    if is_train:
        steps.append(transforms.RandomResizedCrop(args.img_size, scale=(0.85, 1.0), ratio=(0.9,1.1)))
        if args.hflip_prob > 0:
            steps.append(transforms.RandomHorizontalFlip(p=args.hflip_prob))
        steps.append(transforms.RandomRotation(degrees=args.rotation_deg))
        steps.append(transforms.ColorJitter(brightness=0.10, contrast=0.10))
    steps.append(ToRGB())
    steps.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(steps)

# ---------------------------------------------------------------------------
# Helper to build model by backbone name
# ---------------------------------------------------------------------------
def build_model(backbone: str):
    if backbone == "efficientnet_b4":
        weights = EfficientNet_B4_Weights.DEFAULT
        model = efficientnet_b4(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)
    elif backbone == "densenet169":
        weights = DenseNet169_Weights.DEFAULT
        model = densenet169(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 2)
    else:
        raise ValueError(f"Backbone desconocido: {backbone}")
    return model, weights

# ---------------------------------------------------------------------------
# Main training script
# ---------------------------------------------------------------------------
def parse_args():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="TB Detection – EfficientNet/DenseNet + U-Net + enhancement")
    # Directories
    parser.add_argument("--data-dir", type=str, default=os.path.join(base_dir, "..", "data_prepared_mixed"))
    parser.add_argument("--external-dir", type=str, default=os.path.join(base_dir, "..", "data2"))
    parser.add_argument("--model-dir", type=str, default=os.path.join(base_dir, "..", "models"))
    parser.add_argument("--output-name", type=str, default="",
                        help="Nombre del checkpoint final (.pt). Si vacio, usa <backbone>_tb_best.pt")
    # Training
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--fast-amd", action="store_true")
    # WHO-TPP thresholds
    parser.add_argument("--min-sensitivity", type=float, default=0.90)
    parser.add_argument("--min-specificity", type=float, default=0.70)
    parser.add_argument("--threshold-policy", type=str, default="balanced",
                        choices=["who_tpp", "strict", "balanced", "hybrid"])
    parser.add_argument("--selection-policy", type=str, default="clinical",
                        choices=["auc", "clinical"],
                        help="Criterio para guardar mejor checkpoint durante entrenamiento.")
    # Enhancement
    parser.add_argument("--enhancement-mode", type=str, default="clahe_gamma",
                        choices=["none","histeq","clahe","gamma","complement","bcet",
                                 "clahe_gamma","clahe_unsharp","tricanal"])
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--clahe-tile-grid", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=1.1)
    parser.add_argument("--bcet-target-mean", type=float, default=110.0)
    parser.add_argument("--unsharp-sigma", type=float, default=1.0)
    parser.add_argument("--unsharp-amount", type=float, default=1.0)
    # Lung segmentation
    parser.add_argument("--lung-segmentation-mode", type=str, default="attention_unet",
                        choices=["none","heuristic","unet","attention_unet"])
    parser.add_argument("--lung-segmentation-outside-scale", type=float, default=0.08)
    parser.add_argument("--lung-unet-checkpoint", type=str,
                        default=os.path.join(base_dir, "..", "models", "lung_attention_unet_best.pt"))
    parser.add_argument("--lung-mask-cache", action="store_true",
                        help="Cachea mascaras pulmonares en RAM para acelerar epochs posteriores.")
    parser.add_argument("--lung-mask-cache-max-items", type=int, default=2500)
    # Augmentation
    parser.add_argument("--hflip-prob", type=float, default=0.0)
    parser.add_argument("--rotation-deg", type=float, default=5.0)
    # Evaluation splits
    parser.add_argument("--test-internal-subdir", type=str, default="test_1")
    parser.add_argument("--test-external-subdir", type=str, default="test_2")
    parser.add_argument("--val-external-subdir", type=str, default="val_2")
    parser.add_argument("--val2-weight", type=float, default=0.5)
    parser.add_argument("--enable-external-eval", action="store_true")
    # Backbone selection  <-- NUEVO
    parser.add_argument("--backbone", type=str, default="efficientnet_b4",
                        choices=["efficientnet_b4", "densenet169"],
                        help="Red troncal del modelo.")
    # Score-CAM
    parser.add_argument("--make-cam-grid", action="store_true")
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
    # Apical weight
    parser.add_argument("--apical-weight", type=float, default=1.8)
    # Mixed precision
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision (CUDA only)")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Validate ranges
    for attr, lo, hi in [
        ("val2_weight",0,1), ("hflip_prob",0,1),
        ("scorecam_activation_quantile",0,0.95), ("cam_alpha",0,1),
        ("segment_outside_scale",0,1), ("lung_segmentation_outside_scale",0,1),
        ("apical_weight",1.0,5.0)
    ]:
        val = getattr(args, attr)
        if not (lo <= val <= hi):
            raise ValueError(f"--{attr.replace('_','-')} debe estar en [{lo},{hi}]")

    # Directories
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val_1")
    if not os.path.isdir(val_dir):
        val_dir = os.path.join(args.data_dir, "val")
    val_ext_dir = os.path.join(args.data_dir, args.val_external_subdir) if args.enable_external_eval else None
    test_int_dir = os.path.join(args.data_dir, args.test_internal_subdir)
    test_dir = test_int_dir if os.path.isdir(test_int_dir) else os.path.join(args.data_dir, "test")
    test_ext_dir = os.path.join(args.data_dir, args.test_external_subdir) if args.enable_external_eval else None

    for p in [train_dir, val_dir, test_dir]:
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Missing directory: {p}")

    os.makedirs(args.model_dir, exist_ok=True)
    device, device_name = select_device()
    logging.info(f"Using device: {device_name} ({device})")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if args.fast_amd and device_name == "DirectML":
        args.img_size = min(args.img_size, 300)
        args.batch_size = max(args.batch_size, 12)

    # Build enhancer and lung segmenter
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
        args.lung_unet_checkpoint,
        use_cache=args.lung_mask_cache,
        cache_max_items=args.lung_mask_cache_max_items,
    )
    logging.info(f"Enhancement: {args.enhancement_mode} / Segmentation: {args.lung_segmentation_mode}")
    if args.lung_segmentation_mode in ("unet", "attention_unet") and not os.path.isfile(args.lung_unet_checkpoint):
        logging.warning("U-Net checkpoint not found, falling back to heuristic segmentation.")
        lung_segmenter = HeuristicLungSegmentationEnhancer(args.lung_segmentation_outside_scale)
    if args.num_workers > 0 and args.lung_segmentation_mode in ("unet", "attention_unet"):
        logging.warning(
            "num_workers>0 con segmentacion U-Net puede degradar rendimiento/estabilidad en Windows. "
            "Forzando num_workers=0 para evitar cuelgues de multiprocessing."
        )
        args.num_workers = 0

    # Build model (nuevo)
    model, weights = build_model(args.backbone)
    logging.info(f"Using backbone: {args.backbone}")
    mean = weights.transforms().mean
    std = weights.transforms().std

    train_transform = build_transforms(args, enhancer, lung_segmenter, mean, std, True)
    eval_transform = build_transforms(args, enhancer, lung_segmenter, mean, std, False)

    # Datasets with collage filtering
    TrainDataset = FilteredImageFolder if not args.allow_collage_image else datasets.ImageFolder
    train_dataset = TrainDataset(train_dir, transform=train_transform, allow_collage=args.allow_collage_image)
    val_dataset = FilteredImageFolder(val_dir, transform=eval_transform, allow_collage=args.allow_collage_image)
    test_dataset = FilteredImageFolder(test_dir, transform=eval_transform, allow_collage=args.allow_collage_image)

    val_ext_dataset = None
    if val_ext_dir and os.path.isdir(val_ext_dir):
        val_ext_dataset = FilteredImageFolder(val_ext_dir, transform=eval_transform, allow_collage=args.allow_collage_image)
    ext_test_dataset = None
    if test_ext_dir and os.path.isdir(test_ext_dir):
        ext_test_dataset = FilteredImageFolder(test_ext_dir, transform=eval_transform, allow_collage=args.allow_collage_image)

    class_to_idx = train_dataset.class_to_idx
    if len(class_to_idx) != 2:
        raise ValueError("Se esperaban exactamente 2 clases (TB y NORMAL).")
    logging.info(f"Classes: {class_to_idx}")

    tb_idx_train = get_tb_index(class_to_idx)
    tb_idx_val = get_tb_index(val_dataset.class_to_idx)
    tb_idx_test = get_tb_index(test_dataset.class_to_idx)
    tb_idx_val_ext = get_tb_index(val_ext_dataset.class_to_idx) if val_ext_dataset else None
    tb_idx_ext = get_tb_index(ext_test_dataset.class_to_idx) if ext_test_dataset else None

    # Weighted sampling
    class_counts = np.bincount(np.asarray(train_dataset.targets), minlength=len(class_to_idx))
    sample_weights = [1.0 / max(1, class_counts[t]) for t in train_dataset.targets]
    train_sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), len(sample_weights), replacement=True)

    dl_kw = {"num_workers": args.num_workers, "pin_memory": device.type == "cuda",
             "persistent_workers": args.num_workers > 0, "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None}
    train_loader = DataLoader(train_dataset, args.batch_size, sampler=train_sampler, **{k:v for k,v in dl_kw.items() if v is not None})
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, **{k:v for k,v in dl_kw.items() if v is not None})
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, **{k:v for k,v in dl_kw.items() if v is not None})
    val_ext_loader = DataLoader(val_ext_dataset, args.batch_size, shuffle=False, **{k:v for k,v in dl_kw.items() if v is not None}) if val_ext_dataset else None
    ext_loader = DataLoader(ext_test_dataset, args.batch_size, shuffle=False, **{k:v for k,v in dl_kw.items() if v is not None}) if ext_test_dataset else None

    logging.info(f"Train: NORMAL={class_counts[0]}, TB={class_counts[1]}")

    # Move model to device
    model = model.to(device)
    # Freeze feature extractor initially
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr_head, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    best_monitor_score = -1.0
    best_state = None
    best_epoch = 0
    wait = 0
    head_epochs = max(1, int(args.epochs * 0.35))

    # Training loop
    for epoch in range(1, args.epochs + 1):
        if epoch == head_epochs + 1:
            for p in model.features[-3:].parameters():
                p.requires_grad = True
            optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                    lr=args.lr_finetune, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
            logging.info(f"Fine-tuning started at epoch {epoch}")

        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    loss = criterion(model(images), labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        val_loss, val_true, val_prob = run_inference(model, val_loader, device, criterion, tb_idx_val)
        val_auc = roc_auc_score(val_true, val_prob) if len(np.unique(val_true)) > 1 else 0.0
        monitor_auc = val_auc
        val2_auc = None
        cal_true, cal_prob = [val_true], [val_prob]
        if val_ext_loader and tb_idx_val_ext is not None:
            _, v2t, v2p = run_inference(model, val_ext_loader, device, criterion, tb_idx_val_ext)
            val2_auc = roc_auc_score(v2t, v2p) if len(np.unique(v2t)) > 1 else 0.0
            monitor_auc = (1 - args.val2_weight) * val_auc + args.val2_weight * val2_auc
            cal_true.append(v2t)
            cal_prob.append(v2p)

        # Selección de checkpoint por score clínico o por AUC.
        val_true_all = np.concatenate(cal_true)
        val_prob_all = np.concatenate(cal_prob)
        epoch_thr, _ = find_best_threshold(
            val_true_all, val_prob_all,
            args.min_sensitivity, args.min_specificity,
            args.threshold_policy,
        )
        epoch_metrics = compute_binary_metrics(val_true_all, val_prob_all, epoch_thr)
        clinical_score = compute_clinical_score(epoch_metrics, args.min_sensitivity, args.min_specificity)
        monitor_score = clinical_score if args.selection_policy == "clinical" else monitor_auc

        scheduler.step(monitor_auc)
        logging.info(
            f"Epoch {epoch:02d}/{args.epochs} | loss={train_loss:.4f} | val_auc={val_auc:.4f}"
            + (f" | val2_auc={val2_auc:.4f}" if val2_auc else "")
            + f" | thr={epoch_thr:.3f} | f1={epoch_metrics['f1']:.4f} | bal={epoch_metrics['balanced_accuracy']:.4f}"
            + f" | sens={epoch_metrics['sensitivity']:.4f} | spec={epoch_metrics['specificity']:.4f}"
            + f" | monitor={monitor_score:.4f}"
        )

        if monitor_score > best_monitor_score:
            best_monitor_score = monitor_score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                logging.info("Early stopping triggered.")
                break

    if best_state is None:
        raise RuntimeError("No best model found.")
    model.load_state_dict(best_state)
    model.to(device)

    # Threshold selection
    _, vt, vp = run_inference(model, val_loader, device, criterion, tb_idx_val)
    cal_true, cal_prob = [vt], [vp]
    if val_ext_loader and tb_idx_val_ext is not None:
        _, v2t, v2p = run_inference(model, val_ext_loader, device, criterion, tb_idx_val_ext)
        cal_true.append(v2t); cal_prob.append(v2p)
    cal_true_all = np.concatenate(cal_true)
    cal_prob_all = np.concatenate(cal_prob)
    threshold, thr_note = find_best_threshold(cal_true_all, cal_prob_all,
                                              args.min_sensitivity, args.min_specificity,
                                              args.threshold_policy)
    logging.info(f"Threshold: {threshold:.3f} ({thr_note})")

    metrics_summary = {
        "threshold": threshold, "threshold_rule": thr_note,
        "enhancement_mode": args.enhancement_mode,
        "lung_segmentation_mode": args.lung_segmentation_mode,
        "lung_mask_cache": bool(args.lung_mask_cache),
        "backbone": args.backbone,
        "selection_policy": args.selection_policy,
        "best_monitor_score": float(best_monitor_score),
        "best_epoch": int(best_epoch),
    }

    # Test evaluation
    _, tt, tp = run_inference(model, test_loader, device, criterion, tb_idx_test)
    metrics_summary["test_1"] = print_eval_block("TEST 1", tt, tp, threshold)
    if ext_loader and tb_idx_ext is not None:
        _, et, ep = run_inference(model, ext_loader, device, criterion, tb_idx_ext)
        metrics_summary["test_2"] = print_eval_block("TEST 2 (externo)", et, ep, threshold)

    # Score-CAM grid
    if args.make_cam_grid:
        tb_class = next(n for n, i in class_to_idx.items() if i == tb_idx_train)
        ref_img = args.cam_grid_image or _find_first_image_in_class(test_dir, tb_class)
        if ref_img and os.path.isfile(ref_img):
            model_stem = os.path.splitext(args.output_name)[0] if args.output_name else f"{args.backbone}_tb_best"
            cam_out = args.cam_grid_output or os.path.join(args.model_dir, f"{model_stem}_cam_grid.png")
            model.eval()
            try:
                generate_cam_grid_figure(model, device, tb_idx_train, ref_img, cam_out, mean, std, args)
                metrics_summary["cam_grid"] = {"source": ref_img, "output": cam_out, "apical_weight": args.apical_weight}
            except Exception as e:
                logging.error(f"Error generating CAM grid: {e}")
        else:
            logging.warning("No reference image found for Score-CAM.")

    # Save model and metrics
    output_name = args.output_name.strip()
    if output_name:
        if not output_name.lower().endswith(".pt"):
            output_name = f"{output_name}.pt"
    else:
        output_name = f"{args.backbone}_tb_best.pt"
    model_stem = os.path.splitext(output_name)[0]
    model_path = os.path.join(args.model_dir, output_name)
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_to_idx_train": class_to_idx,
        "tb_index_train": tb_idx_train,
        "img_size": args.img_size,
        "threshold": threshold,
        "mean": mean, "std": std,
        "enhancement_mode": args.enhancement_mode,
        "lung_segmentation_mode": args.lung_segmentation_mode,
        "lung_segmentation_outside_scale": args.lung_segmentation_outside_scale,
        "lung_unet_checkpoint": args.lung_unet_checkpoint,
        "backbone": args.backbone,
        "apical_weight": args.apical_weight,
    }, model_path)
    metrics_path = os.path.join(args.model_dir, f"{model_stem}_training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    logging.info(f"Model saved to {model_path}\nMetrics saved to {metrics_path}.")

if __name__ == "__main__":
    main()
