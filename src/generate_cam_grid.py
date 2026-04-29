import argparse
import os
from types import SimpleNamespace

import torch
from torchvision.models import efficientnet_b4

from train import generate_cam_grid_figure


def parse_args() -> argparse.Namespace:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_path = os.path.abspath(os.path.join(base_dir, "..", "models", "efficientnet_b4_tb_best.pt"))
    default_output_path = os.path.abspath(os.path.join(base_dir, "..", "models", "tb_enhancement_cam_grid.png"))
    parser = argparse.ArgumentParser(description="Genera grilla Score-CAM sin reentrenar el modelo.")
    parser.add_argument("--model-path", type=str, default=default_model_path, help="Checkpoint .pt entrenado.")
    parser.add_argument("--image-path", type=str, required=True, help="Radiografia CXR individual.")
    parser.add_argument("--output-path", type=str, default=default_output_path)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--clahe-tile-grid", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=1.1)
    parser.add_argument("--bcet-target-mean", type=float, default=110.0)
    parser.add_argument("--unsharp-sigma", type=float, default=1.0)
    parser.add_argument("--unsharp-amount", type=float, default=1.0)
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
    parser.add_argument(
        "--lung-segmentation-mode",
        type=str,
        default="none",
        choices=["none", "heuristic"],
        help="Modo de mascara pulmonar para fila segmentada del CAM.",
    )
    parser.add_argument("--allow-collage-image", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"No existe el checkpoint: {args.model_path}")
    if not os.path.isfile(args.image_path):
        raise FileNotFoundError(f"No existe la imagen CXR: {args.image_path}")

    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
    class_to_idx = ckpt.get("class_to_idx_train", {"NORMAL": 0, "TB": 1})
    model = efficientnet_b4(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, len(class_to_idx))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    tb_idx = int(ckpt.get("tb_index_train", class_to_idx.get("TB", 1)))
    mean = ckpt.get("mean", [0.485, 0.456, 0.406])
    std = ckpt.get("std", [0.229, 0.224, 0.225])
    img_size = int(ckpt.get("img_size", 380))

    cam_args = SimpleNamespace(
        img_size=img_size,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid=args.clahe_tile_grid,
        gamma=args.gamma,
        bcet_target_mean=args.bcet_target_mean,
        unsharp_sigma=args.unsharp_sigma,
        unsharp_amount=args.unsharp_amount,
        scorecam_max_maps=args.scorecam_max_maps,
        scorecam_batch_size=args.scorecam_batch_size,
        scorecam_layer=args.scorecam_layer,
        scorecam_activation_quantile=args.scorecam_activation_quantile,
        cam_smooth_sigma=args.cam_smooth_sigma,
        cam_low_percentile=args.cam_low_percentile,
        cam_high_percentile=args.cam_high_percentile,
        cam_threshold=args.cam_threshold,
        cam_alpha=args.cam_alpha,
        segment_outside_scale=args.segment_outside_scale,
        lung_segmentation_mode=args.lung_segmentation_mode,
        allow_collage_image=args.allow_collage_image,
    )

    generate_cam_grid_figure(
        model=model,
        device=device,
        tb_index=tb_idx,
        img_path=args.image_path,
        output_path=args.output_path,
        mean=mean,
        std=std,
        args=cam_args,
    )
    print(f"Grilla guardada en: {args.output_path}")


if __name__ == "__main__":
    main()
