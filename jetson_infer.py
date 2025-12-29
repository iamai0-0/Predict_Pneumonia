#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""jetson_infer.py (clean)

Jetson Nano inference for Predict_Pneumonia.

This script is intentionally a SINGLE entrypoint (no duplicated main blocks).

- Loads weights saved as PyTorch state_dict (.pt/.pth)
- Auto-detects DenseNet variant (121/169/201/161) from classifier in_features when possible
- Preprocess matches the common Colab pipeline:
    Resize(256) -> CenterCrop(224) -> ToTensor()
  with optional Normalize:
    --normalize none (default)
    --normalize imagenet
- Default class mapping: index 0 = NORMAL, index 1 = PNEUMONIA
  (override via --classes)

Examples:
  python3 jetson_infer.py -i Data/test/PNEUMONIA/xxx.jpeg -w best-model-weighted.pt --cpu
  python3 jetson_infer.py -i Data/test/PNEUMONIA/xxx.jpeg -w best-model-weighted.pt --fp16 --normalize none
"""

import os
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Reduce native crashes on Jetson (threads/BLAS/OpenMP)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


ARCH_BY_INFEATURES = {
    1024: "densenet121",
    1664: "densenet169",
    1920: "densenet201",
    2208: "densenet161",
}
SUPPORTED_ARCHES = ["auto", "densenet121", "densenet169", "densenet201", "densenet161"]

# Default mapping (typical ImageFolder alphabetical: NORMAL=0, PNEUMONIA=1)
DEFAULT_CLASSES = ["NORMAL", "PNEUMONIA"]
NORMALIZE_CHOICES = ["none", "imagenet"]


def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not sd:
        return sd
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def _looks_like_state_dict(obj: Dict[str, Any]) -> bool:
    if not obj:
        return False
    if not all(isinstance(k, str) for k in obj.keys()):
        return False
    return any(k.endswith(".weight") or k.endswith(".bias") for k in obj.keys())


def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    # Direct state_dict
    if isinstance(obj, dict) and _looks_like_state_dict(obj):
        return obj

    # Common checkpoint wrappers
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
            if key in obj and isinstance(obj[key], dict) and _looks_like_state_dict(obj[key]):
                return obj[key]

    raise ValueError(
        "Khong doc duoc state_dict tu file weights. "
        "Hay dam bao ban luu bang torch.save(model.state_dict(), 'xxx.pt')."
    )


def load_state_dict(weights_path: str) -> Dict[str, torch.Tensor]:
    weights_path = str(weights_path)
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    raw = torch.load(weights_path, map_location="cpu")  # load on CPU first
    sd = _extract_state_dict(raw)
    sd = _strip_module_prefix(sd)
    return sd


def infer_arch_and_classes(
    sd: Dict[str, torch.Tensor],
    forced_arch: str = "auto",
) -> Tuple[str, int, Optional[int]]:
    num_classes = 2
    in_features: Optional[int] = None

    if "classifier.weight" in sd:
        w = sd["classifier.weight"]
        if hasattr(w, "shape") and len(w.shape) == 2:
            num_classes = int(w.shape[0])
            in_features = int(w.shape[1])

    if forced_arch != "auto":
        return forced_arch, num_classes, in_features

    if in_features in ARCH_BY_INFEATURES:
        return ARCH_BY_INFEATURES[in_features], num_classes, in_features

    # Fallback if cannot infer
    return "densenet161", num_classes, in_features


def build_model(arch: str, num_classes: int) -> torch.nn.Module:
    if arch == "densenet121":
        model = models.densenet121(pretrained=False)
    elif arch == "densenet169":
        model = models.densenet169(pretrained=False)
    elif arch == "densenet201":
        model = models.densenet201(pretrained=False)
    elif arch == "densenet161":
        model = models.densenet161(pretrained=False)
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.eval()
    return model


def make_preprocess(img_size: int, normalize: str):
    ops = [
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ]
    if normalize == "imagenet":
        ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]))
        desc = f"Resize(256)->CenterCrop({img_size})->ToTensor() | normalize=imagenet"
    else:
        desc = f"Resize(256)->CenterCrop({img_size})->ToTensor() | normalize=none"
    return transforms.Compose(ops), desc


def preprocess_image(image_path: str, tfm) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    return tfm(img).unsqueeze(0)


@torch.no_grad()
def predict(model: torch.nn.Module, x: torch.Tensor, device: torch.device, fp16: bool = False):
    x = x.to(device)
    if fp16 and device.type == "cuda":
        x = x.half()
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    pred_idx = int(torch.argmax(probs).item())
    conf = float(probs[pred_idx].item())
    return pred_idx, conf, [float(p.item()) for p in probs.cpu()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", "-i", required=True, help="Path to a chest X-ray image (jpg/png).")
    ap.add_argument("--weights", "-w", required=True, help="Path to weights (state_dict) .pt/.pth.")
    ap.add_argument("--arch", choices=SUPPORTED_ARCHES, default="auto",
                    help="DenseNet architecture. Use 'auto' to infer from weights.")
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                    help="Class names in index order. Default: NORMAL PNEUMONIA")
    ap.add_argument("--normalize", choices=NORMALIZE_CHOICES, default="none",
                    help="Normalization: none (default) or imagenet")
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    ap.add_argument("--fp16", action="store_true", help="Use FP16 on CUDA (faster, less memory).")
    ap.add_argument("--img-size", type=int, default=224, help="CenterCrop size (default 224).")
    args = ap.parse_args()

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("Device:", device)

    tfm, tfm_desc = make_preprocess(args.img_size, args.normalize)
    print("Preprocess:", tfm_desc)

    # Load weights on CPU first
    sd = load_state_dict(args.weights)
    arch, num_classes, in_features = infer_arch_and_classes(sd, forced_arch=args.arch)
    msg = f"Arch: {arch} | num_classes: {num_classes}"
    if in_features is not None:
        msg += f" | in_features(from weights): {in_features}"
    print(msg)

    classes = list(args.classes)
    if len(classes) != num_classes:
        print(f"Warning: you provided {len(classes)} class names but model has {num_classes} classes.")
        if len(classes) < num_classes:
            classes += [f"class_{i}" for i in range(len(classes), num_classes)]
        else:
            classes = classes[:num_classes]
    print("Class mapping (index->name):", {i: classes[i] for i in range(len(classes))})

    model = build_model(arch=arch, num_classes=num_classes)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print("Warning: missing keys (first 10):", missing[:10])
    if unexpected:
        print("Warning: unexpected keys (first 10):", unexpected[:10])

    model.to(device)
    if args.fp16 and device.type == "cuda":
        model.half()

    x = preprocess_image(str(image_path), tfm)
    pred_idx, conf, probs = predict(model, x, device=device, fp16=args.fp16)

    print("Predict:", classes[pred_idx])
    print("Confidence:", conf)
    print("Probs:", probs)

    # Prevent segfault-at-exit on some Jetson setups
    if device.type == "cuda":
        torch.cuda.synchronize()
    os._exit(0)


if __name__ == "__main__":
    main()

