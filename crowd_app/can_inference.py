from __future__ import annotations

"""
crowd_app/can_inference.py
--------------------------
Self-contained CAN (Context-Aware Network) inference module.

Usage
-----
    from crowd_app.can_inference import load_can_model, predict_crowd

    # Load once at startup (expensive — use Django's AppConfig.ready() or
    # a module-level singleton, not inside each request).
    model = load_can_model(weights_path)

    # Call per request
    result = predict_crowd(model, image_path, save_density_map_to)
    # result = {'crowd_count': int, 'density_map_saved': bool}

Architecture (v2 — matches can_checkpoints_A_v2/best_model.pth)
------------
  Frontend : VGG16 [:23] — conv1 through conv4_3, 512 ch, spatial stride 8
  Context  : Difference-based sigmoid attention (scales/bottleneck/weight_net)
  Backend  : 6 dilated conv layers (dilation=2) WITH BatchNorm
             [512→512, 512→512, 512→512, 512→256, 256→128, 128→64]
  Output   : 1×1 conv → density map (sum ≈ crowd count)
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')           # non-interactive backend — safe for Django
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# 1. Model definition  (must be identical to the v2 training notebook)
# ─────────────────────────────────────────────────────────────────────────────

class ContextualModule(nn.Module):
    """
    Difference-based sigmoid attention across 4 pooling scales.
    Each scale computes sigmoid(weight_net(original - scaled)) as its
    attention weight.  The weighted average is concatenated with the
    original features and compressed via a 1024→512 bottleneck conv.
    """

    def __init__(self, features=512, out_features=512, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.scales = nn.ModuleList(
            [self._make_scale(features, s) for s in sizes]
        )
        self.bottleneck = nn.Conv2d(features * 2, out_features, kernel_size=1)
        self.relu = nn.ReLU()
        self.weight_net = nn.Conv2d(features, features, kernel_size=1)

    @staticmethod
    def _make_scale(features, size):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(size, size)),
            nn.Conv2d(features, features, kernel_size=1, bias=False),
        )

    def _make_weight(self, feature, scale_feature):
        return torch.sigmoid(self.weight_net(feature - scale_feature))

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        multi_scales = [
            F.interpolate(stage(feats), size=(h, w),
                          mode='bilinear', align_corners=False)
            for stage in self.scales
        ]
        weights = [self._make_weight(feats, s) for s in multi_scales]
        weight_sum = sum(weights)
        weighted_avg = sum(s * wt for s, wt in zip(multi_scales, weights)) / weight_sum
        combined = torch.cat([weighted_avg, feats], dim=1)   # (B, 1024, H, W)
        return self.relu(self.bottleneck(combined))           # (B,  512, H, W)


class CAN(nn.Module):
    """Context-Aware Network — v2 architecture (512 ch, 6-layer BN backend)."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=None)   # weights loaded from checkpoint
        # [:23] = conv1 through conv4_3, before pool4 → 512 ch at H/8
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])
        self.context  = ContextualModule(512, 512)

        backend_cfg = [512, 512, 512, 256, 128, 64]
        self.backend = self._make_backend(backend_cfg, in_channels=512, dilation=2)
        self.output  = nn.Conv2d(64, 1, kernel_size=1)

    @staticmethod
    def _make_backend(cfg, in_channels, dilation):
        layers = []
        for v in cfg:
            layers += [
                nn.Conv2d(in_channels, v, kernel_size=3,
                          padding=dilation, dilation=dilation),
                nn.BatchNorm2d(v),
                nn.ReLU(inplace=True),
            ]
            in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.frontend(x)
        x = self.context(x)
        x = self.backend(x)
        return self.output(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pre-processing  (must match training normalisation)
# ─────────────────────────────────────────────────────────────────────────────

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def _pad_to_multiple_of_8(img: Image.Image) -> Image.Image:
    """Resize image so both dimensions are multiples of 8 (required by stride-8 frontend)."""
    w, h = img.size
    new_h = max(8, round(h / 8) * 8)
    new_w = max(8, round(w / 8) * 8)
    if new_h != h or new_w != w:
        img = img.resize((new_w, new_h), Image.BILINEAR)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# 3. Public API
# ─────────────────────────────────────────────────────────────────────────────

# Module-level cache so the model is loaded only once per worker process.
_MODEL_CACHE: dict = {}

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_can_model(weights_path: str | Path) -> CAN:
    """
    Load a v2 CAN model from a checkpoint saved by the training notebook.

    Parameters
    ----------
    weights_path : path to the .pth checkpoint (e.g. 'model/best_model.pth')

    Returns
    -------
    CAN model in eval mode, moved to GPU if available.
    """
    weights_path = str(weights_path)
    if weights_path in _MODEL_CACHE:
        return _MODEL_CACHE[weights_path]

    model = CAN().to(_DEVICE)
    ckpt = torch.load(weights_path, map_location=_DEVICE)

    # Checkpoints saved by the notebook wrap the state dict under 'model_state_dict'
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    _MODEL_CACHE[weights_path] = model
    return model


def _save_density_map(dm_np: np.ndarray, save_path: str | Path) -> None:
    """Save a density map array as a jet-colourmap PNG."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.imshow(dm_np, cmap='jet')
    ax.axis('off')
    plt.tight_layout(pad=0)
    fig.savefig(str(save_path), bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def predict_crowd(
    model: CAN,
    image_path: str | Path,
    density_map_save_path: str | Path | None = None,
) -> dict:
    """
    Run CAN inference on a single image.

    Parameters
    ----------
    model                 : loaded CAN instance (from load_can_model)
    image_path            : absolute path to the input image
    density_map_save_path : if given, the density map PNG is saved here

    Returns
    -------
    dict with keys:
        crowd_count        (int)   — predicted head count
        density_map_saved  (bool)  — True if the map was saved successfully
    """
    image_path = Path(image_path)

    # Load and pre-process
    img = Image.open(image_path).convert('RGB')
    img = _pad_to_multiple_of_8(img)
    tensor = _preprocess(img).unsqueeze(0).to(_DEVICE)   # (1, 3, H, W)

    # Forward pass
    with torch.no_grad():
        density_pred = model(tensor)                      # (1, 1, H/8, W/8)

    crowd_count = int(round(density_pred.sum().item()))

    # Save density map if requested
    saved = False
    if density_map_save_path is not None:
        try:
            dm_np = density_pred.squeeze().cpu().numpy()
            _save_density_map(dm_np, density_map_save_path)
            saved = True
        except Exception as exc:
            # Non-fatal — the results page falls back to the SVG placeholder
            print(f'[can_inference] density map save failed: {exc}')

    return {
        'crowd_count':       crowd_count,
        'density_map_saved': saved,
    }
