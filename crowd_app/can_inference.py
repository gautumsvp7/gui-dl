"""
CAN (Context-Aware Network) inference module.
Loads a trained checkpoint and runs crowd-count prediction on a single image.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class ContextualModule(nn.Module):
    """
    Multi-scale context aggregation with difference-based sigmoid attention.
    Each pooling scale's weight is sigmoid(weight_net(original - scaled)).
    The weighted average is concatenated with the original and bottlenecked.
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
            F.interpolate(stage(feats), size=(h, w), mode='bilinear', align_corners=False)
            for stage in self.scales
        ]
        weights = [self._make_weight(feats, s) for s in multi_scales]
        weight_sum = sum(weights)
        weighted_avg = sum(s * wt for s, wt in zip(multi_scales, weights)) / weight_sum
        combined = torch.cat([weighted_avg, feats], dim=1)
        return self.relu(self.bottleneck(combined))


class CAN(nn.Module):
    """
    Context-Aware Network for crowd counting.
    Frontend: VGG16 conv1-conv4_3 ([:23], 512ch, stride 8)
    Backend:  6 dilated conv layers with BatchNorm
    """

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=None)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])
        self.context = ContextualModule(512, 512)

        backend_cfg = [512, 512, 512, 256, 128, 64]
        self.backend = self._make_backend(backend_cfg, in_channels=512, dilation=2)
        self.output = nn.Conv2d(64, 1, kernel_size=1)

    @staticmethod
    def _make_backend(cfg, in_channels, dilation):
        layers = []
        for v in cfg:
            layers += [
                nn.Conv2d(in_channels, v, kernel_size=3, padding=dilation, dilation=dilation),
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


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

_MODEL_CACHE = {}
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Threshold: expected crowd counts at or below this use the Part B (sparse) model;
# counts above it use the Part A (dense) model.
DENSITY_THRESHOLD = 200


def load_can_model(weights_path):
    """Load a CAN checkpoint and cache it so it is only read from disk once."""
    weights_path = str(weights_path)
    if weights_path in _MODEL_CACHE:
        return _MODEL_CACHE[weights_path]

    model = CAN().to(_DEVICE)
    ckpt = torch.load(weights_path, map_location=_DEVICE)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    _MODEL_CACHE[weights_path] = model
    return model


def select_model(model_a, model_b, expected_count):
    """Return the appropriate model and a human-readable label based on expected_count."""
    if expected_count <= DENSITY_THRESHOLD:
        return model_b, 'CAN – Part B (sparse crowds, ≤{} expected)'.format(DENSITY_THRESHOLD)
    else:
        return model_a, 'CAN – Part A (dense crowds, >{} expected)'.format(DENSITY_THRESHOLD)


def _save_density_map(dm_np, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.imshow(dm_np, cmap='jet')
    ax.axis('off')
    plt.tight_layout(pad=0)
    fig.savefig(str(save_path), bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def predict_crowd(model, image_path, density_map_save_path=None):
    image_path = Path(image_path)

    img = Image.open(image_path).convert('RGB')
    # resize to nearest multiple of 8 (required by stride-8 frontend)
    w, h = img.size
    img = img.resize((round(w / 8) * 8, round(h / 8) * 8), Image.BILINEAR)

    tensor = _preprocess(img).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        density_pred = model(tensor)

    crowd_count = int(round(density_pred.sum().item()))

    saved = False
    if density_map_save_path is not None:
        try:
            dm_np = density_pred.squeeze().cpu().numpy()
            _save_density_map(dm_np, density_map_save_path)
            saved = True
        except Exception as exc:
            print(f'density map save failed: {exc}')

    return {
        'crowd_count':       crowd_count,
        'density_map_saved': saved,
    }
