from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class ContextualModule(nn.Module):
    def __init__(self, features=512, out_features=512, sizes=(1, 2, 3, 6)):
        super(ContextualModule, self).__init__()

        self.scales = nn.ModuleList()
        for size in sizes:
            self.scales.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(size, size)),
                nn.Conv2d(features, features, kernel_size=1, bias=False)
            ))

        self.bottleneck = nn.Conv2d(features * 2, out_features, kernel_size=1)
        self.relu = nn.ReLU()
        self.weight_net = nn.Conv2d(features, features, kernel_size=1)

    def compute_weight(self, feature, scale_feature):
        return torch.sigmoid(self.weight_net(feature - scale_feature))

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)

        scale_outputs = []
        for branch in self.scales:
            out = branch(feats)
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
            scale_outputs.append(out)

        weights = [self.compute_weight(feats, s) for s in scale_outputs]
        total_weight = sum(weights)
        weighted_context = sum(s * w for s, w in zip(scale_outputs, weights)) / total_weight

        out = torch.cat([weighted_context, feats], dim=1)
        return self.relu(self.bottleneck(out))


class CAN(nn.Module):
    def __init__(self):
        super(CAN, self).__init__()

        vgg16 = models.vgg16(weights=None)
        self.frontend = nn.Sequential(*list(vgg16.features.children())[:23])
        self.context_module = ContextualModule(512, 512)

        backend_channels = [512, 512, 512, 256, 128, 64]
        self.backend = self.build_backend(backend_channels, in_ch=512, dilation=2)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def build_backend(self, channel_list, in_ch, dilation):
        layers = []
        for out_ch in channel_list:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            in_ch = out_ch
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.frontend(x)
        x = self.context_module(x)
        x = self.backend(x)
        return self.output_layer(x)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Part A = dense crowds, Part B = sparse; threshold determines which model to use
DENSITY_THRESHOLD = 200


def load_can_model(weights_path):
    model = CAN().to(device)
    checkpoint = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def select_model(model_a, model_b, expected_count):
    if expected_count <= DENSITY_THRESHOLD:
        return model_b, 'CAN Part B (sparse, up to {} people)'.format(DENSITY_THRESHOLD)
    return model_a, 'CAN Part A (dense, > {} people)'.format(DENSITY_THRESHOLD)


def save_density_map(density_map, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.imshow(density_map, cmap='jet')
    ax.axis('off')
    plt.tight_layout(pad=0)
    fig.savefig(str(save_path), bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def predict_crowd(model, image_path, density_map_save_path=None):
    image_path = Path(image_path)
    img = Image.open(image_path).convert('RGB')

    # dimensions must be divisible by 8 due to network stride
    w, h = img.size
    new_w = round(w / 8) * 8
    new_h = round(h / 8) * 8
    img = img.resize((new_w, new_h), Image.BILINEAR)

    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        density_pred = model(tensor)

    crowd_count = int(round(density_pred.sum().item()))

    if density_map_save_path is not None:
        dm_np = density_pred.squeeze().cpu().numpy()
        save_density_map(dm_np, density_map_save_path)

    return crowd_count
