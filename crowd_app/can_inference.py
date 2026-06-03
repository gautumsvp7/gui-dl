from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# CAN architecture based on the paper:
# "Context-Aware Crowd Counting" (Liu et al., 2019)
# The key idea is to use multi-scale context features weighted by how
# different they are from the original — so scales that add useful info
# contribute more than scales that are redundant.

class ContextualModule(nn.Module):
    """
    This is the context-aware module from the CAN paper.
    It pools features at 4 different scales (1x1, 2x2, 3x3, 6x6),
    then weights each scale based on how much it differs from the input.
    """
    def __init__(self, features=512, out_features=512, sizes=(1, 2, 3, 6)):
        super(ContextualModule, self).__init__()

        # one pooling branch per scale
        self.scales = nn.ModuleList()
        for size in sizes:
            self.scales.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(size, size)),
                nn.Conv2d(features, features, kernel_size=1, bias=False)
            ))

        # after concatenating weighted context + original, compress back down
        self.bottleneck = nn.Conv2d(features * 2, out_features, kernel_size=1)
        self.relu = nn.ReLU()

        # used to compute the attention weights per scale
        self.weight_net = nn.Conv2d(features, features, kernel_size=1)

    def compute_weight(self, feature, scale_feature):
        # weight = sigmoid(W * (feature - scale_feature))
        # scales that look very different from the input get higher weight
        return torch.sigmoid(self.weight_net(feature - scale_feature))

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)

        # upsample all scale features back to original spatial size
        scale_outputs = []
        for branch in self.scales:
            out = branch(feats)
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
            scale_outputs.append(out)

        # compute attention weights for each scale
        weights = [self.compute_weight(feats, s) for s in scale_outputs]
        total_weight = sum(weights)

        # weighted average across scales
        weighted_context = sum(s * w for s, w in zip(scale_outputs, weights)) / total_weight

        # concat with original features and project
        out = torch.cat([weighted_context, feats], dim=1)
        return self.relu(self.bottleneck(out))


class CAN(nn.Module):
    """
    Full CAN model.
    - Frontend: first 10 conv layers of VGG-16 (up to pool3), extracts feature maps
    - Context module: multi-scale context aggregation
    - Backend: dilated conv layers to estimate the density map
    """
    def __init__(self):
        super(CAN, self).__init__()

        # reuse VGG16 frontend — take layers up to the 23rd child (conv4_3)
        vgg16 = models.vgg16(weights=None)
        self.frontend = nn.Sequential(*list(vgg16.features.children())[:23])

        self.context_module = ContextualModule(512, 512)

        # backend: dilated convolutions so we don't lose too much spatial resolution
        backend_channels = [512, 512, 512, 256, 128, 64]
        self.backend = self.build_backend(backend_channels, in_ch=512, dilation=2)

        # final 1x1 conv outputs the density map (1 channel)
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


# standard ImageNet normalisation values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# threshold for choosing Part A vs Part B model
# Part A is trained on denser crowds, Part B on sparser ones
DENSITY_THRESHOLD = 200


def load_can_model(weights_path):
    """Load a saved CAN model from a .pth file."""
    model = CAN().to(device)
    checkpoint = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def select_model(model_a, model_b, expected_count):
    """
    Pick Part A or Part B based on the user's expected crowd size.
    If they expect <= DENSITY_THRESHOLD people, use Part B (sparse).
    Otherwise use Part A (dense).
    """
    if expected_count <= DENSITY_THRESHOLD:
        return model_b, 'CAN Part B (sparse, up to {} people)'.format(DENSITY_THRESHOLD)
    return model_a, 'CAN Part A (dense, > {} people)'.format(DENSITY_THRESHOLD)


def save_density_map(density_map, save_path):
    """Save the density map as a heatmap image (jet colormap)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.imshow(density_map, cmap='jet')
    ax.axis('off')
    plt.tight_layout(pad=0)
    fig.savefig(str(save_path), bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def predict_crowd(model, image_path, density_map_save_path=None):
    """
    Run the model on an image and return the estimated crowd count.
    Also saves the density map heatmap if a path is given.
    """
    image_path = Path(image_path)

    img = Image.open(image_path).convert('RGB')

    # resize so dimensions are divisible by 8 (required by the network stride)
    w, h = img.size
    new_w = round(w / 8) * 8
    new_h = round(h / 8) * 8
    img = img.resize((new_w, new_h), Image.BILINEAR)

    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        density_pred = model(tensor)

    # sum over the density map gives total estimated count
    crowd_count = int(round(density_pred.sum().item()))

    if density_map_save_path is not None:
        dm_np = density_pred.squeeze().cpu().numpy()
        save_density_map(dm_np, density_map_save_path)

    return crowd_count
