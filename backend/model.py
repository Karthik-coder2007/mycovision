import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class CoordinateAttention(nn.Module):
    def __init__(self, channels=1280, reduction=32):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv_h = nn.Conv2d(hidden, channels, 1, bias=True)
        self.conv_w = nn.Conv2d(hidden, channels, 1, bias=True)

    def forward(self, x):
        _, _, h, w = x.shape
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).transpose(2, 3)
        y = torch.cat([x_h, x_w], dim=2)
        y = F.hardswish(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.transpose(2, 3)
        return x * torch.sigmoid(self.conv_h(x_h)) * torch.sigmoid(self.conv_w(x_w))

class BackgroundAwareHead(nn.Module):
    def __init__(self, channels=1280, num_classes=9):
        super().__init__()
        self.background_branch = nn.Sequential(
            nn.Conv2d(channels, 320, 3, padding=1, bias=False),
            nn.BatchNorm2d(320), nn.ReLU(inplace=True),
            nn.Conv2d(320, 1, 1)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, features):
        mask = torch.sigmoid(self.background_branch(features))
        weighted = features * (1.0 + mask)
        logits = self.classifier(self.dropout(self.pool(weighted).flatten(1)))
        return logits, mask

class ProposedMushroomModel(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.backbone = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=0)
        self.coordinate_attention = CoordinateAttention(1280)
        self.background_head = BackgroundAwareHead(1280, num_classes)
        self.last_features = None

    def forward(self, x):
        features = self.backbone.forward_features(x)
        if features.requires_grad:
            features.retain_grad()
        self.last_features = features
        features = self.coordinate_attention(features)
        return self.background_head(features)
