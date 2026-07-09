import torch
import torch.nn as nn
import torch.nn.functional as F
from core.extractor import ResidualBlock

class SFT_Layer(nn.Module):
    """
    Spatial Feature Transform (SFT) Module
    """
    def __init__(self, depth_channels=64, event_channels=64):
        super(SFT_Layer, self).__init__()
        self.conv_gamma = nn.Sequential(
            nn.Conv2d(depth_channels, depth_channels, 3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(depth_channels, event_channels, 3, padding=1)
        )
        self.conv_beta = nn.Sequential(
            nn.Conv2d(depth_channels, depth_channels, 3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(depth_channels, event_channels, 3, padding=1)
        )

    def forward(self, event_feat, depth_feat):
        # Generate the scaling factor gamma and the translation factor beta
        gamma = self.conv_gamma(depth_feat)
        beta = self.conv_beta(depth_feat)
        # Perform spatial adaptive affine transformation on the event feature
        return event_feat * (1 + gamma) + beta

class DepthGuidedEventEncoder(nn.Module):
    """
     SFT-DGLER V3
    """
    def __init__(self, in_channels=5, output_dim=256, norm_fn='instance', dropout=0.0):
        super(DepthGuidedEventEncoder, self).__init__()
        self.norm_fn = norm_fn

        def get_norm(planes):
            if self.norm_fn == 'group':
                return nn.GroupNorm(num_groups=8, num_channels=planes)
            elif self.norm_fn == 'batch':
                return nn.BatchNorm2d(planes)
            elif self.norm_fn == 'instance':
                return nn.InstanceNorm2d(planes)
            else:
                return nn.Sequential()

        # 1. Depth stem
        self.depth_stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            get_norm(32),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            get_norm(64),
            nn.LeakyReLU(0.1, True)
        )

        # 2. event stem
        self.event_stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            get_norm(64),
            nn.LeakyReLU(0.1, True)
        )

        # 3. core：SFT layer
        self.sft = SFT_Layer(depth_channels=64, event_channels=64)

        # 4. Subsequent residual blocks
        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None

        # 5. Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        self.in_planes = dim
        return nn.Sequential(layer1, layer2)

    def forward(self, event_voxel, depth_map):
        # Extract depth features
        depth_feat = self.depth_stem(depth_map)
        
        # Extract the initial spatiotemporal features of the event
        event_feat = self.event_stem(event_voxel)

        # Inject the deep prior into the event features using SFT-DGLER
        x = self.sft(event_feat, depth_feat)

        # residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        return x