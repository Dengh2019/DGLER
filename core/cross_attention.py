# FILE: ./DGLER/core/cross_attention.py

import torch
import torch.nn as nn

class CrossModalChannelAttention(nn.Module):
    """
    Cross-Modal Channel Attention, CMCA
    Specifically designed for the RAFT optical flow architecture, it only performs channel-level re-calibration without disrupting the spatial local features.
   
    """
    def __init__(self, channels=256, reduction=16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
       
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, fmap_lidar, fmap_event):
        b, c, _, _ = fmap_lidar.size()

       
        y_lidar = self.avg_pool(fmap_lidar).view(b, c)
        y_event = self.avg_pool(fmap_event).view(b, c)

        # Cross-Excitation
        
        weight_lidar = self.sigmoid(self.mlp(y_event)).view(b, c, 1, 1)
        weight_event = self.sigmoid(self.mlp(y_lidar)).view(b, c, 1, 1)

        
        out_lidar = fmap_lidar + fmap_lidar * weight_lidar
        out_event = fmap_event + fmap_event * weight_event

        return out_lidar, out_event