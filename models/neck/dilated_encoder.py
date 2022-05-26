import torch
import torch.nn as nn

from ..basic.conv import Conv
from utils import weight_init


# Dilated Encoder
class Bottleneck(nn.Module):
    def __init__(self, 
                 in_dim, 
                 dilation=1, 
                 expand_ratio=0.25,
                 act_type='relu',
                 norm_type='BN',
                 depthwise=False):
        super(Bottleneck, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.branch = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(inter_dim, inter_dim, k=3,
                 p=dilation, d=dilation,
                 act_type=act_type,
                 norm_type=norm_type,
                 depthwise=depthwise),
            Conv(inter_dim, in_dim, k=1, act_type=act_type, norm_type=norm_type)
        )

    def forward(self, x):
        return x + self.branch(x)


class DilatedEncoder(nn.Module):
    """ DilateEncoder """
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 expand_ratio=0.25, 
                 dilation_list=[2, 4, 6, 8],
                 act_type='relu',
                 norm_type='BN',
                 depthwise=False):
        super(DilatedEncoder, self).__init__()
        self.projector = nn.Sequential(
            Conv(in_dim, out_dim, k=1, act_type=None, norm_type=norm_type),
            Conv(out_dim, out_dim, k=3, p=1, act_type=None, norm_type=norm_type, depthwise=depthwise)
        )
        encoders = []
        for d in dilation_list:
            encoders.append(Bottleneck(
                in_dim=out_dim, 
                dilation=d, 
                expand_ratio=expand_ratio, 
                act_type=act_type,
                norm_type=norm_type,
                depthwise=depthwise))
        self.encoders = nn.Sequential(*encoders)

        self._init_weight()

    def _init_weight(self):
        for m in self.projector:
            if isinstance(m, nn.Conv2d):
                weight_init.c2_xavier_fill(m)
                weight_init.c2_xavier_fill(m)
            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.encoders.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.projector(x)
        x = self.encoders(x)

        return x
