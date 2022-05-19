import torch
import torch.nn as nn
import torch.nn.functional as F
from ..basic.conv import Conv
from ..basic.bottleneck_csp import BottleneckCSP



# YoloPaFPN
class YoloPaFPN(nn.Module):
    def __init__(self, 
                 in_dims=[256, 512, 1024],
                 depth=3, 
                 depthwise=False,
                 norm_type='BN',
                 act_type='lrelu'):
        super(YoloPaFPN, self).__init__()
        self.in_dims = in_dims
        c3, c4, c5 = in_dims
        nblocks = int(depth)

        self.head_conv_0 = Conv(c5, c5//2, k=1, norm_type=norm_type, act_type=act_type)  # 10
        self.head_csp_0 = BottleneckCSP(c4 + c5//2, c4, n=nblocks, shortcut=False, depthwise=depthwise, norm_type=norm_type, act_type=act_type)

        # P3/8-small
        self.head_conv_1 = Conv(c4, c4//2, k=1, norm_type=norm_type, act_type=act_type)  # 14
        self.head_csp_1 = BottleneckCSP(c3 + c4//2, c3, n=nblocks, shortcut=False, depthwise=depthwise, norm_type=norm_type, act_type=act_type)

        # P4/16-medium
        self.head_conv_2 = Conv(c3, c3, k=3, p=1, s=2, depthwise=depthwise, norm_type=norm_type, act_type=act_type)
        self.head_csp_2 = BottleneckCSP(c3 + c4//2, c4, n=nblocks, shortcut=False, depthwise=depthwise, norm_type=norm_type, act_type=act_type)

        # P8/32-large
        self.head_conv_3 = Conv(c4, c4, k=3, p=1, s=2, depthwise=depthwise, norm_type=norm_type, act_type=act_type)
        self.head_csp_3 = BottleneckCSP(c4 + c5//2, c5, n=nblocks, shortcut=False, depthwise=depthwise)


    def forward(self, features):
        c3, c4, c5 = features

        c6 = self.head_conv_0(c5)
        c7 = F.interpolate(c6, scale_factor=2.0)   # s32->s16
        c8 = torch.cat([c7, c4], dim=1)
        c9 = self.head_csp_0(c8)
        # P3/8
        c10 = self.head_conv_1(c9)
        c11 = F.interpolate(c10, scale_factor=2.0)   # s16->s8
        c12 = torch.cat([c11, c3], dim=1)
        c13 = self.head_csp_1(c12)  # to det
        # p4/16
        c14 = self.head_conv_2(c13)
        c15 = torch.cat([c14, c10], dim=1)
        c16 = self.head_csp_2(c15)  # to det
        # p5/32
        c17 = self.head_conv_3(c16)
        c18 = torch.cat([c17, c6], dim=1)
        c19 = self.head_csp_3(c18)  # to det

        return [c13, c16, c19] # [P3, P4, P5]
