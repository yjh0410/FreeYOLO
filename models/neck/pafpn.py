import torch
import torch.nn as nn
import torch.nn.functional as F
from ..basic.conv import Conv
from ..basic.repconv import RepConv
from ..basic.bottleneck_csp import BottleneckCSP


class ELANBlock(nn.Module):
    """
    ELAN BLock of YOLOv7's head
    """
    def __init__(self, in_dim, out_dim, expand_ratio=0.5, depthwise=False, act_type='silu', norm_type='BN'):
        super(ELANBlock, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        inter_dim2 = int(inter_dim * expand_ratio)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv3 = Conv(inter_dim, inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.cv4 = Conv(inter_dim2, inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.cv5 = Conv(inter_dim2, inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.cv6 = Conv(inter_dim2, inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)

        self.out = Conv(inter_dim*2+inter_dim2*4, out_dim, k=1)


    def forward(self, x):
        """
        Input:
            x: [B, C_in, H, W]
        Output:
            out: [B, C_out, H, W]
        """
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)
        x5 = self.cv5(x4)
        x6 = self.cv6(x5)

        # [B, C_in, H, W] -> [B, C_out, H, W]
        out = self.out(torch.cat([x1, x2, x3, x4, x5, x6], dim=1))

        return out


class DownSample(nn.Module):
    def __init__(self, in_dim, depthwise=False, act_type='silu', norm_type='BN'):
        super().__init__()
        inter_dim = in_dim
        self.mp = nn.MaxPool2d((2, 2), 2)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(inter_dim, inter_dim, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )

    def forward(self, x):
        """
        Input:
            x: [B, C, H, W]
        Output:
            out: [B, 2C, H//2, W//2]
        """
        # [B, C, H, W] -> [B, C//2, H//2, W//2]
        x1 = self.cv1(self.mp(x))
        x2 = self.cv2(x)

        # [B, C, H//2, W//2]
        out = torch.cat([x1, x2], dim=1)

        return out


# PaFPN-CSP
class PaFPNCSP(nn.Module):
    def __init__(self, 
                 in_dims=[256, 512, 1024],
                 out_dim=256,
                 depth=3, 
                 depthwise=False,
                 norm_type='BN',
                 act_type='lrelu'):
        super(PaFPNCSP, self).__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        c3, c4, c5 = in_dims
        nblocks = int(depth)

        self.head_conv_0 = Conv(c5, c5//2, k=1, norm_type=norm_type, act_type=act_type)  # 10
        self.head_csp_0 = BottleneckCSP(c4 + c5//2, c4, n=nblocks,
                                        shortcut=False, depthwise=depthwise,
                                        norm_type=norm_type, act_type=act_type)

        # P3/8-small
        self.head_conv_1 = Conv(c4, c4//2, k=1, norm_type=norm_type, act_type=act_type)  # 14
        self.head_csp_1 = BottleneckCSP(c3 + c4//2, c3, n=nblocks,
                                        shortcut=False, depthwise=depthwise,
                                        norm_type=norm_type, act_type=act_type)

        # P4/16-medium
        self.head_conv_2 = Conv(c3, c3, k=3, p=1, s=2, depthwise=depthwise, norm_type=norm_type, act_type=act_type)
        self.head_csp_2 = BottleneckCSP(c3 + c4//2, c4, n=nblocks,
                                        shortcut=False, depthwise=depthwise,
                                        norm_type=norm_type, act_type=act_type)

        # P8/32-large
        self.head_conv_3 = Conv(c4, c4, k=3, p=1, s=2, depthwise=depthwise, norm_type=norm_type, act_type=act_type)
        self.head_csp_3 = BottleneckCSP(c4 + c5//2, c5, n=nblocks,
                                        shortcut=False, depthwise=depthwise,
                                        norm_type=norm_type, act_type=act_type)

        # output proj layers
        if self.out_dim is not None:
            self.out_layers = nn.ModuleList([
                Conv(in_dim, self.out_dim, k=1,
                     norm_type=norm_type, act_type=act_type)
                     for in_dim in in_dims
                     ])

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

        out_feats = [c13, c16, c19] # [P3, P4, P5]
        # output proj layers
        if self.out_dim is not None:
            out_feats_proj = []
            for feat, layer in zip(out_feats, self.out_layers):
                out_feats_proj.append(layer(feat))
            return out_feats_proj

        return out_feats


# PaFPN-ELAN (YOLOv7's)
class PaFPNELAN(nn.Module):
    def __init__(self, 
                 in_dims=[256, 512, 512],
                 out_dim=[256, 512, 1024],
                 depthwise=False,
                 norm_type='BN',
                 act_type='silu'):
        super(PaFPNELAN, self).__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        c3, c4, c5 = in_dims

        # top dwon
        ## P5 -> P4
        self.cv1 = Conv(c5, 256, k=1, norm_type=norm_type, act_type=act_type)
        self.cv2 = Conv(c4, 256, k=1, norm_type=norm_type, act_type=act_type)
        self.head_elan_1 = ELANBlock(in_dim=256 + 256,
                                     out_dim=256,
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        # P4 -> P3
        self.cv3 = Conv(256, 128, k=1, norm_type=norm_type, act_type=act_type)
        self.cv4 = Conv(c3, 128, k=1, norm_type=norm_type, act_type=act_type)
        self.head_elan_2 = ELANBlock(in_dim=128 + 128,
                                     out_dim=128,  # 128
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        # bottom up
        # P3 -> P4
        self.mp1 = DownSample(128, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.head_elan_3 = ELANBlock(in_dim=256 + 256,
                                     out_dim=256,  # 256
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        # P4 -> P5
        self.mp2 = DownSample(256, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.head_elan_4 = ELANBlock(in_dim=512 + 512,
                                     out_dim=512,  # 512
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        # RepConv
        self.repconv_1 = RepConv(128, out_dim[0], k=3, s=1, p=1)
        self.repconv_2 = RepConv(256, out_dim[1], k=3, s=1, p=1)
        self.repconv_3 = RepConv(512, out_dim[2], k=3, s=1, p=1)


    def forward(self, features):
        c3, c4, c5 = features

        # Top down
        ## P5 -> P4
        c6 = self.cv1(c5)
        c7 = F.interpolate(c6, scale_factor=2.0)
        c8 = torch.cat([c7, self.cv2(c4)], dim=1)
        c9 = self.head_elan_1(c8)
        ## P4 -> P3
        c10 = self.cv3(c9)
        c11 = F.interpolate(c10, scale_factor=2.0)
        c12 = torch.cat([c11, self.cv4(c3)], dim=1)
        c13 = self.head_elan_2(c12)

        # Bottom up
        # p3 -> P4
        c14 = self.mp1(c13)
        c15 = torch.cat([c14, c9], dim=1)
        c16 = self.head_elan_3(c15)
        # P4 -> P5
        c17 = self.mp2(c16)
        c18 = torch.cat([c17, c5], dim=1)
        c19 = self.head_elan_4(c18)

        # RepCpnv
        c20 = self.repconv_1(c13)
        c21 = self.repconv_2(c16)
        c22 = self.repconv_3(c19)

        out_feats = [c20, c21, c22] # [P3, P4, P5]

        return out_feats
