import torch
import torch.nn as nn
from .conv import Conv


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, d=1, e=0.5, depthwise=False, norm_type='BN', act_type='lrelu'):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels            
        self.cv1 = Conv(c1, c_, k=1, norm_type=norm_type, act_type=act_type)
        self.cv2 = Conv(c_, c2, k=3, p=d, d=d, norm_type=norm_type, act_type=act_type, depthwise=depthwise)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5, depthwise=False, norm_type='BN', act_type='lrelu'):
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k=1, norm_type=norm_type, act_type=act_type)
        self.cv2 = Conv(c1, c_, k=1, norm_type=norm_type, act_type=act_type)
        self.cv3 = Conv(2 * c_, c2, k=1, norm_type=norm_type, act_type=act_type)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, e=1.0, depthwise=depthwise, 
                                            norm_type=norm_type, act_type=act_type) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
