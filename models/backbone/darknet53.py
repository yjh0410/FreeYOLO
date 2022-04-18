import torch
import torch.nn as nn


class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        convs = []
        convs.append(nn.Conv2d(in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation, bias=False))
        convs.append(nn.BatchNorm2d(out_channels))
        convs.append(nn.LeakyReLU(0.1, inplace=True))

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        return self.convs(x)


class ResBlock(nn.Module):
    def __init__(self, ch, nblocks=1):
        super().__init__()
        self.module_list = nn.ModuleList()
        for _ in range(nblocks):
            resblock_one = nn.Sequential(
                Conv_BN_LeakyReLU(ch, ch//2, ksize=1),
                Conv_BN_LeakyReLU(ch//2, ch, ksize=3, padding=1)
            )
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            x = module(x) + x
        return x


class DarkNet_53(nn.Module):
    """
    DarkNet-53.
    """
    def __init__(self):
        super(DarkNet_53, self).__init__()
        # stride = 2
        self.layer_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, ksize=3, padding=1),
            Conv_BN_LeakyReLU(32, 64, ksize=3, padding=1, stride=2),
            ResBlock(64, nblocks=1)
        )
        # stride = 4
        self.layer_2 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, ksize=3, padding=1, stride=2),
            ResBlock(128, nblocks=2)
        )
        # stride = 8
        self.layer_3 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, ksize=3, padding=1, stride=2),
            ResBlock(256, nblocks=8)
        )
        # stride = 16
        self.layer_4 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, ksize=3, padding=1, stride=2),
            ResBlock(512, nblocks=8)
        )
        # stride = 32
        self.layer_5 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, ksize=3, padding=1, stride=2),
            ResBlock(1024, nblocks=4)
        )

        self.freeze()


    def forward(self, x):
        outputs = dict()
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        outputs["layer2"] = c3
        outputs["layer3"] = c4
        outputs["layer4"] = c5

        return outputs


def darknet53():
    """Constructs a darknet-53 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DarkNet_53()
    feats = [256, 512, 1024] # C3, C4, C5

    return model, feats


if __name__ == '__main__':
    x = torch.ones(2, 3, 64, 64)
    m, f = darknet53()
    out = m(x)

    for k in out.keys():
        y = out[k]
        print(y.size())
