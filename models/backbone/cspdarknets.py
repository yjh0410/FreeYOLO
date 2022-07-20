import torch
import torch.nn as nn
import torch.nn.functional as F
import os


model_urls = {
    "cspd-l": "",
}


# Basic conv layer
class Conv(nn.Module):
    def __init__(self, 
                 c1,                   # in channels
                 c2,                   # out channels 
                 k=1,                  # kernel size 
                 p=0,                  # padding
                 s=1,                  # padding
                 d=1,                  # dilation
                 act=True,             # activation
                 depthwise=False):
        super(Conv, self).__init__()
        convs = []
        if depthwise:
            # depthwise conv
            convs.append(nn.Conv2d(c1, c1, kernel_size=k, stride=s, padding=p, dilation=d, groups=c1, bias=False))
            convs.append(nn.BatchNorm2d(c1))
            if act:
                convs.append(nn.SiLU(inplace=True))

            # pointwise conv
            convs.append(nn.Conv2d(c1, c2, kernel_size=1, stride=s, padding=0, dilation=d, groups=1, bias=False))
            convs.append(nn.BatchNorm2d(c2))
            if act:
                convs.append(nn.SiLU(inplace=True))

        else:
            convs.append(nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, dilation=d, groups=1, bias=False))
            convs.append(nn.BatchNorm2d(c2))
            if act:
                convs.append(nn.SiLU(inplace=True))
            
        self.convs = nn.Sequential(*convs)


    def forward(self, x):
        return self.convs(x)


class ResidualBlock(nn.Module):
    """
    basic residual block for CSP-Darknet
    """
    def __init__(self, in_ch, depthwise=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv(in_ch, in_ch, k=1, depthwise=depthwise)
        self.conv2 = Conv(in_ch, in_ch, k=3, p=1, depthwise=depthwise)

    def forward(self, x):
        h = self.conv2(self.conv1(x))
        out = x + h

        return out


class CSPStage(nn.Module):
    def __init__(self, c1, n=1, depthwise=False):
        super(CSPStage, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k=1)
        self.cv2 = Conv(c1, c_, k=1)
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(in_ch=c_, depthwise=depthwise)
            for _ in range(n)
            ])
        self.cv3 = Conv(2 * c_, c1, k=1)

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.res_blocks(self.cv2(x))

        return self.cv3(torch.cat([y1, y2], dim=1))


# CSPDarknet
class CSPDarkNet(nn.Module):
    """
    CSPDarknet_53.
    """
    def __init__(self, width=1.0, depth=1.0, depthwise=False, num_classes=1000):
        super(CSPDarkNet, self).__init__()
        # init w&d cfg
        basic_w_cfg = [32, 64, 128, 256, 512, 1024]
        basic_d_cfg = [1, 3, 9, 9, 6]
        # init w&d cfg
        w_cfg = [int(w*width) for w in basic_w_cfg]
        d_cfg = [int(d*depth) for d in basic_d_cfg]
        d_cfg[0] = 1
        print('=================================')
        print('Width: ', w_cfg)
        print('Depth: ', d_cfg)
        print('=================================')

        self.layer_1 = nn.Sequential(
            Conv(3, w_cfg[0], k=3, p=1, depthwise=depthwise),      
            Conv(w_cfg[0], w_cfg[1], k=3, p=1, s=2, depthwise=depthwise),
            CSPStage(c1=w_cfg[1], n=d_cfg[0], depthwise=depthwise)                       # p1/2
        )
        self.layer_2 = nn.Sequential(   
            Conv(w_cfg[1], w_cfg[2], k=3, p=1, s=2, depthwise=depthwise),             
            CSPStage(c1=w_cfg[2], n=d_cfg[1], depthwise=depthwise)                      # P2/4
        )
        self.layer_3 = nn.Sequential(
            Conv(w_cfg[2], w_cfg[3], k=3, p=1, s=2, depthwise=depthwise),             
            CSPStage(c1=w_cfg[3], n=d_cfg[2], depthwise=depthwise)                      # P3/8
        )
        self.layer_4 = nn.Sequential(
            Conv(w_cfg[3], w_cfg[4], k=3, p=1, s=2, depthwise=depthwise),             
            CSPStage(c1=w_cfg[4], n=d_cfg[3], depthwise=depthwise)                      # P4/16
        )
        self.layer_5 = nn.Sequential(
            Conv(w_cfg[4], w_cfg[5], k=3, p=1, s=2, depthwise=depthwise),             
            CSPStage(c1=w_cfg[5], n=d_cfg[4], depthwise=depthwise)                     # P5/32
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(w_cfg[5], num_classes)


    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        outputs = {
            'layer2': c3,
            'layer3': c4,
            'layer4': c5
        }
        return outputs


def cspdarknet_n(pretrained=False, width=0.25, depth=0.34):
    # model
    model = CSPDarkNet(width=width, depth=depth, depthwise=True)

    # load weight
    if pretrained:
        print('Loading pretrained weight ...')
        url = model_urls['cspd-n']
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True)
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        # model state dict
        model_state_dict = model.state_dict()
        # check
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
            else:
                checkpoint_state_dict.pop(k)
                print(k)

        model.load_state_dict(checkpoint_state_dict)

    return model


def cspdarknet_t(pretrained=False, width=0.25, depth=0.34):
    # model
    model = CSPDarkNet(width=width, depth=depth)

    # load weight
    if pretrained:
        print('Loading pretrained weight ...')
        url = model_urls['cspd-t']
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True)
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        # model state dict
        model_state_dict = model.state_dict()
        # check
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
            else:
                checkpoint_state_dict.pop(k)
                print(k)

        model.load_state_dict(checkpoint_state_dict)

    return model


def cspdarknet_s(pretrained=False, width=0.5, depth=0.34):
    # model
    model = CSPDarkNet(width=width, depth=depth)

    # load weight
    if pretrained:
        print('Loading pretrained weight ...')
        url = model_urls['cspd-s']
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True)
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        # model state dict
        model_state_dict = model.state_dict()
        # check
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
            else:
                checkpoint_state_dict.pop(k)
                print(k)

        model.load_state_dict(checkpoint_state_dict)

    return model


def cspdarknet_m(pretrained=False, width=0.75, depth=0.67):
    # model
    model = CSPDarkNet(width=width, depth=depth)

    # load weight
    if pretrained:
        print('Loading pretrained weight ...')
        url = model_urls['cspd-m']
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True)
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        # model state dict
        model_state_dict = model.state_dict()
        # check
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
            else:
                checkpoint_state_dict.pop(k)
                print(k)

        model.load_state_dict(checkpoint_state_dict)

    return model


def cspdarknet_l(pretrained=False, width=1.0, depth=1.0):
    # model
    model = CSPDarkNet(width=width, depth=depth)

    # load weight
    if pretrained:
        print('Loading pretrained weight ...')
        url = model_urls['cspd-l']
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True)
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        # model state dict
        model_state_dict = model.state_dict()
        # check
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
            else:
                checkpoint_state_dict.pop(k)
                print(k)

        model.load_state_dict(checkpoint_state_dict)

    return model


def cspdarknet_x(pretrained=False, width=1.25, depth=1.34):
    # model
    model = CSPDarkNet(width=width, depth=depth)

    # load weight
    if pretrained:
        print('Loading pretrained weight ...')
        url = model_urls['cspd-x']
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True)
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        # model state dict
        model_state_dict = model.state_dict()
        # check
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
            else:
                checkpoint_state_dict.pop(k)
                print(k)

        model.load_state_dict(checkpoint_state_dict)

    return model


# Build CSP-DarkNets
def build_cspd(model_name='cspd-l', pretrained=False):
    if model_name == 'cspd-n':
        backbone = cspdarknet_n(pretrained)
        feat_dims = [64, 128, 256]

    elif model_name == 'cspd-t':
        backbone = cspdarknet_t(pretrained)
        feat_dims = [64, 128, 256]

    elif model_name == 'cspd-s':
        backbone = cspdarknet_s(pretrained)
        feat_dims = [128, 256, 512]

    elif model_name == 'cspd-m':
        backbone = cspdarknet_m(pretrained)
        feat_dims = [192, 384, 768]

    elif model_name == 'cspd-l':
        backbone = cspdarknet_l(pretrained)
        feat_dims = [256, 512, 1024]

    elif model_name == 'cspd-x':
        backbone = cspdarknet_x(pretrained)
        feat_dims = [320, 640, 1280]

    return backbone, feat_dims



if __name__ == '__main__':
    import time
    net = cspdarknet_x()
    x = torch.randn(1, 3, 224, 224)
    t0 = time.time()
    y = net(x)
    t1 = time.time()
    print('Time: ', t1 - t0)