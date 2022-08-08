import torch
import torch.nn as nn


model_urls = {
    "elannet": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet.pth",
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
                 act_type='silu',             # activation
                 depthwise=False):
        super(Conv, self).__init__()
        convs = []
        if depthwise:
            # depthwise conv
            convs.append(nn.Conv2d(c1, c1, kernel_size=k, stride=s, padding=p, dilation=d, groups=c1, bias=False))
            convs.append(nn.BatchNorm2d(c1))
            if act_type is not None:
                if act_type == 'silu':
                    convs.append(nn.SiLU(inplace=True))
                elif act_type == 'lrelu':
                    convs.append(nn.LeakyReLU(0.1, inplace=True))

            # pointwise conv
            convs.append(nn.Conv2d(c1, c2, kernel_size=1, stride=s, padding=0, dilation=d, groups=1, bias=False))
            convs.append(nn.BatchNorm2d(c2))
            if act_type is not None:
                if act_type == 'silu':
                    convs.append(nn.SiLU(inplace=True))
                elif act_type == 'lrelu':
                    convs.append(nn.LeakyReLU(0.1, inplace=True))

        else:
            convs.append(nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, dilation=d, groups=1, bias=False))
            convs.append(nn.BatchNorm2d(c2))
            if act_type is not None:
                if act_type == 'silu':
                    convs.append(nn.SiLU(inplace=True))
                elif act_type == 'lrelu':
                    convs.append(nn.LeakyReLU(0.1, inplace=True))
            
        self.convs = nn.Sequential(*convs)


    def forward(self, x):
        return self.convs(x)


class ELANBlock(nn.Module):
    """
    ELAN BLock of YOLOv7's backbone
    """
    def __init__(self, in_dim, out_dim, expand_ratio=0.5, model_size='large', act_type='silu', depthwise=False):
        super(ELANBlock, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        if model_size == 'large':
            depth = 2
        elif model_size == 'tiny':
            depth = 1
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type)
        self.cv2 = Conv(in_dim, inter_dim, k=1, act_type=act_type)
        self.cv3 = nn.Sequential(*[
            Conv(inter_dim, inter_dim, k=3, p=1, act_type=act_type, depthwise=depthwise)
            for _ in range(depth)
        ])
        self.cv4 = nn.Sequential(*[
            Conv(inter_dim, inter_dim, k=3, p=1, act_type=act_type, depthwise=depthwise)
            for _ in range(depth)
        ])

        self.out = Conv(inter_dim*4, out_dim, k=1)



    def forward(self, x):
        """
        Input:
            x: [B, C, H, W]
        Output:
            out: [B, 2C, H, W]
        """
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)

        # [B, C, H, W] -> [B, 2C, H, W]
        out = self.out(torch.cat([x1, x2, x3, x4], dim=1))

        return out


class DownSample(nn.Module):
    def __init__(self, in_dim, act_type='silu'):
        super().__init__()
        inter_dim = in_dim // 2
        self.mp = nn.MaxPool2d((2, 2), 2)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type)
        self.cv2 = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type),
            Conv(inter_dim, inter_dim, k=3, p=1, s=2, act_type=act_type)
        )

    def forward(self, x):
        """
        Input:
            x: [B, C, H, W]
        Output:
            out: [B, C, H//2, W//2]
        """
        # [B, C, H, W] -> [B, C//2, H//2, W//2]
        x1 = self.cv1(self.mp(x))
        x2 = self.cv2(x)

        # [B, C, H//2, W//2]
        out = torch.cat([x1, x2], dim=1)

        return out


# ELANNet of YOLOv7
class ELANNet(nn.Module):
    """
    ELAN-Net of YOLOv7.
    """
    def __init__(self, depthwise=False, model_size='large'):
        super(ELANNet, self).__init__()
        
        if model_size == 'large':
            # large backbone
            self.layer_1 = nn.Sequential(
                Conv(3, 32, k=3, p=1, act_type='silu', depthwise=depthwise),      
                Conv(32, 64, k=3, p=1, s=2, act_type='silu', depthwise=depthwise),
                Conv(64, 64, k=3, p=1, act_type='silu', depthwise=depthwise)                                                   # P1/2
            )
            self.layer_2 = nn.Sequential(   
                Conv(64, 128, k=3, p=1, s=2, act_type='silu', depthwise=depthwise),             
                ELANBlock(in_dim=128, out_dim=256, expand_ratio=0.5, act_type='silu', depthwise=depthwise)                     # P2/4
            )
            self.layer_3 = nn.Sequential(
                DownSample(in_dim=256, act_type='silu'),             
                ELANBlock(in_dim=256, out_dim=512, expand_ratio=0.5, act_type='silu', depthwise=depthwise)                     # P3/8
            )
            self.layer_4 = nn.Sequential(
                DownSample(in_dim=512, act_type='silu'),             
                ELANBlock(in_dim=512, out_dim=1024, expand_ratio=0.5, act_type='silu', depthwise=depthwise)                    # P4/16
            )
            self.layer_5 = nn.Sequential(
                DownSample(in_dim=1024, act_type='silu'),             
                ELANBlock(in_dim=1024, out_dim=1024, expand_ratio=0.25, act_type='silu', depthwise=depthwise)                  # P5/32
            )

        elif model_size == 'tiny':
            # tiny backbone
            self.layer_1 = Conv(3, 32, k=3, p=1, s=2, act_type='lrelu', depthwise=depthwise)                                   # P1/2

            self.layer_2 = nn.Sequential(   
                Conv(32, 64, k=3, p=1, s=2, act_type='lrelu', depthwise=depthwise),             
                ELANBlock(in_dim=64, out_dim=64, expand_ratio=0.5,
                        model_size='tiny', act_type='lrelu', depthwise=depthwise)                                              # P2/4
            )
            self.layer_3 = nn.Sequential(
                nn.MaxPool2d((2, 2), 2),             
                ELANBlock(in_dim=64, out_dim=128, expand_ratio=0.5,
                        model_size='tiny', act_type='lrelu', depthwise=depthwise)                                              # P3/8
            )
            self.layer_4 = nn.Sequential(
                nn.MaxPool2d((2, 2), 2),             
                ELANBlock(in_dim=128, out_dim=256, expand_ratio=0.5,
                        model_size='tiny', act_type='lrelu', depthwise=depthwise)                                              # P4/16
            )
            self.layer_5 = nn.Sequential(
                nn.MaxPool2d((2, 2), 2),             
                ELANBlock(in_dim=256, out_dim=512, expand_ratio=0.5,
                        model_size='tiny', act_type='lrelu', depthwise=depthwise)                                               # P5/32
            )


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


# build ELAN-Net
def build_elannet(pretrained=False, model_size='large'):
    # model
    backbone = ELANNet(model_size=model_size)
    if model_size == 'large':
        arch = 'elannet'
        feat_dims = [512, 1024, 1024]
    elif model_size == 'tiny':
        arch = 'elannet_tiny'
        feat_dims = [128, 256, 512]

    # load weight
    if pretrained:
        print('Loading pretrained weight ...')
        url = model_urls[arch]
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True)
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        # model state dict
        model_state_dict = backbone.state_dict()
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

        backbone.load_state_dict(checkpoint_state_dict)

    return backbone, feat_dims


if __name__ == '__main__':
    import time
    model, feats = build_elannet(pretrained=True, model_size='large')
    x = torch.randn(1, 3, 224, 224)
    t0 = time.time()
    outputs = model(x)
    t1 = time.time()
    print('Time: ', t1 - t0)
    for k in outputs.keys():
        print(outputs[k].shape)
