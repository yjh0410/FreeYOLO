"""
    This is a CSPDarkNet-53 with Mish.
"""
import os
import torch
import torch.nn as nn


model_urls = {
    "cspdarknet53": "https://github.com/yjh0410/PyTorch_YOLO-Family/releases/download/yolo-weight/cspdarknet53.pth",
}



def ConvNormActivation(inplanes,
                       planes,
                       kernel_size=3,
                       stride=1,
                       padding=0,
                       dilation=1,
                       groups=1):
    """
    A help function to build a 'conv-bn-activation' module
    """
    layers = []
    layers.append(nn.Conv2d(inplanes,
                            planes,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                            groups=groups,
                            bias=False))
    layers.append(nn.BatchNorm2d(planes, eps=1e-4, momentum=0.03))
    layers.append(nn.Mish(inplace=True))
    return nn.Sequential(*layers)


def make_cspdark_layer(block,
                       inplanes,
                       planes,
                       num_blocks,
                       is_csp_first_stage,
                       dilation=1):
    downsample = ConvNormActivation(
        inplanes=planes,
        planes=planes if is_csp_first_stage else inplanes,
        kernel_size=1,
        stride=1,
        padding=0
    )

    layers = []
    for i in range(0, num_blocks):
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes if is_csp_first_stage else inplanes,
                downsample=downsample if i == 0 else None,
                dilation=dilation
            )
        )
    return nn.Sequential(*layers)


class DarkBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 dilation=1,
                 downsample=None):
        """Residual Block for DarkNet.
        This module has the dowsample layer (optional),
        1x1 conv layer and 3x3 conv layer.
        """
        super(DarkBlock, self).__init__()

        self.downsample = downsample

        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-4, momentum=0.03)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-4, momentum=0.03)

        self.conv1 = nn.Conv2d(
            planes,
            inplanes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.conv2 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=False
        )

        self.activation = nn.Mish(inplace=True)

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out += identity

        return out


class CrossStagePartialBlock(nn.Module):
    """CSPNet: A New Backbone that can Enhance Learning Capability of CNN.
    Refer to the paper for more details: https://arxiv.org/abs/1911.11929.
    In this module, the inputs go throuth the base conv layer at the first,
    and then pass the two partial transition layers.
    1. go throuth basic block (like DarkBlock)
        and one partial transition layer.
    2. go throuth the other partial transition layer.
    At last, They are concat into fuse transition layer.
    Args:
        inplanes (int): number of input channels.
        planes (int): number of output channels
        stage_layers (nn.Module): the basic block which applying CSPNet.
        is_csp_first_stage (bool): Is the first stage or not.
            The number of input and output channels in the first stage of
            CSPNet is different from other stages.
        dilation (int): conv dilation
        stride (int): stride for the base layer
    """

    def __init__(self,
                 inplanes,
                 planes,
                 stage_layers,
                 is_csp_first_stage,
                 dilation=1,
                 stride=2):
        super(CrossStagePartialBlock, self).__init__()

        self.base_layer = ConvNormActivation(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation
        )
        self.partial_transition1 = ConvNormActivation(
            inplanes=planes,
            planes=inplanes if not is_csp_first_stage else planes,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.stage_layers = stage_layers

        self.partial_transition2 = ConvNormActivation(
            inplanes=inplanes if not is_csp_first_stage else planes,
            planes=inplanes if not is_csp_first_stage else planes,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.fuse_transition = ConvNormActivation(
            inplanes=planes if not is_csp_first_stage else planes * 2,
            planes=planes,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        x = self.base_layer(x)

        out1 = self.partial_transition1(x)

        out2 = self.stage_layers(x)
        out2 = self.partial_transition2(out2)

        out = torch.cat([out2, out1], dim=1)
        out = self.fuse_transition(out)

        return out


class CSPDarkNet53(nn.Module):
    """CSPDarkNet backbone.
    Refer to the paper for more details: https://arxiv.org/pdf/1804.02767
    Args:
        depth (int): Depth of Darknet, from {53}.
        num_stages (int): Darknet stages, normally 5.
        with_csp (bool): Use cross stage partial connection or not.
        out_features (List[str]): Output features.
        norm_type (str): type of normalization layer.
        res5_dilation (int): dilation for the last stage
    """

    def __init__(self, res5_dilation=False):
        super(CSPDarkNet53, self).__init__()
     
        self.block =  DarkBlock
        self.stage_blocks = (1, 2, 8, 8, 4)
        self.with_csp = True
        self.inplanes = 32
        self.res5_dilation = res5_dilation

        self.backbone = nn.ModuleDict()
        self.layer_names = []
        # First stem layer
        self.backbone["conv1"] = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False)
        self.backbone["bn1"] = nn.BatchNorm2d(self.inplanes, eps=1e-4, momentum=0.03)
        self.backbone["act1"] = nn.Mish(inplace=True)

        for i, num_blocks in enumerate(self.stage_blocks):
            planes = 64 * 2 ** i
            dilation = 1
            stride = 2
            if i == 4 and res5_dilation:
                dilation = 2
                stride = 1

            layer = make_cspdark_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                is_csp_first_stage=True if i == 0 else False,
                dilation=dilation
            )
            layer = CrossStagePartialBlock(
                self.inplanes,
                planes,
                stage_layers=layer,
                is_csp_first_stage=True if i == 0 else False,
                dilation=dilation,
                stride=stride
            )
            self.inplanes = planes
            layer_name = 'layer{}'.format(i + 1)
            self.backbone[layer_name]=layer
            self.layer_names.append(layer_name)


    def forward(self, x):
        outs = []
        x = self.backbone["conv1"](x)
        x = self.backbone["bn1"](x)
        x = self.backbone["act1"](x)

        for i, layer_name in enumerate(self.layer_names):
            layer = self.backbone[layer_name]
            x = layer(x)
            outs.append(x)

        outputs = {
            'layer2': outs[-3],
            'layer3': outs[-2],
            'layer4': outs[-1]
        }
        return outputs


# Build CSPDarkNet
def build_cspdarknet53(pretrained=False, res5_dilation=False):
    # build backbone
    backbone = CSPDarkNet53(res5_dilation=res5_dilation)
    feat_dims = [256, 512, 1024]

    # load weight
    if pretrained:
        print('Loading pretrained weight ...')
        url = model_urls['cspdarknet53']
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
    model, feats = build_cspdarknet53(pretrained=True, res5_dilation=False)
    x = torch.randn(1, 3, 224, 224)
    t0 = time.time()
    outputs = model(x)
    t1 = time.time()
    print('Time: ', t1 - t0)
    for k in outputs.keys():
        print(outputs[k].shape)
