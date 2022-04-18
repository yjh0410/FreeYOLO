import torch
from thop import profile


def FLOPs_and_Params(model, min_size, max_size, device):
    if min_size is None:
        min_size = max_size
    x = torch.randn(1, 3, min_size, max_size).to(device)
    print('==============================')
    flops, params = profile(model, inputs=(x, ))
    print('==============================')
    print('FLOPs : {:.2f} B'.format(flops / 1e9))
    print('Params : {:.2f} M'.format(params / 1e6))


if __name__ == "__main__":
    pass
