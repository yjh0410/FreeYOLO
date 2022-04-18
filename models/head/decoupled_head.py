import torch
import torch.nn as nn

from ..basic.conv import Conv


class DecoupledHead(nn.Module):
    def __init__(self, 
                 head_dim=256,
                 num_cls_head=4,
                 num_reg_head=4,
                 act_type='relu',
                 norm_type=''):
        super().__init__()

        print('==============================')
        print('Head: Decoupled Head')

        self.cls_feats = nn.Sequential(*[Conv(head_dim, 
                                              head_dim, 
                                              k=3, p=1, s=1, 
                                              act_type=act_type, 
                                              norm_type=norm_type) for _ in range(num_cls_head)])
        self.reg_feats = nn.Sequential(*[Conv(head_dim, 
                                              head_dim, 
                                              k=3, p=1, s=1, 
                                              act_type=act_type, 
                                              norm_type=norm_type) for _ in range(num_reg_head)])

        self._init_weight()


    def _init_weight(self):
        # init weight of detection head
        for m in [self.cls_feats, self.reg_feats]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        """
            in_feats: (Tensor) [B, C, H, W]
        """
        cls_feats = self.cls_feats(x)
        reg_feats = self.reg_feats(x)

        return cls_feats, reg_feats
