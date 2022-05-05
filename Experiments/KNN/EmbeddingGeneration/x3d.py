import torch
import torch.nn as nn
from pytorchvideo.models.head import ResNetBasicHead
import pytorchvideo.models.x3d


class X3D(nn.Module):
    def __init__(self, device):
        super(X3D, self).__init__()
        self.net = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)

        # Replace classification head
        pool = pytorchvideo.models.x3d.ProjectedPool(
                    pre_conv=nn.Conv3d(
                        in_channels=192, out_channels=432, kernel_size=(1, 1, 1), bias=False
                    ),
                    pre_norm=nn.BatchNorm3d(num_features=432, eps=1e-5, momentum=0.1),
                    pre_act=nn.ReLU(),
                    pool=nn.AvgPool3d((16, 7, 7), stride=1),
                    post_conv=nn.Conv3d(
                        in_channels=432, out_channels=2048, kernel_size=(1, 1, 1), bias=False
                    ),
                    post_norm=None,
                    post_act=nn.ReLU(),
                )
        cls_head = ResNetBasicHead(
            proj=nn.Identity(),
            activation=None,
            pool=pool,
            dropout=None,
            output_pool=nn.AdaptiveAvgPool3d(1),
        )
        self.net.blocks._modules['5'] = cls_head

    def forward(self, video):
        out = self.net(video)

        return out