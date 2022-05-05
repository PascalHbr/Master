import torch
import torch.nn as nn
from pytorchvideo.models.head import ResNetBasicHead
import pytorchvideo.models.x3d


class X3D(nn.Module):
    def __init__(self, num_classes, pretrained=True, freeze=False, keep_head=False, device=None):
        super(X3D, self).__init__()
        self.net = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=pretrained)

        # Freeze baseline
        if freeze:
            for param in self.net.parameters():
                param.requires_grad = False

        # Replace classification head
        if not keep_head:
            cls_head = pytorchvideo.models.x3d.create_x3d_head(dim_in=192, dim_inner=432, dim_out=2048, activation=None,
                                                               num_classes=num_classes, pool_kernel_size=(16, 7, 7))
            self.net.blocks._modules['5'] = cls_head

    def forward(self, video):
        out = self.net(video)

        return out