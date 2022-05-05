import torch
import torch.nn as nn
from pytorchvideo.models.head import ResNetBasicHead


class SlowR50(nn.Module):
    def __init__(self, device):
        super(SlowR50, self).__init__()
        self.net = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)

        # Replace classification head
        cls_head = ResNetBasicHead(pool=nn.AvgPool3d(kernel_size=(8, 7, 7), stride=(1, 1, 1), padding=(0, 0, 0)),
                                   dropout=nn.Dropout(p=0.5, inplace=False),
                                   proj=nn.Identity(),
                                   output_pool=nn.AdaptiveAvgPool3d(output_size=1))
        self.net.blocks._modules['5'] = cls_head

    def forward(self, video):
        out = self.net(video)

        return out