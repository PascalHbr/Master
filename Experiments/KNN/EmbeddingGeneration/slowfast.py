import torch
import torch.nn as nn
from pytorchvideo.models.head import ResNetBasicHead


class SlowFastR50(nn.Module):
    def __init__(self, device):
        super(SlowFastR50, self).__init__()
        self.net = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)

        # Replace classification head
        cls_head = ResNetBasicHead(dropout=nn.Dropout(p=0.5, inplace=False),
                                   proj=nn.Identity(),
                                   output_pool=nn.AdaptiveAvgPool3d(output_size=1))
        self.net.blocks._modules['6'] = cls_head

    def forward(self, video):
        out = self.net(video)

        return out