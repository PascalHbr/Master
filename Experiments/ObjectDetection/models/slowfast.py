import torch
import torch.nn as nn
from pytorchvideo.models.head import ResNetBasicHead


class SlowFastR50(nn.Module):
    def __init__(self, num_classes, pretrained=True, freeze=False, keep_head=False, device=None):
        super(SlowFastR50, self).__init__()
        self.net = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=pretrained)

        # Freeze baseline
        if freeze:
            for param in self.net.parameters():
                param.requires_grad = False

        # Replace classification head
        if not keep_head:
            cls_head = ResNetBasicHead(dropout=nn.Dropout(p=0.5, inplace=False),
                                       proj=nn.Linear(in_features=2304, out_features=num_classes, bias=True),
                                       output_pool=nn.AdaptiveAvgPool3d(output_size=1))
            self.net.blocks._modules['6'] = cls_head

    def forward(self, video):
        out = self.net(video)

        return out