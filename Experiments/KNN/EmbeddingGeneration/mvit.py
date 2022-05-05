import torch.nn as nn
from pytorchvideo.models.hub import mvit_base_16x4


class MViT(nn.Module):
    def __init__(self, device):
        super(MViT, self).__init__()
        self.net = mvit_base_16x4(pretrained=True)

    def forward(self, video):
        out = self.net(video)

        return out