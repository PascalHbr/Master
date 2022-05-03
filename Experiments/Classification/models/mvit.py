import torch.nn as nn
from pytorchvideo.models.head import VisionTransformerBasicHead, SequencePool
from pytorchvideo.models.hub import mvit_base_16x4


class MViT(nn.Module):
    def __init__(self, device):
        super(MViT, self).__init__()
        self.net = mvit_base_16x4(pretrained=True)

        # Replace classification head
        cls_head = VisionTransformerBasicHead(sequence_pool=SequencePool("mean"),
                                              dropout=nn.Dropout(p=0.5, inplace=False),
                                              proj=nn.Identity())
        self.net.head = cls_head

    def forward(self, video):
        out = self.net(video)

        return out