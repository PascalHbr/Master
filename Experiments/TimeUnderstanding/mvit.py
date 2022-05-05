import torch.nn as nn
from pytorchvideo.models.head import VisionTransformerBasicHead, SequencePool
from pytorchvideo.models.hub import mvit_base_16x4


class MViT(nn.Module):
    def __init__(self, num_classes, pretrained=True, freeze=False, keep_head=False, device=None):
        super(MViT, self).__init__()
        self.net = mvit_base_16x4(pretrained=pretrained)

        # Freeze baseline
        if freeze:
            for param in self.net.parameters():
                param.requires_grad = False

        # Replace classification head
        if not keep_head:
            cls_head = VisionTransformerBasicHead(sequence_pool=SequencePool("mean"),
                                                  dropout=nn.Dropout(p=0.5, inplace=False),
                                                  proj=nn.Linear(in_features=768, out_features=num_classes, bias=True))
            self.net.head = cls_head

    def forward(self, video):
        out = self.net(video)

        return out