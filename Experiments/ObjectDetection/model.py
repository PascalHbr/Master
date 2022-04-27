import pytorchvideo.models.x3d
import torch.nn.functional as F
from pytorchvideo.models.head import ResNetBasicHead, VisionTransformerBasicHead, SequencePool
from pytorchvideo.models.hub import mvit_base_16x4
from load_dalle import load_clip_model

import os
import warnings
import pickle

from vimpac_utils import *


class SlowR50(nn.Module):
    def __init__(self, num_classes, pretrained=True, freeze=False, keep_head=False, device=None):
        super(SlowR50, self).__init__()
        self.net = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=pretrained)
        # Freeze baseline
        if freeze:
            for param in self.net.parameters():
                param.requires_grad = False

        # Replace classification head
        if not keep_head:
            cls_head = ResNetBasicHead(pool=nn.AvgPool3d(kernel_size=(8, 7, 7), stride=(1, 1, 1), padding=(0, 0, 0)),
                                       dropout=nn.Dropout(p=0.5, inplace=False),
                                       proj=nn.Linear(in_features=2048, out_features=num_classes, bias=True),
                                       output_pool=nn.AdaptiveAvgPool3d(output_size=1))
            self.net.blocks._modules['5'] = cls_head

    def forward(self, video):
        out = self.net(video)

        return out


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


class VIMPAC(nn.Module):
    def __init__(self, num_classes, pretrained=True, freeze=False, keep_head=False, device="cpu"):
        super(VIMPAC, self).__init__()
        self.visible = True
        self.device = device

        self.VQVAE = load_clip_model(device=self.device)
        self.vimpac = torch.nn.Sequential(OrderedDict([
            ("backbone", TransformerLayout(
                height=128 // 8,
                width=128 // 8,
                length=5,
                vocab_size=8192 + 5,
                hid_dim=512,
                layers=6,
                heads=512 // 64,
                dropout=0,
                use_cls_token=True,
                pre_activation=False,
                pos_emb_args=OPTION2ARGS["hw_separation"],
                layout="T,H|W",
                args=None,
            )),
            ("dropout", nn.Dropout(0.3)),
            ("cls_fc1", nn.Linear(512, 128)),
            ("relu", nn.ReLU()),
            ("cls_fc2", nn.Linear(128, num_classes)),
        ]))

        # Load pretrained weights
        if pretrained:
            self.load_model()

        # Freeze VQ-VAE
        for param in self.VQVAE.parameters():
            param.requires_grad = False

        # Freeze Vimpac backbone
        if freeze:
            for param in self.vimpac[0].parameters():
                param.requires_grad = False

    def load_model(self, model_path='../TimeUnderstanding/VIMPAC_small/last/classifier.pt', strict=False):
        model_dir = os.path.dirname(model_path)
        load_args = pickle.load(open(f"{model_dir}/args.pickle", 'rb'))
        assert load_args.pre_activation == False

        state_dict = torch.load(os.path.join(model_path), map_location=self.device)
        load_keys = set(state_dict.keys())

        # The vocab size might be different, we need to handle this here.
        # We always assume that the vocab_size are started from 0 to VOCAB_SIZE
        #   special tokens (e.g., [CLS], [MASK], [PAD]) are after VOCAB_SIZE.
        if "backbone.embedding.weight" in load_keys:
            load_vocab_size = state_dict["backbone.embedding.weight"].shape[0]
            current_vocab_size = self.vimpac.backbone.embedding.weight.shape[0]
            # current_vocab_size = self.args.vocab_size
            if load_vocab_size != current_vocab_size:
                assert load_vocab_size >= current_vocab_size
                state_dict["backbone.embedding.weight"] = state_dict["backbone.embedding.weight"][:current_vocab_size]
                if self.visible:
                    warnings.warn(f"We shrink the vocab size frm {load_vocab_size} to {current_vocab_size}."
                                  f"We assume that special tokens are after 0  ..  VOCAB_SIZE - 1 ({current_vocab_size - 1}). "
                                  f"E.g., [MASK] = VOCAB_SIZE, [PAD] = VOCAB_SIZE + 1")

        # If we need to change the shape of the positional embedding
        for key in load_keys:
            if key.startswith("backbone.positional_embedding"):
                load_value = state_dict[key]  # (shape), dim
                model_value = getattr(self.vimpac.backbone.positional_embedding,
                                      key[len("backbone.positional_embedding."):])  # (model_shape), dim

                if load_value.shape != model_value.shape:
                    model_shape = model_value.shape[:-1]  # (model_shape), dim --> (model_shape)

                    if self.visible:
                        print(f"Modifying key {key}")
                        print(f"\tshape before interpolation {load_value.shape}")

                    load_value = load_value.permute(-1, *range(len(model_shape))).unsqueeze(
                        0)  # (shape), dim --> dim, (shape) --> 1, dim, (shape)
                    load_value = F.interpolate(load_value, model_shape, mode="linear", align_corners=False)
                    load_value = load_value.squeeze(0).permute(*[i + 1 for i in range(len(model_shape))], 0)

                    if self.visible:
                        print(f"\tshape after interpolation {load_value.shape}")

                    state_dict[key] = load_value

        # if self.visible:
        #     load_keys = set(state_dict.keys())
        #     model_keys = set(self.vimpac.state_dict().keys())
            # if load_keys != model_keys:
            #     # print("Weights in load but not in model")
            #     # for key in sorted(load_keys - model_keys):
            #     #     print(f"\t{key}")
            #
            #     print("Weights in model but not in load")
            #     for key in sorted(model_keys - load_keys):
            #         print(f"\t{key}")

        self.vimpac.load_state_dict(state_dict, strict=strict)

    @torch.no_grad()
    def tokenize(self, video_batch):
        tokens = []
        for video in video_batch:
            token = self.VQVAE(video)
            tokens.append(token.unsqueeze(0))

        return torch.cat(tokens, dim=0)

    def forward(self, video):
        token = self.tokenize(video)
        output = self.vimpac(token)

        return output


__models__ = {'slow': SlowR50,
              'slowfast': SlowFastR50,
              'x3d': X3D,
              'mvit': MViT,
              'vimpac': VIMPAC}


def get_model(model):
    return __models__[model]


if __name__ == '__main__':
    video = [torch.randn(2, 3, 8, 256, 256), torch.randn(2, 3, 32, 256, 256)]
    model = SlowFastR50(num_classes=4, pretrained=True, freeze=True, keep_head=True)
    output = model(video)
    print(output.shape)
