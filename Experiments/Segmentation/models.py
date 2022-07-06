import torch
import torch.nn as nn
from pytorchvideo.models.head import ResNetBasicHead
import pytorchvideo.models.x3d
from pytorchvideo.models.head import VisionTransformerBasicHead, SequencePool
from pytorchvideo.models.hub import mvit_base_16x4
import pickle
import os
import warnings
import torch.nn.functional as F
from load_dalle import load_clip_model
from collections import OrderedDict
from vimpac_utils import TransformerLayout, OPTION2ARGS
from timm.models import create_model
from mae_utils import vit_base_patch16_224, load_from_ckpt


class SlowR50old(nn.Module):
    def __init__(self, device):
        super(SlowR50, self).__init__()
        self.net = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)

        # Replace classification head
        cls_head = ResNetBasicHead(pool=nn.AvgPool3d(kernel_size=(8, 7, 7), stride=(1, 1, 1), padding=(0, 0, 0)),
                                   dropout=nn.Dropout(p=0.5, inplace=False),
                                   proj=nn.Identity(),
                                   output_pool=nn.AdaptiveAvgPool3d(output_size=1))
        self.net.blocks._modules['5'] = cls_head

        # Freeze baseline
        for param in self.net.parameters():
            param.requires_grad = False

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2048, out_channels=512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)
        )

    def forward(self, video):
        embedding = self.net(video).unsqueeze(-1).unsqueeze(-1)
        out = self.decoder(embedding).permute(0, 2, 3, 1)

        return out


class SlowFastR50old(nn.Module):
    def __init__(self, device):
        super(SlowFastR50, self).__init__()
        self.net = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)

        # Replace classification head
        cls_head = ResNetBasicHead(dropout=nn.Dropout(p=0.5, inplace=False),
                                   proj=nn.Identity(),
                                   output_pool=nn.AdaptiveAvgPool3d(output_size=1))
        self.net.blocks._modules['6'] = cls_head

        # Freeze baseline
        for param in self.net.parameters():
            param.requires_grad = False

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2304, out_channels=512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)
        )

    def forward(self, video):
        embedding = self.net(video).unsqueeze(-1).unsqueeze(-1)
        out = self.decoder(embedding).permute(0, 2, 3, 1)

        return out


class X3Dold(nn.Module):
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

        # Freeze baseline
        for param in self.net.parameters():
            param.requires_grad = False

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2048, out_channels=512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)
        )

    def forward(self, video):
        embedding = self.net(video).unsqueeze(-1).unsqueeze(-1)
        out = self.decoder(embedding).permute(0, 2, 3, 1)

        return out


class SlowR50(nn.Module):
    def __init__(self, device):
        super(SlowR50, self).__init__()
        self.net = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)

        # Replace classification head
        self.net.blocks._modules['5'] = nn.Conv3d(2048, 2048, kernel_size=1)

        # Freeze baseline
        for param in self.net.parameters():
            param.requires_grad = False

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2048, out_channels=512, kernel_size=3),
            # nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=1)
        )

    def forward(self, video):
        embedding = torch.mean(self.net(video), dim=2)
        out = self.decoder(embedding).permute(0, 2, 3, 1)

        return out


class SlowFastR50(nn.Module):
    def __init__(self, device):
        super(SlowFastR50, self).__init__()
        self.net = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)

        # Replace classification head
        self.net.blocks._modules['6'] = nn.Identity()

        # Freeze baseline
        for param in self.net.parameters():
            param.requires_grad = False

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2304, out_channels=512, kernel_size=3),
            # nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=1),
        )

    def forward(self, video):
        embedding = torch.mean(self.net(video), dim=2)
        out = self.decoder(embedding).permute(0, 2, 3, 1)

        return out


class X3D(nn.Module):
    def __init__(self, device):
        super(X3D, self).__init__()
        self.net = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)

        # Replace classification head
        cls_head = nn.Identity()
        self.net.blocks._modules['5'] = cls_head

        # Freeze baseline
        for param in self.net.parameters():
            param.requires_grad = False

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=192, out_channels=512, kernel_size=3),
            # nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=1)
        )

    def forward(self, video):
        embedding = torch.mean(self.net(video), dim=2)
        out = self.decoder(embedding).permute(0, 2, 3, 1)

        return out


class MViT(nn.Module):
    def __init__(self, device):
        super(MViT, self).__init__()
        self.net = mvit_base_16x4(pretrained=True)

        # Replace classification head
        cls_head = VisionTransformerBasicHead(sequence_pool=SequencePool("mean"),
                                              dropout=nn.Dropout(p=0.5, inplace=False),
                                              proj=nn.Identity())
        self.net.head = cls_head

        # Freeze baseline
        for param in self.net.parameters():
            param.requires_grad = False

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=768, out_channels=512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)
        )

    def forward(self, video):
        embedding = self.net(video).unsqueeze(-1).unsqueeze(-1)
        out = self.decoder(embedding).permute(0, 2, 3, 1)

        return out


class VIMPAC(nn.Module):
    def __init__(self, device="cpu"):
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
        ]))

        # Load pretrained weights
        self.load_model()

        # Freeze VQ-VAE
        for param in self.VQVAE.parameters():
            param.requires_grad = False
        #
        # # Freeze baseline
        for param in self.vimpac.parameters():
            param.requires_grad = False

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)
        )

    def load_model(self, model_path='../../model_checkpoints/VIMPAC_small/last/classifier.pt', strict=False):
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
        embedding = self.vimpac(token).unsqueeze(-1).unsqueeze(-1)
        output = self.decoder(embedding).permute(0, 2, 3, 1)

        return output


class VideoMAE(nn.Module):
    def __init__(self, device=None):
        super(VideoMAE, self).__init__()
        self.net = create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=400,
            all_frames=16,
            tubelet_size=2,
            drop_rate=0.,
            drop_path_rate=0.1,
            attn_drop_rate=0.,
            drop_block_rate=None,
            use_mean_pooling=True,
            init_scale=0.001,
        )
        # load_from_ckpt(self.net, path='../../model_checkpoints/VideoMAE/checkpoint.pth')
        load_from_ckpt(self.net, path='../../model_checkpoints/VideoMAE/checkpoint2.pth', print_keys=False)
        self.net.head = nn.Identity()

        # Freeze baseline
        for param in self.net.parameters():
            param.requires_grad = False

        # Build Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=768, out_channels=512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)
        )

    def forward(self, video):
        embedding = self.net(video).unsqueeze(-1).unsqueeze(-1)
        out = self.decoder(embedding).permute(0, 2, 3, 1)

        return out


__models__ = {'slow': SlowR50,
              'slowfast': SlowFastR50,
              'x3d': X3D,
              'mvit': MViT,
              'vimpac': VIMPAC,
              'mae': VideoMAE}


def get_model(model):
    return __models__[model]


if __name__ == '__main__':
    frame = torch.ones(2, 3, 16, 256, 256)
    model = X3D(device="cpu")
    output = model(frame)
    print(output.shape)