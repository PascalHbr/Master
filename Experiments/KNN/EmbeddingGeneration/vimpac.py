import pickle
from .load_dalle import *
from .vimpac_utils import *


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

    def load_model(self, model_path='../../../../model_checkpoints/VIMPAC_small/last/classifier.pt', strict=False):
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
        output = self.vimpac(token)

        return output