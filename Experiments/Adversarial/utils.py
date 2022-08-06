import torch
import random
import numpy as np
import torch.nn.functional as nnf


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


def set_random_seed(id=42):
    # Set random seeds
    random.seed(id)
    torch.manual_seed(id)
    torch.cuda.manual_seed(id)
    np.random.seed(id)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(id)
    rng = np.random.RandomState(id)


def check_correct(pred, label):
    # basketbal
    if pred.item() in [107, 220, 296]:
        if label.item() in [107, 220, 296]:
            return 1
        else:
            return 0

    # haircut
    elif pred.item() in [32, 36, 80, 108, 38]:
        if label.item() in [32, 36, 80, 108, 38]:
            return 1
        else:
            return 0

    # long jump
    elif pred.item() in [182, 367]:
        if label.item() in [182, 367]:
            return 1
        else:
            return 0

    # sweeping floor
    elif pred.item() in [60, 198]:
        if label.item() in [60, 198]:
            return 1
        else:
            return 0

    else:
        return torch.sum(pred == label.data).item()


class Shuffler(object):
    def patchify(self, x, patch_size, crop_size):
        # divide the batch of images into non-overlapping patches
        u = nnf.unfold(x.permute(1, 0, 2, 3), kernel_size=patch_size, stride=patch_size, padding=0)
        # permute the patches of each image in the batch
        indices = torch.randperm(u.shape[-1])
        while torch.equal(indices,torch.arange(u.shape[-1])):
            indices = torch.randperm(u.shape[-1])
        pu = u[:, :, indices]
        # fold the permuted patches back together
        f = nnf.fold(pu, x.shape[-2:], kernel_size=patch_size, stride=patch_size, padding=0).permute(1, 0, 2, 3)
        return f