import torch
import torch.nn.functional as nnf


class Shuffler(object):
    def patchify(self, x, patch_size):
        x = torch.from_numpy(x).permute(0, 3, 1, 2)
        # divide the batch of images into non-overlapping patches
        u = nnf.unfold(x, kernel_size=patch_size, stride=patch_size, padding=0)
        # permute the patches of each image in the batch
        indices = torch.randperm(u.shape[-1])
        pu = u[:, :, indices]
        # fold the permuted patches back together
        f = nnf.fold(pu, x.shape[-2:], kernel_size=patch_size, stride=patch_size, padding=0).permute(0, 2, 3, 1).numpy()
        return f