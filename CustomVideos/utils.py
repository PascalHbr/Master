import torch
import random
import numpy as np
import os

from captum.attr import Saliency, GuidedBackprop, DeepLift, NoiseTunnel
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
from typing import Union

import wandb

from captum.attr import visualization as viz


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


def print_stats(stats):
    for key, value in stats.items():
        print("-" * 50)
        print(f"Video: {key}")
        print(f"Scene label: {value['scene_label']} \t Action label: {value['action_label']}")
        for i in range(5):
            print(f"{value['top_5_classes'][i]:33}: {value['top_5_acc'][i] * 100:5.2f}%")


def video_as_numpy(video):
    video = video.squeeze().cpu().detach().numpy()
    return video


def get_heatmap(model, video, target, method, noise=False):
    model.zero_grad()
    if method == "GuidedBackprop":
        gradients = GuidedBackprop(model)
    elif method == "DeepLift":
        gradients = DeepLift(model)
    elif method == "Saliency":
        gradients = Saliency(model)
    if method == "DeepLift":
        attributions = gradients.attribute(video, target=target, baselines=video * 0)
        heatmap = video_as_numpy(attributions)
    else:
        if noise:
            attributions = gradients.attribute(video, target=target)
            heatmap = video_as_numpy(attributions)
        else:
            nt = NoiseTunnel(gradients)
            attributions = nt.attribute(video, target=target, nt_type='smoothgrad', nt_samples=10)
            heatmap = video_as_numpy(attributions)
    return heatmap


def save_heatmap(model, heatmap, video_path, method, label, category):
    video_name = video_path[0][-6:-4]
    if not os.path.exists(f'/export/home/phuber/archive/Heatmap/custom/{method}/{model}'):
        os.makedirs(f'/export/home/phuber/archive/Heatmap/custom/{method}/{model}')
    np.save(f'/export/home/phuber/archive/Heatmap/custom/{method}/{model}/{video_name}_{label}_{category}.npy', heatmap)


def normalize_scale(attr, scale_factor):
    attr_norm = attr / scale_factor
    return np.clip(attr_norm, -1, 1)


def cumulative_sum_threshold(values, percentile: Union[int, float]):
    # given values should be non-negative
    assert percentile >= 0 and percentile <= 100, (
        "Percentile for thresholding must be " "between 0 and 100 inclusive."
    )
    sorted_vals = np.sort(values.flatten())
    cum_sums = np.cumsum(sorted_vals)
    threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
    return sorted_vals[threshold_id]


def normalize_image_attr(
    attr, sign: str, outlier_perc: Union[int, float] = 2
):
    attr_combined = np.sum(attr, axis=2)
    # Choose appropriate signed values and rescale, removing given outlier percentage.
    attr_combined = (attr_combined > 0) * attr_combined
    threshold = cumulative_sum_threshold(attr_combined, 100 - outlier_perc)

    return normalize_scale(attr_combined, threshold)


def pick_frame(video, n=3, heatmap=False, sigma=1):
    frame = np.transpose(video[:, n], (1, 2, 0))
    if heatmap:
        frame = gaussian_filter(frame, sigma=sigma)
    return frame


def plot_heatmaps(heatmaps, dataloader):
    for i, (video, video_path, scene_label, action_label, video_org) in enumerate(dataloader):
        video_org = video_org[0]
        heatmap = np.load(heatmaps[i])
        heatmap_category = "_".join(heatmaps[i].split("/")[-1].split("_")[2:])[:-4]
        video_name = heatmaps[i].split("/")[-1][:2]

        # heatmap_norm = normalize_image_attr(gaussian_filter(heatmap.transpose(1, 2, 3, 0), sigma=3), "positive", 1)
        for j in range(heatmap.shape[1]):
            plt_fig, plt_axis = plt.subplots(figsize=(6, 6))

            # Remove ticks and tick labels from plot.
            plt_axis.xaxis.set_ticks_position("none")
            plt_axis.yaxis.set_ticks_position("none")
            plt_axis.set_yticklabels([])
            plt_axis.set_xticklabels([])
            plt_axis.grid(b=False)

            vmin, vmax = 0, 1
            alpha_overlay = 0.5
            cmap = cm.jet
            norm_attr = normalize_image_attr(pick_frame(heatmap, j, True, 3), "positive", 1)
            plt_axis.imshow(np.mean(pick_frame(video_org.numpy(), j), axis=2), cmap="gray")
            heat_map = plt_axis.imshow(norm_attr, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha_overlay)
            plt.text(20, 20, heatmap_category, color='red', bbox=dict(facecolor='wheat', boxstyle='round'))

            wandb.log({f"Heatmap_{video_name}": plt})