import os
import numpy as np
import torch
import wandb
from urllib import request
import cv2
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import glob
from utils import *

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)

from pytorchvideo.transforms import (
    ShortSideScale,
    UniformTemporalSubsample,
)


class CustomDataset(Dataset):
    def __init__(self, mode, model):
        self.mode = mode
        self.model = model

        self.valid_categories = {107: "basketball", 330: "squats", 37: "brushing_teeth", 142: "golf", 159: "hulahoop",
                                 311: "rope_jump", 183: "lunges", 232: "guitar", 241: "piano", 255: "pullups",
                                 260: "pushups", 365: "shaving_beard", 171: "soccer_juggling", 297: "soccer_penalty",
                                 246: "tennis"}

        self.scene_labels = {"01": 107, "02": 107, "03": 107, "04": 171, "05": 311, "06": 171, "07": 107, "08": 107,
                       "09": 246, "10": 107, "11": 246, "12": 246, "13": 330, "14": 246, "15": 107, "16": 171,
                       "17": 171, "18": 297, "19": 171, "20": 297, "21": 297, "22": 297, "23": 297, "24": 297,
                       "25": 297, "26": 297}

        self.action_labels = {"01": 107, "02": 107, "03": 171, "04": 171, "05": 311, "06": 171, "07": 311, "08": 171,
                       "09": 246, "10": 246, "11": 246, "12": 246, "13": 330, "14": 330, "15": 330, "16": 171,
                       "17": 171, "18": 297, "19": 260, "20": 255, "21": 246, "22": 246, "23": 297, "24": 297,
                       "25": 311, "26": 311}

        self.videos = sorted(glob.glob("videos/npy/*.npy"))
        self.kinetics_labels = self.read_kinetics_labels()
        self.ucf_labels = self.read_ucf_labels()

        # Set transformation
        self.transform = self.set_transform()

        # Specify number of classes
        self.num_classes = 101

    def read_kinetics_labels(self):
        KINETICS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
        with request.urlopen(KINETICS_URL) as obj:
            labels = [line.decode("utf-8").strip() for line in obj.readlines()]
        return labels

    def read_ucf_labels(self):
        labels = pd.read_csv("/export/home/phuber/Master/I3D_augmentations/labels.csv")["Corresponding Kinetics Labels"].to_list()
        return labels

    def load_video(self, path):
        video = np.load(path)
        return np.array(video, dtype=np.uint8)

    def set_transform(self, normalize=True):
        if self.model == 'slow':
            side_size = 256
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
            crop_size = 256
            num_frames = 8

            if normalize:
                transform = Compose(
                    [
                        UniformTemporalSubsample(num_frames),
                        Lambda(lambda x: x / 255.0),
                        NormalizeVideo(mean, std),
                        ShortSideScale(size=side_size),
                        CenterCropVideo(crop_size=(crop_size, crop_size))
                    ]
                )
            else:
                transform = Compose(
                    [
                        UniformTemporalSubsample(num_frames),
                        Lambda(lambda x: x / 255.0),
                        ShortSideScale(size=side_size),
                        CenterCropVideo(crop_size=(crop_size, crop_size))
                    ]
                )

        elif self.model == 'slowfast':
            side_size = 256
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
            crop_size = 256
            num_frames = 32
            alpha = 4

            if normalize:
                transform = Compose(
                    [
                        UniformTemporalSubsample(num_frames),
                        Lambda(lambda x: x / 255.0),
                        NormalizeVideo(mean, std),
                        ShortSideScale(size=side_size),
                        CenterCropVideo(crop_size),
                        PackPathway(alpha)
                    ]
                )
            else:
                transform = Compose(
                    [
                        UniformTemporalSubsample(num_frames),
                        Lambda(lambda x: x / 255.0),
                        ShortSideScale(size=side_size),
                        CenterCropVideo(crop_size),
                        PackPathway(alpha)
                    ]
                )

        elif self.model == 'x3d':
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
            model_transform_params = {
                "x3d_xs": {
                    "side_size": 182,
                    "crop_size": 182,
                    "num_frames": 4,
                    "sampling_rate": 12,
                },
                "x3d_s": {
                    "side_size": 182,
                    "crop_size": 182,
                    "num_frames": 13,
                    "sampling_rate": 6,
                },
                "x3d_m": {
                    "side_size": 256,
                    "crop_size": 256,
                    "num_frames": 16,
                    "sampling_rate": 5,
                }
            }

            model_name = 'x3d_m'
            transform_params = model_transform_params[model_name]
            if normalize:
                transform = Compose(
                    [
                        UniformTemporalSubsample(transform_params["num_frames"]),
                        Lambda(lambda x: x / 255.0),
                        NormalizeVideo(mean, std),
                        ShortSideScale(size=transform_params["side_size"]),
                        CenterCropVideo(crop_size=(transform_params["crop_size"], transform_params["crop_size"]))
                    ]
                )
            else:
                transform = Compose(
                    [
                        UniformTemporalSubsample(transform_params["num_frames"]),
                        Lambda(lambda x: x / 255.0),
                        ShortSideScale(size=transform_params["side_size"]),
                        CenterCropVideo(crop_size=(transform_params["crop_size"], transform_params["crop_size"]))
                    ]
                )

        elif self.model == 'mvit':
            side_size = 256
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
            crop_size = 224
            num_frames = 16

            if normalize:
                transform = Compose(
                    [
                        UniformTemporalSubsample(num_frames),
                        Lambda(lambda x: x / 255.0),
                        NormalizeVideo(mean, std),
                        ShortSideScale(size=side_size),
                        CenterCropVideo(crop_size),
                    ]
                )
            else:
                transform = Compose(
                    [
                        UniformTemporalSubsample(num_frames),
                        Lambda(lambda x: x / 255.0),
                        ShortSideScale(size=side_size),
                        CenterCropVideo(crop_size),
                    ]
                )

        elif self.model == 'vimpac':
            side_size = 128
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            crop_size = 128
            num_frames = 5

            if normalize:
                transform = Compose(
                    [
                        UniformTemporalSubsample(num_frames),
                        Lambda(lambda x: x / 255.0),
                        NormalizeVideo(mean, std),
                        ShortSideScale(size=side_size),
                        CenterCropVideo(crop_size),
                    ]
                )
            else:
                transform = Compose(
                    [
                        UniformTemporalSubsample(num_frames),
                        Lambda(lambda x: x / 255.0),
                        ShortSideScale(size=side_size),
                        CenterCropVideo(crop_size),
                    ]
                )

        elif self.model == 'mae':
            side_size = 256
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            crop_size = 224
            num_frames = 16

            if normalize:
                transform = Compose(
                    [
                        UniformTemporalSubsample(num_frames),
                        Lambda(lambda x: x / 255.0),
                        NormalizeVideo(mean, std),
                        ShortSideScale(size=side_size),
                        CenterCropVideo(crop_size),
                    ]
                )
            else:
                transform = Compose(
                    [
                        UniformTemporalSubsample(num_frames),
                        Lambda(lambda x: x / 255.0),
                        ShortSideScale(size=side_size),
                        CenterCropVideo(crop_size),
                    ]
                )

        return transform

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, item):
        # Get video
        video_path = self.videos[item]
        video = self.load_video(video_path)

        # Take model-specific number of frames
        if self.model in ['slow', 'slowfast', 'x3d', 'mvit', 'mae']:
            n_frames = 64
        elif self.model == 'vimpac':
            n_frames = 32
        start_frame = max((video.shape[1] - n_frames) // 2, 0)
        video = video[start_frame:start_frame + n_frames]

        # Make transformation
        video = torch.from_numpy(video)
        video_org = self.set_transform(normalize=False)(video).numpy()
        video = self.transform(video)
        # Make permutation
        if self.model == 'vimpac':
            video = video.permute(1, 0, 2, 3)
        # Set label
        scene_label = torch.tensor(self.scene_labels[video_path[-6:-4]])
        action_label = torch.tensor(self.action_labels[video_path[-6:-4]])

        return video, video_path, scene_label, action_label, video_org


__datasets__ = {'custom': CustomDataset}


def get_dataset(dataset_name):
    return __datasets__[dataset_name]


if __name__ == '__main__':
    model = "slow"
    dataset = CustomDataset(mode='test', model=model)
    video, video_path, scene_label, action_label, video_org = dataset[2]
    print(video.shape)
    print(video_path)
    print(scene_label)
    print(action_label)
    print(video_org.shape)
    # wandb.init(project='Sample Videos')
    # wandb.log({"video": wandb.Video(255*video.permute(1, 0, 2, 3).numpy(), fps=4, format="mp4")})

    # heatmap = np.load(f"/export/home/phuber/archive/Heatmap/ucf/InputXGradient/{model}/tennis/v_TennisSwing_g03_c06_1.npy")
    # wandb.init(project='Heatmap')
    # for frame in range(heatmap.shape[1]):
    #     plot_video_on_wandb(video_org, heatmap, frame)

    # wandb.log({"video": wandb.Video(255*np.transpose(heatmap_overlap, (1, 0, 2, 3)), fps=1, format="mp4")})