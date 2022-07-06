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


class UCF101(Dataset):
    def __init__(self, mode, model, categories_to_plot=None):
        self.dataset_path = "/export/scratch/compvis/datasets/UCF101/videos"
        self.mode = mode
        self.model = model

        self.valid_categories = {107: "basketball", 330: "squats", 37: "brushing_teeth", 142: "golf", 159: "hulahoop",
                                 311: "rope_jump", 183: "lunges", 232: "guitar", 241: "piano", 255: "pullups",
                                 260: "pushups", 365: "shaving_beard", 171: "soccer_juggling", 297: "soccer_penalty",
                                 246: "tennis"}
        if categories_to_plot is not None:
            self.valid_categories = {k: v for (k, v) in self.valid_categories.items() if v in categories_to_plot}
        # self.valid_categories = {246: "tennis"}

        self.kinetics_labels = self.read_kinetics_labels()
        self.ucf_labels = self.read_ucf_labels('/export/home/phuber/Master/I3D_augmentations/labels.csv')
        self.all_videos = self.get_all_videos()
        self.categories, self.label_dict = self.get_classification_categories()
        self.videos = np.array([item for sublist in self.categories.values() for item in sublist])
        self.idx_to_category = {i: category for i, category in enumerate(
            list(sorted(set(os.listdir("/export/scratch/compvis/datasets/UCF101/videos/")))))}
        self.category_to_idx = {v.lower(): k for k, v in self.idx_to_category.items()}

        # Set transformation
        self.transform = self.set_transform()

        # Specify number of classes
        self.num_classes = 101

    def read_kinetics_labels(self):
        KINETICS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
        with request.urlopen(KINETICS_URL) as obj:
            labels = [line.decode("utf-8").strip() for line in obj.readlines()]
        return labels

    def read_ucf_labels(self, path_labels):
        labels = pd.read_csv(path_labels)["Corresponding Kinetics Labels"].to_list()
        return labels

    def get_all_videos(self):
        if self.mode == 'train':
            with open('/export/home/phuber/Master/ImageClassification/splits/UCF101/trainlist.txt', 'r') as f:
                videos = list(map(lambda x: x.split(' ')[0], f.readlines()))
        else:
            with open('/export/home/phuber/Master/ImageClassification/splits/UCF101/testlist.txt', 'r') as f:
                videos = list(map(lambda x: x[:-1], f.readlines()))

        return list(sorted(set(videos)))

    def get_classification_categories(self):
        videos = glob.glob(self.dataset_path + "/**/*.avi", recursive=True)
        video_list = list(sorted(set(videos)))
        categories = {}
        for video in video_list:
            category = video.split('/')[-1][2:-12]
            if category not in categories:
                categories[category] = []
            if video in self.all_videos:
                categories[category].append(video)

        # Choose selected categories from labels.txt
        valid_categories = []
        for i, (category, sequences) in enumerate(categories.items()):
            kinetics_id = self.ucf_labels[i]
            if kinetics_id >= 0 and kinetics_id in self.valid_categories:
                valid_categories.append(category)
        selected_categories = {valid_category: categories[valid_category] for valid_category in valid_categories}

        # Make dictionary that gives kinetics label for category
        label_dict = {}
        for i, category in enumerate(categories):
            if category in selected_categories:
                label_dict[category] = self.ucf_labels[i]

        print("Found %d videos in %d categories." % (sum(list(map(len, selected_categories.values()))),
                                                     len(selected_categories)))
        return selected_categories, label_dict

    def print_summary(self):
        for category, sequences in self.categories.items():
            summary = ", ".join(sequences[:1])
            print("%-20s %4d videos (%s, ...)" % (category, len(sequences), summary))

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)
        finally:
            cap.release()

        return np.array(frames, dtype=np.uint8)

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
        # video_path = "/export/scratch/compvis/datasets/UCF101/videos/TennisSwing/v_TennisSwing_g03_c06.avi"
        video = self.load_video(video_path)

        # Take model-specific number of frames
        if self.model in ['slow', 'slowfast', 'mvit', 'mae']:
            n_frames = 64
        elif self.model == 'x3d':
            n_frames = 80
        elif self.model == 'vimpac':
            n_frames = 32
        start_frame = max((video.shape[0] - n_frames) // 2, 0)
        video = video[start_frame:start_frame + n_frames]

        # Get label
        category = video_path.split('/')[-1][2:-12]
        label = self.label_dict[category]

        # Make transformation
        video = torch.from_numpy(video).permute(3, 0, 1, 2)
        video_org = self.set_transform(normalize=False)(video).numpy()
        video = self.transform(video)
        label = torch.tensor(label)
        # Make permutation
        if self.model == 'vimpac':
            video = video.permute(1, 0, 2, 3)

        return video, video_path, label, video_org


__datasets__ = {'ucf': UCF101}


def get_dataset(dataset_name):
    return __datasets__[dataset_name]


if __name__ == '__main__':
    model = "mvit"
    dataset = UCF101(mode='test', model=model)
    print(dataset.categories.keys())
    # video, video_path, label, video_org = dataset[9]
    # print(dataset.videos)
    # wandb.init(project='Sample Videos')
    # wandb.log({"video": wandb.Video(255*video_org.transpose((1, 0, 2, 3)), fps=4, format="mp4")})

    # heatmap = np.load(f"/export/home/phuber/archive/Heatmap/custom/GuidedBackprop/{model}/custom/10.npy")
    # wandb.init(project='Heatmap')
    # for frame in range(heatmap.shape[1]):
    #     plot_video_on_wandb(video_org, heatmap, frame)

    # wandb.log({"video": wandb.Video(255*np.transpose(heatmap_overlap, (1, 0, 2, 3)), fps=1, format="mp4")})