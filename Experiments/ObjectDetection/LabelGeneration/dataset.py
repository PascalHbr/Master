import torch

import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.transforms.transforms import Resize
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
)

from pytorchvideo.transforms import (
    ShortSideScale,
)


class UCF101(Dataset):
    def __init__(self, mode, model):
        self.dataset_path = "/export/scratch/compvis/datasets/UCF101/videos"
        self.mode = mode
        self.model = model

        self.videos = self.get_videos()
        self.categories, self.label_dict = self.get_categories()

        self.transform = self.set_transform()

    def set_transform(self):
        if self.model in ['slowfast', 'x3d']:
            transform = Compose(
                [
                    Lambda(lambda x: x / 255.0),
                    ShortSideScale(size=256),
                    CenterCropVideo(256),
                ]
            )

        elif self.model == 'mvit':
            transform = Compose(
                [
                    Lambda(lambda x: x / 255.0),
                    ShortSideScale(size=256),
                    CenterCropVideo(224),
                ]
            )

        elif self.model == 'vimpac':
            transform = Compose(
                [
                    Lambda(lambda x: x / 255.0),
                    ShortSideScale(size=256),
                    Resize(size=160),
                    CenterCropVideo(128),
                ]
            )

        return transform

    def read_ucf_labels(self, path_labels):
        labels = pd.read_csv(path_labels)["Corresponding Kinetics Labels"].to_list()
        return labels

    def get_videos(self):
        if self.mode == 'train':
            with open('/export/home/phuber/Master/ImageClassification/splits/UCF101/trainlist.txt', 'r') as f:
                videos = list(map(lambda x: x.split(' ')[0], f.readlines()))
        else:
            with open('/export/home/phuber/Master/ImageClassification/splits/UCF101/testlist.txt', 'r') as f:
                videos = list(map(lambda x: x[:-1], f.readlines()))

        return videos

    def get_categories(self):
        categories = {}
        for video in self.videos:
            category = video.split('/')[-1][2:-12]
            if category not in categories:
                categories[category] = []
            categories[category].append(video)

        # Make dictionary that gives kinetics label for category
        label_dict = {}
        for i, category in enumerate(categories):
            label_dict[category] = i

        print("Found %d videos in %d categories." % (sum(list(map(len, categories.values()))),
                                                     len(categories)))
        return categories, label_dict

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

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, item):
        # Get video
        video_path = self.videos[item]
        video = self.load_video(video_path)

        # Make transformation
        video = torch.from_numpy(video).permute(3, 0, 1, 2)
        video = self.transform(video).permute(1, 0, 2, 3)

        return video, video_path