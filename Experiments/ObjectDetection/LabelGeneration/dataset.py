import torch
import csv
import numpy as np
import cv2
import glob
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
        if self.model in ['slow', 'slowfast', 'x3d']:
            transform = Compose(
                [
                    Lambda(lambda x: x / 255.0),
                    ShortSideScale(size=256),
                    CenterCropVideo(256),
                ]
            )

        elif self.model in ['mvit', 'mae']:
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


class Kinetics400(Dataset):
    def __init__(self, mode, model):
        self.base_path = '/export/data/compvis/kinetics/'
        self.mode = mode if mode == 'train' else 'eval'
        self.model = model

        self.annotations = self.get_annotations()
        self.categories, self.label_dict = self.get_categories()

        # Set transform
        self.transform = self.set_transform()

    def get_annotations(self):
        if self.mode == 'train':
            annotations = pd.read_csv(self.base_path + 'train.csv')
            with open('/export/home/phuber/archive/valid_train_ids.csv', 'r') as csvfile:
                reader = csv.reader(csvfile)
                valid_ids = [int(row[0]) for row in reader if row]
            annotations = annotations[annotations.index.isin(valid_ids)]
        else:
            annotations = pd.read_csv(self.base_path + 'validate.csv')
            with open('/export/home/phuber/archive/valid_test_ids.csv', 'r') as csvfile:
                reader = csv.reader(csvfile)
                valid_ids = [int(row[0]) for row in reader if row]
            annotations = annotations[annotations.index.isin(valid_ids)]

        return annotations.sort_values(by=['label']).reset_index(drop=True)

    def get_categories(self):
        categories = {}
        for index, row in self.annotations.iterrows():
            category = row['label']
            if category not in categories:
                categories[category] = []
            categories[category].append(row['youtube_id'])

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
        frames = []
        for filename in sorted(glob.glob(path + '/*.jpg')):
            frame = cv2.imread(filename)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

        return np.array(frames, dtype=np.uint8)

    def set_transform(self):
        if self.model in ['slow', 'slowfast', 'x3d']:
            transform = Compose(
                [
                    Lambda(lambda x: x / 255.0),
                    ShortSideScale(size=256),
                    CenterCropVideo(256),
                ]
            )

        elif self.model in ['mvit', 'mae']:
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

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):
        # Get video
        video_path = self.base_path + self.mode + '/' + self.annotations['label'][item] + '/' + self.annotations['youtube_id'][item]
        video_path = video_path.replace(' ', '_')
        video = self.load_video(video_path)

        # TODO: use only middle frames
        if self.model in ['slow', 'slowfast', 'mvit', 'mae']:
            n_frames = 64
        elif self.model == 'x3d':
            n_frames = 80
        elif self.model == 'vimpac':
            n_frames = 32
        start_frame = max((video.shape[0] - n_frames) // 2, 0)
        video = video[start_frame:start_frame + n_frames]

        # Make transformation
        video = torch.from_numpy(video).permute(3, 0, 1, 2)
        video = self.transform(video).permute(1, 0, 2, 3)

        return video, video_path


__datasets__ = {'kinetics': Kinetics400,
                'ucf': UCF101}


def get_dataset(dataset_name):
    return __datasets__[dataset_name]