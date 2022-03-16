import random

import torch

import wandb
from urllib import request

import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import glob
import csv
import itertools
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
    def __init__(self, mode, task, model, n_permute=4, n_blackout=4):
        self.dataset_path = "/export/scratch/compvis/datasets/UCF101/videos"
        self.mode = mode
        self.task = task
        self.model = model

        # if task == 'classification':
        #     self.kinetics_labels = self.read_kinetics_labels()
        #     self.ucf_labels = self.read_ucf_labels('/export/home/phuber/Master/I3D_augmentations/labels.csv')
        #     self.categories, self.label_dict = self.get_classification_categories()
        #     self.videos = np.array([item for sublist in self.categories.values() for item in sublist])
        # else:
        self.videos = self.get_videos()
        self.categories, self.label_dict = self.get_categories()

        self.transform = self.set_transform()

        # Task specific parameters
        self.n_permute = n_permute
        self.n_blackout = n_blackout
        self.permutation_array = list(itertools.permutations(range(0, self.n_permute)))

        # Specify number of classes
        if task == 'classification':
            self.num_classes = 101
        elif task == 'permutation':
            self.num_classes = len(self.permutation_array)
        elif task == 'blackout' or 'freeze':
            self.num_classes = n_blackout
        elif task == 'reverse':
            self.num_classes = 2

    def read_kinetics_labels(self):
        KINETICS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
        with request.urlopen(KINETICS_URL) as obj:
            labels = [line.decode("utf-8").strip() for line in obj.readlines()]
        return labels

    def read_ucf_labels(self, path_labels):
        labels = pd.read_csv(path_labels)["Corresponding Kinetics Labels"].to_list()
        #labels = [list(map(int, label)) for label in labels]
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

    def get_classification_categories(self):
        videos = glob.glob(self.dataset_path + "/**/*.avi", recursive=True)
        video_list = list(sorted(set(videos)))
        categories = {}
        for video in video_list:
            category = video.split('/')[-1][2:-12]
            if category not in categories:
                categories[category] = []
            categories[category].append(video)

        # Choose selected categories from labels.txt
        valid_categories = []
        for i, (category, sequences) in enumerate(categories.items()):
            kinetics_id = self.ucf_labels[i]
            if kinetics_id >= 0:
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

    def set_transform(self):
        if self.model == 'slow':
            side_size = 256
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
            crop_size = 256
            num_frames = 8

            transform = Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean, std),
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
            transform = Compose(
                [
                    UniformTemporalSubsample(transform_params["num_frames"]),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(size=transform_params["side_size"]),
                    CenterCropVideo(crop_size=(transform_params["crop_size"], transform_params["crop_size"]))
                ]
            )

        elif self.model == 'mvit':
            side_size = 256  # TODO: See if it should be 224
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
            crop_size = 224
            num_frames = 16

            transform = Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean, std),
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

            transform = Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean, std),
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
        if self.model in ['slow', 'slowfast', 'mvit']:
            n_frames = 64
        elif self.model == 'x3d':
            n_frames = 80
        elif self.model == 'vimpac':
            n_frames = 32
        if self.mode == 'train':
            start_frame = random.randint(0, max(video.shape[0]-n_frames, 0))
        else:
            start_frame = max((video.shape[0] - n_frames) // 2, 0)
        video = video[start_frame:start_frame + n_frames]

        # Get label
        if self.task == 'classification':
            category = video_path.split('/')[-1][2:-12]
            label = self.label_dict[category]
        if self.task == 'reverse':
            label = random.choice([0, 1])
            video = np.flip(video, 0).copy() if label == 1 else video
        elif self.task == 'permutation':
            label = random.randint(0, len(self.permutation_array)-1)
            permutation = np.array(self.permutation_array[label])
            chunks = np.array(np.array_split(video, self.n_permute, axis=0), dtype=object)
            video_permutation = chunks[permutation]
            video = np.vstack(video_permutation).astype(np.float32)
        elif self.task == 'blackout':
            label = random.randint(0, self.n_blackout-1)
            chunks = np.array(np.array_split(video, self.n_blackout, axis=0), dtype=object)
            chunks[label] = np.zeros_like(chunks[label])
            video = np.vstack(chunks).astype(np.float32)
            if video.ndim != 4:
                print("Error at item: ", item)
        elif self.task == 'freeze':
            label = random.randint(0, self.n_blackout-1)
            chunks = np.array(np.array_split(video, self.n_blackout, axis=0), dtype=object)
            frames, _, _, _ = chunks[label].shape
            frames = 1 if frames == 0 else frames
            try:
                chunks[label] = np.tile(np.expand_dims(chunks[label][0], 0), (frames, 1, 1, 1))
            except:
                print("Item: ", item)
                print("Video: ", video.shape)
                print(chunks[0].shape)
                print(chunks[1].shape)
                print(chunks[2].shape)
                print(chunks[3].shape)
            video = np.vstack(chunks).astype(np.float32)
            if video.ndim != 4:
                print("Error at item: ", item)

        # Make transformation
        video = torch.from_numpy(video).permute(3, 0, 1, 2)
        video = self.transform(video)
        if self.model == 'vimpac':
            video = video.permute(1, 0, 2, 3)
        label = torch.tensor(label)

        return video, label


class Kinetics400(Dataset):
    def __init__(self, mode, task, model, n_permute=4,  n_blackout=4):
        self.base_path = '/export/data/compvis/kinetics/'
        self.mode = mode if mode == 'train' else 'eval'
        self.task = task
        self.model = model

        self.annotations = self.get_annotations()
        self.categories, self.label_dict = self.get_categories()
        self.transform = self.set_transform()

        # Task specific parameters
        self.n_permute = n_permute
        self.n_blackout = n_blackout
        self.permutation_array = list(itertools.permutations(range(0, self.n_permute)))

        # Specify number of classes
        if task == 'classification':
            self.num_classes = 400
        elif task == 'permutation':
            self.num_classes = len(self.permutation_array)
        elif task == 'blackout' or 'freeze':
            self.num_classes = n_blackout
        elif task == 'reverse':
            self.num_classes = 2

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
            frame = frame[:, :, [2, 1, 0]]  # BGR -> RGB TODO: see if it matters
            frames.append(frame)

        return np.array(frames, dtype=np.uint8)

    def set_transform(self):
        if self.model == 'slow':
            side_size = 256
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
            crop_size = 256
            num_frames = 8

            transform = Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean, std),
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
            transform = Compose(
                [
                    UniformTemporalSubsample(transform_params["num_frames"]),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(size=transform_params["side_size"]),
                    CenterCropVideo(crop_size=(transform_params["crop_size"], transform_params["crop_size"]))
                ]
            )

        elif self.model == 'mvit':
            side_size = 256  # TODO: See if it should be 224
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
            crop_size = 224
            num_frames = 16

            transform = Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(size=side_size),
                    CenterCropVideo(crop_size),
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

        # Take model-specific number of frames
        if self.model in ['slow', 'slowfast', 'mvit']:
            n_frames = 64
        elif self.model == 'x3d':
            n_frames = 80
        start_frame = max((video.shape[0] - n_frames) // 2, 0)
        video = video[start_frame:start_frame+n_frames]
        # start_frames = np.random.randint(0, max(video.shape[0] - n_frames, 1), size=10)
        # videos = torch.cat([self.transform(torch.from_numpy(video[start_frame:start_frame+64]).permute(3, 0, 1, 2)).unsqueeze(0) for start_frame in start_frames], dim=0)
        # freeze_frames = np.random.randint(0, video.shape[0], size=10)
        # videos = torch.cat([self.transform(torch.from_numpy(np.tile(np.expand_dims(video[freeze_frame], 0), (64, 1, 1, 1))).permute(3, 0, 1, 2)).unsqueeze(0) for freeze_frame in freeze_frames], dim=0)

        # Get label
        if self.task == 'classification':
            category = self.annotations['label'][item]
            label = self.label_dict[category]
        if self.task == 'reverse':
            label = random.choice([0, 1])
            video = np.flip(video, 0).copy() if label == 1 else video
        elif self.task == 'permutation':
            label = random.randint(0, len(self.permutation_array)-1)
            permutation = np.array(self.permutation_array[label])
            chunks = np.array(np.array_split(video, self.n_permute, axis=0), dtype=object)
            video_permutation = chunks[permutation]
            video = np.vstack(video_permutation).astype(np.float32)
        elif self.task == 'blackout':
            label = random.randint(0, self.n_blackout-1)
            chunks = np.array(np.array_split(video, self.n_blackout, axis=0), dtype=object)
            chunks[label] = np.zeros_like(chunks[label])
            video = np.vstack(chunks).astype(np.float32)
            if video.ndim != 4:
                print("Error at item: ", item)
        elif self.task == 'freeze':
            label = random.randint(0, self.n_blackout-1)
            chunks = np.array(np.array_split(video, self.n_blackout, axis=0), dtype=object)
            frames, _, _, _ = chunks[label].shape
            chunks[label] = np.tile(np.expand_dims(chunks[label][0], 0), (frames, 1, 1, 1))
            video = np.vstack(chunks).astype(np.float32)
            if video.ndim != 4:
                print("Error at item: ", item)

        # Make transformation
        video = torch.from_numpy(video).permute(3, 0, 1, 2)
        video = self.transform(video)
        label = torch.tensor(label)

        return video, label


__datasets__ = {'kinetics': Kinetics400,
                'ucf': UCF101}


def get_dataset(dataset_name):
    return __datasets__[dataset_name]


if __name__ == '__main__':
    dataset = UCF101(mode='test', task='blackout', model='vimpac', n_blackout=4)
    video, label = dataset[2013]
    print(video.shape)
    print(label)
    for frame in video:
        print(frame[0])
    # wandb.init(project='Sample Videos')
    # wandb.log({"video": wandb.Video(video.permute(1, 0, 2, 3).numpy(), fps=4, format="mp4")})