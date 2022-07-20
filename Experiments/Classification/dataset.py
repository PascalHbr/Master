import os
import numpy as np
import torch
import wandb
from urllib import request
import cv2
from torch.utils.data import Dataset
import pandas as pd
import glob
import csv
from utils import PackPathway
import random
import itertools
import json
import kornia
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
import time

from pytorchvideo.transforms import (
    ShortSideScale,
    UniformTemporalSubsample,
)


class UCF101(Dataset):
    def __init__(self, mode, model, augm=None, n_augm=5, app_augm=None):
        self.dataset_path = "/export/scratch/compvis/datasets/UCF101/videos"
        self.mode = mode
        self.model = model

        self.kinetics_labels = self.read_kinetics_labels()
        self.ucf_labels = self.read_ucf_labels('/export/home/phuber/Master/I3D_augmentations/labels.csv')
        self.all_videos = self.get_all_videos()
        self.categories, self.label_dict = self.get_classification_categories()
        self.videos = np.array([item for sublist in self.categories.values() for item in sublist])
        self.idx_to_category = {i: category for i, category in enumerate(
            list(sorted(set(os.listdir("/export/scratch/compvis/datasets/UCF101/videos/")))))}
        self.category_to_idx = {v.lower(): k for k, v in self.idx_to_category.items()}

        # Set transformation
        self.transform, self.normalize = self.set_transform()

        # Specify number of classes
        self.num_classes = 101

        # Set augmentation
        self.augm = augm
        self.n_augm = n_augm
        self.app_augm = app_augm
        self.app_augm_factor = 0.3
        if self.augm:
            self.permutation_array = list(itertools.permutations(range(0, self.n_augm)))

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
                    ShortSideScale(size=side_size),
                    CenterCropVideo(crop_size=(crop_size, crop_size))
                ]
            )

            normalize = NormalizeVideo(mean, std)

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
                    ShortSideScale(size=side_size),
                    CenterCropVideo(crop_size),
                    PackPathway(alpha)
                ]
            )

            normalize = NormalizeVideo(mean, std)

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
                },
                "x3d_l": {
                    "side_size": 356,
                    "crop_size": 356,
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
                    ShortSideScale(size=transform_params["side_size"]),
                    CenterCropVideo(crop_size=(transform_params["crop_size"], transform_params["crop_size"]))
                ]
            )

            normalize = NormalizeVideo(mean, std)

        elif self.model == 'mvit':
            side_size = 256
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
            crop_size = 224
            num_frames = 16

            transform = Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    ShortSideScale(size=side_size),
                    CenterCropVideo(crop_size),
                ]
            )

            normalize = NormalizeVideo(mean, std)

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
                    ShortSideScale(size=side_size),
                    CenterCropVideo(crop_size),
                ]
            )

            normalize = NormalizeVideo(mean, std)

        elif self.model == 'mae':
            side_size = 256
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            crop_size = 224
            num_frames = 16

            transform = Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    ShortSideScale(size=side_size),
                    CenterCropVideo(crop_size),
                ]
            )

            normalize = NormalizeVideo(mean, std)

        return transform, normalize

    def augment_video(self, video):
        # Only use the middle frame of the video
        if 'brightness' in self.app_augm:
            video = kornia.enhance.adjust_brightness(video.unsqueeze(0).permute(0, 2, 1, 3, 4), self.app_augm_factor)
            video = video.permute(0, 2, 1, 3, 4).squeeze(0)

        if 'saturation' in self.app_augm:
            video = kornia.enhance.adjust_saturation(video.unsqueeze(0).permute(0, 2, 1, 3, 4), 1 - self.app_augm_factor)
            video = video.permute(0, 2, 1, 3, 4).squeeze(0)

        if 'hue' in self.app_augm:  # should be in [-pi, pi]
            video = kornia.enhance.adjust_hue(video.unsqueeze(0).permute(0, 2, 1, 3, 4), self.app_augm_factor)
            video = video.permute(0, 2, 1, 3, 4).squeeze(0)

        if 'posterize' in self.app_augm:
            video = kornia.enhance.posterize(video.unsqueeze(0).permute(0, 2, 1, 3, 4), bits=3)
            video = video.permute(0, 2, 1, 3, 4).squeeze(0)

        if 'clahe' in self.app_augm:
            video = kornia.enhance.equalize_clahe(video.unsqueeze(0).permute(0, 2, 1, 3, 4))
            video = video.permute(0, 2, 1, 3, 4).squeeze(0)

        if 'solarize' in self.app_augm:
            video = kornia.enhance.solarize(video.unsqueeze(0).permute(0, 2, 1, 3, 4))
            video = video.permute(0, 2, 1, 3, 4).squeeze(0)

        return video

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, item):
        # Get video
        video_path = self.videos[item]
        video = self.load_video(video_path)

        # Take model-specific number of frames
        if self.model in ['slow', 'slowfast', 'mvit', 'mae']:
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
        category = video_path.split('/')[-1][2:-12]
        label = self.label_dict[category]
        category_idx = self.category_to_idx[category.lower()]

        # Make augmentation
        if self.augm == 'reverse':
            reverse = random.choice([0, 1])
            video = np.flip(video, 0).copy() if reverse == 1 else video
        elif self.augm == 'blackout':
            blackout_idx = random.randint(0, self.n_augm-1)
            chunks = np.array(np.array_split(video, self.n_augm, axis=0), dtype=object)
            chunks[blackout_idx] = np.zeros_like(chunks[blackout_idx])
            video = np.vstack(chunks).astype(np.float32)
            if video.ndim != 4:
                print("Error at item: ", item)
        elif self.augm == 'whiteout':
            whiteout_idx = random.randint(0, self.n_augm-1)
            chunks = np.array(np.array_split(video, self.n_augm, axis=0), dtype=object)
            chunks[whiteout_idx] = 255 * np.ones_like(chunks[whiteout_idx])
            video = np.vstack(chunks).astype(np.float32)
            if video.ndim != 4:
                print("Error at item: ", item)
        elif self.augm == 'freeze':
            freeze_idx = random.randint(0, self.n_augm-1)
            chunks = np.array(np.array_split(video, self.n_augm, axis=0), dtype=object)
            frames, _, _, _ = chunks[freeze_idx].shape
            frames = 1 if frames == 0 else frames
            chunks[freeze_idx] = np.tile(np.expand_dims(chunks[freeze_idx][0], 0), (frames, 1, 1, 1))
            video = np.vstack(chunks).astype(np.float32)

        label = torch.tensor(label)
        # Make transformation
        video = torch.from_numpy(video).permute(3, 0, 1, 2)
        video = self.transform(video)
        # Make appearance augmentation
        if self.app_augm:
            if self.model == 'slowfast':
                video[0] = self.augment_video(video[0])
                video[1] = self.augment_video(video[1])
            else:
                video = self.augment_video(video)
        # Make permutation
        if self.augm == 'permutation':
            if self.model == 'slowfast':
                indices_slow = torch.randperm(video[0].shape[1])
                indices_fast = torch.randperm(video[1].shape[1])
                video[0] = video[0][:, indices_slow]
                video[1] = video[1][:, indices_fast]
            else:
                indices = torch.randperm(video.shape[1])
                video = video[:, indices]
        if self.model == 'vimpac':
            video = video.permute(1, 0, 2, 3)
        # Normalize
        if self.model == 'slowfast':
            video[0] = self.normalize(video[0])
            video[1] = self.normalize(video[1])
        else:
            video = self.normalize(video)

        return video, category_idx, label


class Kinetics400(Dataset):
    def __init__(self, mode, model, augm=None, n_augm=5, app_augm=None):
        self.base_path = '/export/data/compvis/kinetics/'
        self.mode = mode if mode == 'train' else 'eval'
        self.model = model

        self.annotations = self.get_annotations()
        self.categories, self.label_dict = self.get_categories()
        self.idx_to_category = {i: category for i, category in enumerate(self.categories)}
        self.category_to_idx = {v.lower(): k for k, v in self.idx_to_category.items()}

        self.transform = self.set_transform()

        # Specify number of classes
        self.num_classes = 400

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
            time.sleep(0.001)

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
                },
                "x3d_l": {
                    "side_size": 356,
                    "crop_size": 356,
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
            side_size = 256
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

        elif self.model == 'mae':
            side_size = 256
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
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
        if self.model in ['slow', 'slowfast', 'mvit', 'mae']:
            n_frames = 64
        elif self.model == 'x3d':
            n_frames = 80
        elif self.model == 'vimpac':
            n_frames = 32
        if self.mode == 'train':
            start_frame = random.randint(0, max(video.shape[0] - n_frames, 0))
        else:
            start_frame = max((video.shape[0] - n_frames) // 2, 0)
        video = video[start_frame:start_frame + n_frames]

        # Get label
        category = self.annotations['label'][item]
        label = self.label_dict[category]
        category_idx = self.category_to_idx[category.lower()]

        # Make transformation
        video = torch.from_numpy(video).permute(3, 0, 1, 2)
        video = self.transform(video)
        if self.model == 'vimpac':
            video = video.permute(1, 0, 2, 3)
        label = torch.tensor(label)

        return video, category_idx, label


class SSV2(Dataset):
    def __init__(self, mode, model, augm=None, n_augm=5, app_augm=None):
        self.mode = mode if mode == 'train' else 'validation'
        self.model = model

        self.annotations = self.get_annotations()
        self.videos, self.labels = self.get_videos()
        self.categories, self.label_dict = self.get_categories()

        self.idx_to_category = {i: category for i, category in enumerate(self.categories.keys())}
        self.category_to_idx = {v.lower(): k for k, v in self.idx_to_category.items()}

        # Set transformation
        self.transform = self.set_transform()

        # Specify number of classes
        self.num_classes = 174

        # Set augmentation
        self.augm = augm
        self.n_augm = n_augm
        if self.augm:
            self.permutation_array = list(itertools.permutations(range(0, self.n_augm)))

    def get_annotations(self):
        label_path = '/export/scratch/compvis/datasets/somethingsomethingv2/labels/' + self.mode + '.json'
        with open(label_path, 'r') as json_file:
            annotations = json.load(json_file)

        return annotations

    def get_videos(self):
        video_base_path = '/export/scratch/compvis/datasets/somethingsomethingv2/20bn-something-something-v2/'
        videos = [video_base_path + data['id'] + '.webm' for data in self.annotations]
        labels = [data['template'].replace('[', '').replace(']', '') for data in self.annotations]

        return videos, labels

    def get_categories(self):
        video_base_path = '/export/scratch/compvis/datasets/somethingsomethingv2/20bn-something-something-v2/'
        categories = {}
        for data in self.annotations:
            category = data['template'].replace('[', '').replace(']', '')
            if category not in categories:
                categories[category] = []
            categories[category].append(video_base_path + data['id'] + '.webm')

        # Make dictionary that gives kinetics label for category
        label_path = '/export/scratch/compvis/datasets/somethingsomethingv2/labels/labels.json'
        with open(label_path, 'r') as json_file:
            label_dict = json.load(json_file)

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
                },
                "x3d_l": {
                    "side_size": 312,
                    "crop_size": 312,
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
            side_size = 256
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

        elif self.model in ['mae']:
            side_size = 256
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
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
        return len(self.videos)

    def __getitem__(self, item):
        # Get video
        video_path = self.videos[item]
        video = self.load_video(video_path)

        # Take model-specific number of frames
        if self.model in ['slow', 'slowfast', 'mvit', 'mae']:
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
        category = self.labels[item]
        label = int(self.label_dict[category])
        category_idx = self.category_to_idx[category.lower()]

        # Make augmentation
        if self.augm == 'reverse':
            reverse = random.choice([0, 1])
            video = np.flip(video, 0).copy() if reverse == 1 else video
        elif self.augm == 'blackout':
            blackout_idx = random.randint(0, self.n_augm-1)
            chunks = np.array(np.array_split(video, self.n_augm, axis=0), dtype=object)
            chunks[blackout_idx] = np.zeros_like(chunks[blackout_idx])
            video = np.vstack(chunks).astype(np.float32)
            if video.ndim != 4:
                print("Error at item: ", item)
        elif self.augm == 'whiteout':
            whiteout_idx = random.randint(0, self.n_augm-1)
            chunks = np.array(np.array_split(video, self.n_augm, axis=0), dtype=object)
            chunks[whiteout_idx] = 255 * np.ones_like(chunks[whiteout_idx])
            video = np.vstack(chunks).astype(np.float32)
            if video.ndim != 4:
                print("Error at item: ", item)
        elif self.augm == 'freeze':
            freeze_idx = random.randint(0, self.n_augm-1)
            chunks = np.array(np.array_split(video, self.n_augm, axis=0), dtype=object)
            frames, _, _, _ = chunks[freeze_idx].shape
            frames = 1 if frames == 0 else frames
            chunks[freeze_idx] = np.tile(np.expand_dims(chunks[freeze_idx][0], 0), (frames, 1, 1, 1))
            video = np.vstack(chunks).astype(np.float32)

        # Make transformation
        video = torch.from_numpy(video).permute(3, 0, 1, 2)
        video = self.transform(video)
        label = torch.tensor(label)
        # Make permutation
        if self.augm == 'permutation':
            if self.model == 'slowfast':
                indices_slow = torch.randperm(video[0].shape[1])
                indices_fast = torch.randperm(video[1].shape[1])
                video[0] = video[0][:, indices_slow]
                video[1] = video[1][:, indices_fast]
            else:
                indices = torch.randperm(video.shape[1])
                video = video[:, indices]
        if self.model == 'vimpac':
            video = video.permute(1, 0, 2, 3)

        return video, category_idx, label


__datasets__ = {'kinetics': Kinetics400,
                'ucf': UCF101,
                'ssv2': SSV2}


def get_dataset(dataset_name):
    return __datasets__[dataset_name]


if __name__ == '__main__':
    dataset = UCF101(mode='test', model='slowfast', augm='freeze', n_augm=1)
    video, category_idx, label = dataset[-1]
    print(video[0].shape)
    print(video[1].shape)
    print(category_idx)
    print(label)
    # wandb.init(project='Sample Videos')
    # wandb.log({"video": wandb.Video(255*video.permute(1, 0, 2, 3).numpy(), fps=4, format="mp4")})