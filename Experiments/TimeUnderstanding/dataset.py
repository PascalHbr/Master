import wandb
from urllib import request
import json
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import glob
import csv
import itertools
from utils import *
from torch.utils.data.sampler import Sampler
import time

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

        self.videos = self.get_videos()
        self.categories, self.label_dict = self.get_categories()
        self.idx_to_category = {i: category for i, category in enumerate(self.categories)}
        self.category_to_idx = {v.lower(): k for k, v in self.idx_to_category.items()}

        # Set transform
        self.transform = self.set_transform()

        # Task specific parameters
        self.n_permute = n_permute
        self.n_blackout = n_blackout
        self.permutation_array = list(itertools.permutations(range(0, self.n_permute)))

        # Specify number of classes
        if task == 'permutation':
            self.num_classes = len(self.permutation_array)
        elif task in ['blackout', 'whiteout']:
            self.num_classes = n_blackout
        elif task in ['reverse', 'permutation_cfk', 'freeze_cfk']:
            self.num_classes = 2

    def read_kinetics_labels(self):
        KINETICS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
        with request.urlopen(KINETICS_URL) as obj:
            labels = [line.decode("utf-8").strip() for line in obj.readlines()]
        return labels

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

        return list(sorted(set(videos)))

    def get_categories(self):
        videos = glob.glob(self.dataset_path + "/**/*.avi", recursive=True)
        video_list = list(sorted(set(videos)))
        categories = {}
        for video in video_list:
            category = video.split('/')[-1][2:-12]
            if category not in categories:
                categories[category] = []
            if video in self.videos:
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
        video_org = video[start_frame:start_frame + n_frames]

        # Get label
        category = video_path.split('/')[-1][2:-12]
        category_idx = self.category_to_idx[category.lower()]
        if self.task == 'reverse':
            label = random.choice([0, 1])
            video = np.flip(video_org, 0).copy() if label == 1 else video_org
        elif self.task == 'permutation':
            label = random.randint(0, len(self.permutation_array)-1)
            permutation = np.array(self.permutation_array[label])
            chunks = np.array(np.array_split(video_org, self.n_permute, axis=0), dtype=object)
            video_permutation = chunks[permutation]
            video = np.vstack(video_permutation).astype(np.float32)
        elif self.task == 'permutation_cfk':
            label = random.choice([0, 1])
            permutation = np.random.permutation(video_org.shape[0])
            video = video_org[permutation] if label == 1 else video_org
        elif self.task == 'freeze_cfk':
            label = random.choice([0, 1])
            freeze_idx = random.randint(0, video_org.shape[0] - 1)
            video = np.tile(np.expand_dims(video_org[freeze_idx], 0), (n_frames, 1, 1, 1)) if label == 1 else video_org
        elif self.task == 'blackout':
            label = random.randint(0, self.n_blackout-1)
            chunks = np.array(np.array_split(video_org, self.n_blackout, axis=0), dtype=object)
            chunks[label] = np.zeros_like(chunks[label])
            video = np.vstack(chunks).astype(np.float32)
            if video_org.ndim != 4:
                print("Error at item: ", item)
        elif self.task == 'whiteout':
            label = random.randint(0, self.n_blackout-1)
            chunks = np.array(np.array_split(video_org, self.n_blackout, axis=0), dtype=object)
            chunks[label] = 255 * np.ones_like(chunks[label])
            video = np.vstack(chunks).astype(np.float32)
            if video_org.ndim != 4:
                print("Error at item: ", item)

        # Make transformation
        video = torch.from_numpy(video).permute(3, 0, 1, 2)
        video = self.transform(video)
        if self.model == 'vimpac':
            video = video.permute(1, 0, 2, 3)
        label = torch.tensor(label)

        return video, category_idx, label


class Kinetics400(Dataset):
    def __init__(self, mode, task, model, n_permute=4,  n_blackout=4):
        self.base_path = '/export/data/compvis/kinetics/'
        self.mode = mode if mode == 'train' else 'eval'
        self.task = task
        self.model = model

        self.annotations = self.get_annotations()
        self.categories, self.label_dict = self.get_categories()
        self.idx_to_category = {i: category for i, category in enumerate(self.categories)}
        self.category_to_idx = {v.lower(): k for k, v in self.idx_to_category.items()}

        # Set transform
        self.transform = self.set_transform()

        # Task specific parameters
        self.n_permute = n_permute
        self.n_blackout = n_blackout
        self.permutation_array = list(itertools.permutations(range(0, self.n_permute)))

        # Specify number of classes
        if task == 'permutation':
            self.num_classes = len(self.permutation_array)
        elif task in ['blackout', 'whiteout']:
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
            frame = frame[:, :, [2, 1, 0]]
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
        category_idx = self.category_to_idx[category.lower()]
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
        if self.model == 'vimpac':
            video = video.permute(1, 0, 2, 3)
        label = torch.tensor(label)

        return video, category_idx, label


class SSV2(Dataset):
    def __init__(self, mode, task, model, n_permute=4, n_blackout=4):
        self.mode = mode if mode == 'train' else 'validation'
        self.task = task
        self.model = model

        self.annotations = self.get_annotations()
        self.videos, self.labels = self.get_videos()
        self.categories, self.label_dict = self.get_categories()

        self.idx_to_category = {i: category for i, category in enumerate(self.categories.keys())}
        self.category_to_idx = {v.lower(): k for k, v in self.idx_to_category.items()}

        # Set transform
        self.transform = self.set_transform()

        # Task specific parameters
        self.n_permute = n_permute
        self.n_blackout = n_blackout
        self.permutation_array = list(itertools.permutations(range(0, self.n_permute)))

        # Specify number of classes
        if task == 'permutation':
            self.num_classes = len(self.permutation_array)
        elif task in ['blackout', 'whiteout']:
            self.num_classes = n_blackout
        elif task in ['reverse', 'permutation_cfk', 'freeze_cfk']:
            self.num_classes = 2

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
        video_org = video[start_frame:start_frame + n_frames]

        # Get label
        category = self.labels[item]
        category_idx = self.category_to_idx[category.lower()]
        if self.task == 'reverse':
            label = random.choice([0, 1])
            video = np.flip(video_org, 0).copy() if label == 1 else video_org
        elif self.task == 'permutation':
            label = random.randint(0, len(self.permutation_array)-1)
            permutation = np.array(self.permutation_array[label])
            chunks = np.array(np.array_split(video_org, self.n_permute, axis=0), dtype=object)
            video_permutation = chunks[permutation]
            video = np.vstack(video_permutation).astype(np.float32)
        elif self.task == 'permutation_cfk':
            label = random.choice([0, 1])
            permutation = np.random.permutation(video_org.shape[0])
            video = video_org[permutation] if label == 1 else video_org
        elif self.task == 'freeze_cfk':
            label = random.choice([0, 1])
            freeze_idx = random.randint(0, video_org.shape[0] - 1)
            video = np.tile(np.expand_dims(video_org[freeze_idx], 0), (n_frames, 1, 1, 1)) if label == 1 else video_org
        elif self.task == 'blackout':
            label = random.randint(0, self.n_blackout-1)
            chunks = np.array(np.array_split(video_org, self.n_blackout, axis=0), dtype=object)
            chunks[label] = np.zeros_like(chunks[label])
            video = np.vstack(chunks).astype(np.float32)
            if video_org.ndim != 4:
                print("Error at item: ", item)
        elif self.task == 'whiteout':
            label = random.randint(0, self.n_blackout-1)
            chunks = np.array(np.array_split(video_org, self.n_blackout, axis=0), dtype=object)
            chunks[label] = 255 * np.ones_like(chunks[label])
            video = np.vstack(chunks).astype(np.float32)
            if video_org.ndim != 4:
                print("Error at item: ", item)

        # Make transformation
        video = torch.from_numpy(video).permute(3, 0, 1, 2)
        video = self.transform(video)
        if self.model == 'vimpac':
            video = video.permute(1, 0, 2, 3)
        label = torch.tensor(label)

        return video, category_idx, label


class RandomSampler(Sampler):
    def __init__(self, data_source, num_samples=None):
        super().__init__(data_source)
        self.data_source = data_source
        self._num_samples = num_samples

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        return iter(torch.randperm(n, dtype=torch.int64)[: self.num_samples].tolist())

    def __len__(self):
        return self.num_samples


__datasets__ = {'kinetics': Kinetics400,
                'ucf': UCF101,
                'ssv2': SSV2}


def get_dataset(dataset_name):
    return __datasets__[dataset_name]


if __name__ == '__main__':
    dataset = SSV2(mode='test', task='permutation_cfk', model='mae', n_blackout=5)
    video, category_idx, label = dataset[3420]
    print(video.shape)
    print(category_idx)
    print(label)
    # wandb.init(project='Sample Videos')
    # for i, frame in enumerate(video):
    #     wandb.log({f"frame_{i}": wandb.Image(255*frame.permute(1, 2, 0).numpy())})
    # wandb.log({"video": wandb.Video(255*video.numpy(), fps=4, format="mp4")})