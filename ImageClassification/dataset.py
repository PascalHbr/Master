import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import os
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch
import pandas as pd
import glob
import csv
import json


class UCF101(Dataset):
    def __init__(self, mode, augm, avg, get_torch_tensor=True):
        self.num_classes = 101
        self.mode = mode
        self.videos = self.get_videos()
        if mode == 'train' and augm:
            self.transforms = A.Compose(
                [A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.25),
                A.RandomBrightnessContrast(p=0.25),
                A.CLAHE(p=0.25),
                A.Flip(p=0.25),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=0.25),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()])
        else:
            self.transforms = A.Compose(
                [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()])

        self.categories, self.label_dict = self.get_categories()
        self.avg = avg
        self.get_torch_tensor = get_torch_tensor

    def get_videos(self):
        if self.mode == 'train':
            with open('splits/UCF101/trainlist.txt', 'r') as f:
                videos = list(map(lambda x: x.split(' ')[0], f.readlines()))
        else:
            with open('splits/UCF101/testlist.txt', 'r') as f:
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

    def crop_center_square(self, frame):
        y, x = frame.shape[0:2]
        min_dim = min(y, x)
        start_x = (x // 2) - (min_dim // 2)
        start_y = (y // 2) - (min_dim // 2)
        return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

    def load_video(self, path, max_frames=0, resize=(224, 224)):
        cap = cv2.VideoCapture(path)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = self.crop_center_square(frame)
                # TODO: see if crop center is important
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)

                if len(frames) == max_frames:
                    break
        finally:
            cap.release()
        return np.array(frames, dtype=np.uint8)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, item):
        # Get video
        video_path = self.videos[item]
        video = self.load_video(video_path)

        # Select random frame
        n_frames = video.shape[0]
        if self.mode == 'test' and self.avg:
            frames = video[:100]  # max 100 frames
        elif self.mode == 'test':
            frame_idx = n_frames // 2
            frame = video[frame_idx]
        else:
            frame_idx = np.random.randint(0, n_frames)
            frame = video[frame_idx]

        # Get label
        category = video_path.split('/')[-1][2:-12]
        label = self.label_dict[category]

        if self.get_torch_tensor and self.avg:
            frame = torch.cat([self.transforms(image=frame)["image"].unsqueeze(0) for frame in frames], dim=0)
            label = torch.tensor(label).long()
        elif self.get_torch_tensor:
            frame = self.transforms(image=frame)["image"]
            label = torch.tensor(label).long()

        return frame, label


class HMDB51(Dataset):
    def __init__(self, mode, augm, avg, get_torch_tensor=True):
        self.num_classes = 51
        self.mode = mode
        self.videos = self.get_videos()
        if mode == 'train' and augm:
            self.transforms = A.Compose(
                [A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.25),
                 A.RandomBrightnessContrast(p=0.25),
                 A.CLAHE(p=0.25),
                 A.Flip(p=0.25),
                 A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=0.25),
                 A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                 ToTensorV2()])
        else:
            self.transforms = A.Compose(
                [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                 ToTensorV2()])

        self.categories, self.label_dict = self.get_categories()
        self.avg = avg
        self.get_torch_tensor = get_torch_tensor

    def get_videos(self, split="1"):
        split_path = 'splits/HMDB51/'
        video_path = '/export/scratch/compvis/datasets/HMDB51/videos/'
        txt_files = sorted([f for f in os.listdir(split_path) if f[-5] == split])
        videos = []
        if self.mode == 'train':
            for file in txt_files:
                category = file[:-16] + '/'
                with open(split_path + file, 'r') as f:
                    lines = f.readlines()
                    vids = list(map(lambda x: x.split(' ')[0], lines))
                    idxs = list(map(lambda x: x.split(' ')[1], lines))
                    for i, vid in enumerate(vids):
                        if idxs[i] == '1':
                            videos.append(video_path + category + vid)
        elif self.mode == 'test':
            for file in txt_files:
                category = file[:-16] + '/'
                with open(split_path + file, 'r') as f:
                    lines = f.readlines()
                    vids = list(map(lambda x: x.split(' ')[0], lines))
                    idxs = list(map(lambda x: x.split(' ')[1], lines))
                    for i, vid in enumerate(vids):
                        if idxs[i] == '2':
                            videos.append(video_path + category + vid)

        return videos

    def get_categories(self):
        categories = {}
        for video in self.videos:
            category = video.split('/')[7]
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

    def crop_center_square(self, frame):
        y, x = frame.shape[0:2]
        min_dim = min(y, x)
        start_x = (x // 2) - (min_dim // 2)
        start_y = (y // 2) - (min_dim // 2)
        return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

    def load_video(self, path, max_frames=0, resize=(224, 224)):
        cap = cv2.VideoCapture(path)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = self.crop_center_square(frame)
                # TODO: see if crop center is important
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)

                if len(frames) == max_frames:
                    break
        finally:
            cap.release()
        return np.array(frames, dtype=np.uint8)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, item):
        # Get video
        video_path = self.videos[item]
        video = self.load_video(video_path)

        # Select random frame
        n_frames = video.shape[0]
        if self.mode == 'test' and self.avg:
            frames = video
        elif self.mode == 'test':
            frame_idx = n_frames // 2
            frame = video[frame_idx]
        else:
            frame_idx = np.random.randint(0, n_frames)
            frame = video[frame_idx]

        # Get label
        category = video_path.split('/')[7]
        label = self.label_dict[category]

        if self.get_torch_tensor and self.avg:
            frame = torch.cat([self.transforms(image=frame)["image"].unsqueeze(0) for frame in frames], dim=0)
            label = torch.tensor(label).long()
        elif self.get_torch_tensor:
            frame = self.transforms(image=frame)["image"]
            label = torch.tensor(label).long()

        return frame, label


class Kinetics400(Dataset):
    def __init__(self, mode, augm, avg, get_torch_tensor=True):
        self.num_classes = 400

        self.base_path = '/export/data/compvis/kinetics/'
        self.mode = mode if mode == 'train' else 'eval'
        self.annotations = self.get_annotations()
        if mode == 'train' and augm:
            self.transforms = A.Compose(
                [A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.25),
                 A.RandomBrightnessContrast(p=0.25),
                 A.CLAHE(p=0.25),
                 A.Flip(p=0.25),
                 A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=0.25),
                 A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                 ToTensorV2()])
        else:
            self.transforms = A.Compose(
                [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                 ToTensorV2()])

        self.categories, self.label_dict = self.get_categories()
        self.avg = avg
        self.get_torch_tensor = get_torch_tensor

    def get_annotations(self):
        if self.mode == 'train':
            annotations = pd.read_csv(self.base_path + 'train.csv')
            with open('valid_train_ids.csv', 'r') as csvfile:
                reader = csv.reader(csvfile)
                valid_ids = [int(row[0]) for row in reader if row]
            annotations = annotations[annotations.index.isin(valid_ids)]
        else:
            annotations = pd.read_csv(self.base_path + 'validate.csv')
            with open('valid_test_ids.csv', 'r') as csvfile:
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

    def crop_center_square(self, frame):
        y, x = frame.shape[0:2]
        min_dim = min(y, x)
        start_x = (x // 2) - (min_dim // 2)
        start_y = (y // 2) - (min_dim // 2)
        return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

    def load_video(self, path, start_frame, end_frame, resize=(224, 224)):
        frames = []
        for filename in sorted(glob.glob(path + '/*.jpg')):
            frame_id = int(filename[-8:-4])
            if start_frame <= frame_id < end_frame:
                frame = cv2.imread(filename)
                frame = self.crop_center_square(frame)
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]  # BGR -> RGB TODO: see if it matters
                frames.append(frame)

        return np.array(frames, dtype=np.uint8)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):
        # Get video
        video_path = self.base_path + self.mode + '/' + self.annotations['label'][item] + '/' + self.annotations['youtube_id'][item]
        video_path = video_path.replace(' ', '_')
        video = self.load_video(video_path, self.annotations['time_start'][item], self.annotations['time_end'][item])

        # Select random frame
        n_frames = video.shape[0]
        if self.mode == 'eval' and self.avg:
            frames = video
        elif self.mode == 'eval':
            frame_idx = n_frames // 2
            frame = video[frame_idx]
        else:
            frame_idx = np.random.randint(0, n_frames)
            frame = video[frame_idx]

        # Get label
        category = self.annotations['label'][item]
        label = self.label_dict[category]

        if self.get_torch_tensor and self.avg:
            frame = torch.cat([self.transforms(image=frame)["image"].unsqueeze(0) for frame in frames], dim=0)
            label = torch.tensor(label).long()
        elif self.get_torch_tensor:
            frame = self.transforms(image=frame)["image"]
            label = torch.tensor(label).long()

        return frame, label


class Something2SomethingV2(Dataset):
    def __init__(self, mode, augm, avg, get_torch_tensor=True):
        self.num_classes = 174
        self.mode = mode if mode == 'train' else 'validation'
        self.annotations = self.get_annotations()
        self.videos, self.labels = self.get_videos()
        if mode == 'train' and augm:
            self.transforms = A.Compose(
                [A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.25),
                A.RandomBrightnessContrast(p=0.25),
                A.CLAHE(p=0.25),
                A.Flip(p=0.25),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=0.25),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()])
        else:
            self.transforms = A.Compose(
                [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()])

        self.categories, self.label_dict = self.get_categories()
        self.avg = avg
        self.get_torch_tensor = get_torch_tensor

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

    def crop_center_square(self, frame):
        y, x = frame.shape[0:2]
        min_dim = min(y, x)
        start_x = (x // 2) - (min_dim // 2)
        start_y = (y // 2) - (min_dim // 2)
        return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

    def load_video(self, path, max_frames=0, resize=(224, 224)):
        cap = cv2.VideoCapture(path)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = self.crop_center_square(frame)
                # TODO: see if crop center is important
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)

                if len(frames) == max_frames:
                    break
        finally:
            cap.release()
        return np.array(frames, dtype=np.uint8)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, item):
        # Get video
        video_path = self.videos[item]
        video = self.load_video(video_path)

        # Select random frame
        n_frames = video.shape[0]
        if n_frames == 0:
            return torch.tensor(-1), torch.tensor(item)

        if self.mode == 'validation' and self.avg:
            frames = video
        elif self.mode == 'validation':
            frame_idx = n_frames // 2
            frame = video[frame_idx]
        else:
            frame_idx = np.random.randint(0, n_frames)
            frame = video[frame_idx]

        # Get label
        category = self.labels[item]
        label = int(self.label_dict[category])

        if self.get_torch_tensor and self.avg:
            frame = torch.cat([self.transforms(image=frame)["image"].unsqueeze(0) for frame in frames], dim=0)
            label = torch.tensor(label).long()
        elif self.get_torch_tensor:
            frame = self.transforms(image=frame)["image"]
            label = torch.tensor(label).long()

        return frame, label


__datasets__ = {'ucf': UCF101,
                'hmdb': HMDB51,
                'kinetics': Kinetics400,
                's2s': Something2SomethingV2}


def get_dataset(dataset_name):
    return __datasets__[dataset_name]


if __name__ == '__main__':
    dataset = Something2SomethingV2(mode='test', augm=False, avg=True, get_torch_tensor=True)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)
    for step, (frames, labels) in enumerate(train_loader):
        if frames == -1:
            print(f"None returned at item: {labels}")
        break
    # dataset.print_summary()
    print(dataset[0][0].shape)
    print(dataset[0][1])