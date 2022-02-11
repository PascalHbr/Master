import numpy as np
from urllib import request
import glob
import pandas as pd
import cv2
from torch.utils.data import Dataset
import kornia

from utils import *


class UCFDataset(Dataset):
    def __init__(self, path_labels, augmentation, factor, get_torch_tensor=False, shuffle=True):
        self.dataset_path = "/export/scratch/compvis/datasets/UCF101/videos"

        self.kinetics_labels = self.read_kinetics_labels()
        self.ucf_labels = self.read_ucf_labels(path_labels)
        self.categories, self.label_dict = self.get_categories()
        self.videos = np.array([item for sublist in self.categories.values() for item in sublist])
        if shuffle:
            np.random.shuffle(self.videos)

        self.shuffler = Shuffler()

        self.augmentation = augmentation
        self.factor = factor
        self.get_torch_tensor = get_torch_tensor

    def read_kinetics_labels(self):
        KINETICS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
        with request.urlopen(KINETICS_URL) as obj:
            labels = [line.decode("utf-8").strip() for line in obj.readlines()]
        return labels

    def read_ucf_labels(self, path_labels):
        labels = pd.read_csv(path_labels)["Corresponding Kinetics Labels"].to_list()
        labels = [list(map(int, sublist.split())) for sublist in labels]
        return labels

    def get_categories(self):
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
            kinetics_ids = self.ucf_labels[i]
            if kinetics_ids[0] >= 0:
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
        return np.array(frames) / 255.0

    def augment_video(self, video):
        # Only use the middle frame of the video
        if 'freeze' in self.augmentation:
            frames, _, _, _ = video.shape
            video = np.tile(np.expand_dims(video[frames // 2], 0), (frames, 1, 1, 1))

        if 'shuffle' in self.augmentation:
            video = self.shuffler.patchify(video, patch_size=56)

        if 'brightness' in self.augmentation:
            video = kornia.enhance.adjust_brightness(torch.from_numpy(video).unsqueeze(0).permute(0, 1, 4, 2, 3), self.factor)
            video = video.permute(0, 1, 3, 4, 2).squeeze(0).numpy()
            video = (video - np.min(video)) / (np.max(video) - np.min(video))

        if 'saturation' in self.augmentation:
            video = kornia.enhance.adjust_saturation(torch.from_numpy(video).unsqueeze(0).permute(0, 1, 4, 2, 3), 1 - self.factor)
            video = video.permute(0, 1, 3, 4, 2).squeeze(0).numpy()
            video = (video - np.min(video)) / (np.max(video) - np.min(video))

        if 'hue' in self.augmentation:  # should be in [-pi, pi]
            video = kornia.enhance.adjust_hue(torch.from_numpy(video).unsqueeze(0).permute(0, 1, 4, 2, 3), self.factor)
            video = video.permute(0, 1, 3, 4, 2).squeeze(0).numpy()
            video = (video - np.min(video)) / (np.max(video) - np.min(video))

        if 'posterize' in self.augmentation:
            video = kornia.enhance.posterize(torch.from_numpy(video).unsqueeze(0).permute(0, 1, 4, 2, 3), bits=3)
            video = video.permute(0, 1, 3, 4, 2).squeeze(0).numpy()
            video = (video - np.min(video)) / (np.max(video) - np.min(video))

        if 'clahe' in self.augmentation:
            video = kornia.enhance.equalize_clahe(torch.from_numpy(video).unsqueeze(0).permute(0, 1, 4, 2, 3))
            video = video.permute(0, 1, 3, 4, 2).squeeze(0).numpy()
            video = (video - np.min(video)) / (np.max(video) - np.min(video))

        if 'solarize' in self.augmentation:
            video = kornia.enhance.solarize(torch.from_numpy(video).unsqueeze(0).permute(0, 1, 4, 2, 3))
            video = video.permute(0, 1, 3, 4, 2).squeeze(0).numpy()
            video = (video - np.min(video)) / (np.max(video) - np.min(video))

        if 'shuffle_frames' in self.augmentation:
            np.random.shuffle(video)

        if 'reverse' in self.augmentation:
            video = np.flip(video, axis=0).copy()

        return video

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, item):
        video_path = self.videos[item]
        video_array = self.load_video(video_path)
        if self.augmentation is not None:
            video_array = self.augment_video(video_array)

        category = video_path.split('/')[-1][2:-12]
        labels = self.label_dict[category]

        if self.get_torch_tensor:
            video_array = torch.tensor(video_array)

        return video_array, category, labels


if __name__ == '__main__':
    dataset = UCFDataset('labels.csv', augmentation=[], shuffle=False)
    print(dataset[0][1])
    print(dataset[1][1])
    print(dataset[2][1])