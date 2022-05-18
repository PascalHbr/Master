import os
from glob import glob

from torch.utils.data import Dataset, DataLoader
from utils import *
import pandas as pd
from urllib import request

from torchvision.transforms import Compose, Lambda
from torchvision.transforms.transforms import Resize
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)

from pytorchvideo.transforms import (
    ShortSideScale,
    UniformTemporalSubsample,
)

import wandb


class UCF101(Dataset):
    def __init__(self, mode, model, target, classification, blackout=None):
        self.dataset_path = "/export/scratch/compvis/datasets/UCF101/videos"
        self.mode = mode
        self.model = model
        self.target = target
        self.classification = classification
        self.blackout = blackout

        self.transform = self.set_transform()
        self.frame_transform = self.set_frame_transform()
        self.category_dict = {i: category for i, category in enumerate(list(sorted(set(os.listdir("/export/scratch/compvis/datasets/UCF101/videos/")))))}
        self.category_to_idx_dict = {v.lower(): k for k, v in self.category_dict.items()}

        self.kinetics_labels = self.read_kinetics_labels()
        self.ucf_labels = self.read_ucf_labels()

        if self.classification:
            self.data_dict, self.label_dict = self.get_classification_data()
            self.videos = np.array([item for sublist in self.data_dict.values() for item in sublist])
            self.num_classes = 400

        else:
            self.videos, self.segmentation_masks = self.get_all_videos()
            self.data_dict, _ = self.get_data()
            self.num_classes = None

    def read_kinetics_labels(self):
        KINETICS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
        with request.urlopen(KINETICS_URL) as obj:
            labels = [line.decode("utf-8").strip() for line in obj.readlines()]
        return labels

    def read_ucf_labels(self):
        labels = pd.read_csv("/export/home/phuber/Master/I3D_augmentations/labels.csv")["Corresponding Kinetics Labels"].to_list()
        return labels

    def get_all_videos(self):
        segmentation_masks = glob(f'/export/home/phuber/archive/Segmentation/{self.model}/{self.mode}/*/*.npy')
        videos = ["/export/scratch/compvis/datasets/UCF101/videos" + mask.split(f"{self.mode}")[-1][:-4] + ".avi"
                  for mask in segmentation_masks]
        videos = list(sorted(set(videos)))
        segmentation_masks = list(sorted(set(segmentation_masks)))

        return videos, segmentation_masks

    def get_data(self):
        data_dict = {}
        for video in self.videos:
            category = video.split('/')[-1][2:-12]
            if category not in data_dict:
                data_dict[category] = []
            data_dict[category].append(video)

        # Make dictionary that gives kinetics label for category
        label_dict = {}
        for i, category in enumerate(data_dict):
            label_dict[category] = i

        print("Found %d videos in %d categories." % (sum(list(map(len, data_dict.values()))),
                                                     len(data_dict)))
        return data_dict, label_dict

    def get_classification_data(self):
        videos, self.segmentation_masks = self.get_all_videos()
        categories = {}
        for video in videos:
            category = video.split('/')[-1][2:-12]
            if category not in categories:
                categories[category] = []
            categories[category].append(video)

        # Choose selected categories from labels.txt
        valid_categories = []
        for (i, category) in self.category_dict.items():
            kinetics_id = self.ucf_labels[i]
            if kinetics_id >= 0 and category in categories.keys():
                valid_categories.append(category)
        selected_categories = {valid_category: categories[valid_category] for valid_category in valid_categories}

        # Make dictionary that gives kinetics label for category
        label_dict = {}
        for (i, category) in self.category_dict.items():
            if category in selected_categories:
                label_dict[category] = self.ucf_labels[i]

        print("Found %d videos in %d categories." % (sum(list(map(len, selected_categories.values()))),
                                                     len(selected_categories)))
        return selected_categories, label_dict

    def print_summary(self):
        for category, sequences in self.data_dict.items():
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

    def temporal_subsample(self, segmentation_masks, num_frames):
        temporal_dim = 0
        t = segmentation_masks.shape[temporal_dim]

        indices = torch.linspace(0, t - 1, num_frames)
        indices = torch.clamp(indices, 0, t - 1).long()

        return indices, torch.index_select(segmentation_masks, temporal_dim, indices)

    def set_transform(self):
        if self.model == 'slow':
            self.side_size = 256
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
            self.crop_size = 256
            self.num_frames = 8

            transform = Compose(
                [
                    UniformTemporalSubsample(self.num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(size=self.side_size),
                    CenterCropVideo(crop_size=(self.crop_size, self.crop_size))
                ]
            )

        elif self.model == 'slowfast':
            self.side_size = 256
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
            self.crop_size = 256
            self.num_frames = 32
            self.alpha = 4

            transform = Compose(
                [
                    UniformTemporalSubsample(self.num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(size=self.side_size),
                    CenterCropVideo(self.crop_size),
                    PackPathway(self.alpha)
                ]
            )

        elif self.model == 'x3d':
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
            model_transform_params = {
                "x3d_m": {
                    "side_size": 256,
                    "crop_size": 256,
                    "num_frames": 16,
                    "sampling_rate": 5,
                }
            }

            model_name = 'x3d_m'
            transform_params = model_transform_params[model_name]
            self.side_size = transform_params["side_size"]
            self.crop_size = transform_params["crop_size"]
            self.num_frames = transform_params["num_frames"]
            transform = Compose(
                [
                    UniformTemporalSubsample(self.num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(size=self.side_size),
                    CenterCropVideo(crop_size=(self.crop_size, self.crop_size))
                ]
            )

        elif self.model == 'mvit':
            self.side_size = 256
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
            self.crop_size = 224
            self.num_frames = 16

            transform = Compose(
                [
                    UniformTemporalSubsample(self.num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(size=self.side_size),
                    CenterCropVideo(self.crop_size),
                ]
            )

        elif self.model == 'vimpac':
            self.side_size = 256
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            self.resize = 160
            self.crop_size = 128
            self.num_frames = 5

            transform = Compose(
                [
                    UniformTemporalSubsample(self.num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(size=self.side_size),
                    Resize(size=self.resize),
                    CenterCropVideo(self.crop_size),
                ]
            )

        elif self.model == 'mae':
            self.side_size = 256
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            self.crop_size = 224
            self.num_frames = 16

            transform = Compose(
                [
                    UniformTemporalSubsample(self.num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(size=self.side_size),
                    CenterCropVideo(self.crop_size),
                ]
            )

        return transform

    def set_frame_transform(self):
        if self.model == 'vimpac':
            frame_transform = Compose(
                [
                    Lambda(lambda x: x / 255.0),
                    ShortSideScale(size=self.side_size),
                    Resize(size=self.resize),
                    CenterCropVideo(self.crop_size),
                ]
            )
        else:
            frame_transform = Compose(
                [
                    Lambda(lambda x: x / 255.0),
                    ShortSideScale(size=self.side_size),
                    CenterCropVideo(self.crop_size),
                ]
            )

        return frame_transform

    def blackout_bboxes(self, video, bboxes):  # TODO
        for i in range(video.shape[1]):
            video[:, i, int(bboxes[i, 1].item()):int(bboxes[i, 3].item()),
                  int(bboxes[i, 0].item()):int(bboxes[i, 2].item())] = 0.

        return video

    def blackout_background(self, video, bboxes):  # TODO
        for i in range(video.shape[1]):
            video[:, i, :int(bboxes[i, 1].item()), :] = 0.
            video[:, i, :, :int(bboxes[i, 0].item())] = 0.
            video[:, i, int(bboxes[i, 3].item()):, :] = 0.
            video[:, i, :, int(bboxes[i, 2].item()):] = 0.

        return video

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, item):
        # Get video
        video_path = self.videos[item]
        video = self.load_video(video_path)
        segmentation_masks = np.expand_dims(np.load(self.segmentation_masks[item]), axis=3)
        category = video_path.split('/')[-1][2:-12]
        category_idx = self.category_to_idx_dict[category.lower()]

        if self.classification:
            category_label = self.label_dict[category]
        else:
            category_label = -1

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
        segmentation_masks = segmentation_masks[start_frame:start_frame + n_frames]

        # Make transformation
        video = torch.from_numpy(video_org).permute(3, 0, 1, 2)
        video = self.transform(video)
        if self.model == 'vimpac':
            video = video.permute(1, 0, 2, 3)
        segmentation_masks = torch.from_numpy(segmentation_masks)
        if self.model == 'slowfast':
            indices, segmentation_masks = self.temporal_subsample(segmentation_masks, self.num_frames // self.alpha)
            indices_fast, segmentation_masks_fast = self.temporal_subsample(segmentation_masks, self.num_frames)
        else:
            indices, segmentation_masks = self.temporal_subsample(segmentation_masks, self.num_frames)

        # Select frames according to indices
        if self.classification or self.target == 'all':
            frame = torch.from_numpy(video_org).permute(3, 0, 1, 2)
            frame = self.frame_transform(frame).permute(1, 2, 3, 0)
            if self.model == 'slowfast':
                segmentation_mask = [segmentation_masks.float(), segmentation_masks_fast.float()][0]
            else:
                segmentation_mask = segmentation_masks.float()
        else:
            if self.target == 'first':
                indices = indices[0]
                frame = np.take(video_org, indices, 0)
            elif self.target == 'middle':
                indices = indices[len(indices) // 2]
                frame = np.take(video_org, indices, 0)
            elif self.target == 'last':
                indices = indices[-1]
                frame = np.take(video_org, indices, 0)
            frame = torch.from_numpy(frame).unsqueeze(0).permute(3, 0, 1, 2)
            frame = self.frame_transform(frame).permute(1, 2, 3, 0)
            segmentation_mask = segmentation_masks[indices].float()

        if self.classification:
            # if self.blackout == 'bboxes':
            #     if self.model == 'slowfast':
            #         video = [self.blackout_bboxes(video[0], bboxes.reshape(-1, 4) * self.crop_size),
            #                  self.blackout_bboxes(video[1], bboxes_fast.reshape(-1, 4) * self.crop_size)]
            #     else:
            #         video = self.blackout_bboxes(video, bboxes.reshape(-1, 4) * self.crop_size)
            # elif self.blackout == 'background':
            #     if self.model == 'slowfast':
            #         video = [self.blackout_background(video[0], bboxes.reshape(-1, 4) * self.crop_size),
            #                  self.blackout_background(video[1], bboxes_fast.reshape(-1, 4) * self.crop_size)]
            #     else:
            #         video = self.blackout_background(video, bboxes.reshape(-1, 4) * self.crop_size)
            return video, segmentation_mask, category_label, frame, category_idx

        return video, segmentation_mask, category_label, frame, category_idx


__datasets__ = {'ucf': UCF101}


def get_dataset(dataset_name):
    return __datasets__[dataset_name]


if __name__ == '__main__':
    dataset = UCF101(mode='train', model='slow', target='first', classification=False, blackout=None)
    video, segmentation_mask, category_label, frame, category_idx = dataset[525]
    print(frame.shape)
    print(segmentation_mask.shape)
    # wandb.init(project='Sample Segmentation Image')
    # wandb.log({"video": wandb.Image(frame[0].numpy())})
    # wandb.log({"segmentation": wandb.Image(segmentation_mask.numpy())})
    # wandb.log({"video": wandb.Video(video.permute(1, 0, 2, 3).numpy(), fps=4, format="mp4")})
    # wandb.log({"segmentation": wandb.Video(segmentation_mask.permute(1, 0, 2, 3).numpy(), fps=4, format="mp4")})
