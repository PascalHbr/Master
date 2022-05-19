from torch.utils.data import Dataset, DataLoader
from utils import *
from models import *

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
    def __init__(self, mode, model):
        self.dataset_path = "/export/scratch/compvis/datasets/UCF101/videos"
        self.mode = mode
        self.model = model

        self.transform = self.set_transform()
        self.videos = self.get_all_videos()
        self.data_dict, self.label_dict = self.get_data()

    def get_all_videos(self):
        with open(f'/export/home/phuber/Master/ImageClassification/splits/UCF101/{self.mode}list.txt', 'r') as f:
            if self.mode == 'train':
                videos = list(map(lambda x: x.split(' ')[0], f.readlines()))
            else:
                videos = list(map(lambda x: x.split(' ')[0][:-1], f.readlines()))

        videos = list(sorted(set(videos)))

        return videos

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

    def scale_coordinates(self, bbox):
        x_start, y_start = bbox[0]
        x_end, y_end = bbox[1]

        return np.array([x_start, y_start, x_end, y_end]) / self.crop_size

    def temporal_subsample(self, bboxes, num_frames):
        temporal_dim = 0
        t = bboxes.shape[temporal_dim]

        indices = torch.linspace(0, t - 1, num_frames)
        indices = torch.clamp(indices, 0, t - 1).long()

        return indices, torch.index_select(bboxes, temporal_dim, indices)

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
        label = self.label_dict[video_path.split('/')[-1][2:-12]]

        # Take model-specific number of frames
        if self.model in ['slow', 'slowfast', 'mvit', 'mae']:
            n_frames = 64
        elif self.model == 'x3d':
            n_frames = 80
        elif self.model == 'vimpac':
            n_frames = 32
        start_frame = max((video.shape[0] - n_frames) // 2, 0)
        video_org = video[start_frame:start_frame + n_frames]

        # Make transformation
        video = torch.from_numpy(video_org).permute(3, 0, 1, 2)
        video = self.transform(video)
        if self.model == 'vimpac':
            video = video.permute(1, 0, 2, 3)

        return video, video_path[47:-4]


__datasets__ = {'ucf': UCF101}


def get_dataset(dataset_name):
    return __datasets__[dataset_name]


if __name__ == '__main__':
    dataset = UCF101(mode='test', model='slowfast')
    video, video_path = dataset[0]
    print(video[1].shape)
    print(video_path.split('/')[1])
    # wandb.init(project='Sample Image')
    # wandb.log({"video": wandb.Video(video[1].permute(1, 0, 2, 3).numpy(), fps=4, format="mp4")})
