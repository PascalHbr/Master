import random
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from glob import glob
import os
from natsort import natsorted
import cv2
import torchvision


class KineticsDataset(Dataset):
    def __init__(self, mode, n_frames):
        super(KineticsDataset, self).__init__()
        self.n_frames = n_frames
        self.base_path = f"/export/data/compvis/kinetics/{mode}/"
        self.categories = os.listdir(self.base_path)
        self.video_paths = glob(os.path.join(self.base_path, "*/*"))
        self.transform = transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, item):
        video_path = self.video_paths[item]
        images = natsorted(glob(os.path.join(video_path, "*")))
        start_frame = random.randint(0, len(images) - self.n_frames)
        frame_paths = images[start_frame:start_frame + self.n_frames]
        frames = torch.cat([self.transform(np.array(cv2.imread(frame_path))).unsqueeze(0) for frame_path in frame_paths])

        return frames


if __name__ == '__main__':
    dataset = KineticsDataset(mode='train', n_frames=10)
    print(torch.mean(dataset[0]))
