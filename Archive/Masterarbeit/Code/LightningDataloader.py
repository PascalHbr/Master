import os
import pytorch_lightning
import pytorchvideo.data
import torch.utils.data

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip
)


class KineticsDataModule(pytorch_lightning.LightningDataModule):
    # Dataset configuration
    _DATA_PATH = "/export/data/compvis/kinetics/"
    _CLIP_DURATION = 2  # Duration of sampled clip for each video
    _BATCH_SIZE = 8
    _NUM_WORKERS = 8  # Number of parallel processes fetching data

    def train_dataloader(self):
        """
          Create the Kinetics train partition from the list of video labels
          in {self._DATA_PATH}/train.csv. Add transform that subsamples and
          normalizes the video before applying the scale, crop and flip augmentations.
          """
        train_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(8),
                            Lambda(lambda x: x / 255.0),
                            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                            RandomShortSideScale(min_size=256, max_size=320),
                            RandomCrop(244),
                            RandomHorizontalFlip(p=0.5),
                        ]
                    ),
                ),
            ]
        )
        train_dataset = pytorchvideo.data.Kinetics(
            data_path="/export/home/phuber/Master/Code/train.csv",
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", self._CLIP_DURATION),
            transform=train_transform
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )

    def val_dataloader(self):
        """
          Create the Kinetics train partition from the list of video labels
          in {self._DATA_PATH}/train.csv. Add transform that subsamples and
          normalizes the video before applying the scale, crop and flip augmentations.
          """
        val_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(8),
                            Lambda(lambda x: x / 255.0),
                            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                            RandomShortSideScale(min_size=256, max_size=320),
                            RandomCrop(244),
                            RandomHorizontalFlip(p=0.5),
                        ]
                    ),
                ),
            ]
        )
        val_dataset = pytorchvideo.data.Kinetics(
            data_path="/export/home/phuber/Master/Code/val.csv",
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self._CLIP_DURATION),
            transform=val_transform
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )
