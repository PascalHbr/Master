import torch.nn as nn
import pytorchvideo.models.resnet


def make_kinetics_resnet():
    return pytorchvideo.models.resnet.create_resnet(
        input_channel=3,  # RGB input from Kinetics
        model_depth=50,  # For the tutorial let's just use a 50 layer network
        model_num_class=400,  # Kinetics has 400 classes so we need out final head to align
        norm=nn.BatchNorm3d,
        activation=nn.ReLU,
    )
