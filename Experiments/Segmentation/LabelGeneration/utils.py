import torch
import numpy as np
import os

SEM_CLASSES = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]


def get_segmentation_mask(model, frames):
    output = model(frames[0])['out']  # Pass the image to the model
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(SEM_CLASSES)}

    normalized_masks = torch.nn.functional.softmax(output, dim=1)

    class_dim = 1
    boolean_person_masks = (normalized_masks.argmax(class_dim) == sem_class_to_idx['person'])

    return boolean_person_masks


def save_segmentation_masks(model, segmentation_masks, video_path, mode):
    category = video_path[0].split("/")[-2]
    video_name = video_path[0].split("/")[-1]
    if not os.path.exists(f'/export/home/phuber/archive/Segmentation/{model}/{mode}/{category}'):
        os.makedirs(f'/export/home/phuber/archive/Segmentation/{model}/{mode}/{category}')
    np.save(f'/export/home/phuber/archive/Segmentation/{model}/{mode}/{category}/{video_name[:-4]}.npy', segmentation_masks.detach().cpu().numpy())