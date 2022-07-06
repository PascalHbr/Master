import torchvision
from tqdm import tqdm

from config import *
from dataset import get_dataset, DataLoader
from utils import *


def main(arg):
    # Set device
    device = torch.device('cuda:' + str(arg.gpu[0]) if torch.cuda.is_available() else 'cpu')

    # Model
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True, progress=False).to(device)
    model.eval()

    # Datasets
    Dataset = get_dataset(arg.dataset)
    train_set = Dataset(mode='train', model=arg.model)
    train_loader = DataLoader(train_set, batch_size=1)
    test_set = Dataset(mode='test', model=arg.model)
    test_loader = DataLoader(test_set, batch_size=1)

    # Iteration
    for step, (video, video_path) in enumerate(tqdm(train_loader)):
        with torch.no_grad():
            video = video.to(device)
            segmentation_masks = []
            for i in range(0, video.shape[1], 100):  # process 100 frames at once
                frames = video[0][i:i+100].unsqueeze(0)
                frames_segmentation_masks = get_segmentation_mask(model, frames)
                segmentation_masks.append(frames_segmentation_masks)
            segmentation_mask = torch.cat(segmentation_masks, dim=0)
            save_segmentation_masks(arg.model, segmentation_mask, video_path, "train", arg.dataset)

    for step, (video, video_path) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            video = video.to(device)
            segmentation_masks = []
            for i in range(0, video.shape[1], 100):  # process 100 frames at once
                frames = video[0][i:i + 100].unsqueeze(0)
                frames_segmentation_masks = get_segmentation_mask(model, frames)
                segmentation_masks.append(frames_segmentation_masks)
            segmentation_mask = torch.cat(segmentation_masks, dim=0)
            save_segmentation_masks(arg.model, segmentation_mask, video_path, "test", arg.dataset)


if __name__ == '__main__':
    arg = parse_args()
    main(arg)