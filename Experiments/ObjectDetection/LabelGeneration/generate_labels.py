import torchvision
from tqdm import tqdm
from config import *

from dataset import *
from utils import *


def main(arg):
    # Set device
    device = torch.device('cuda:' + str(arg.gpu[0]) if torch.cuda.is_available() else 'cpu')

    # Model
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True).to(device)
    model.eval()

    # Datasets
    train_set = UCF101(mode='train', model=arg.model)
    train_loader = DataLoader(train_set)
    test_set = UCF101(mode='test', model=arg.model)
    test_loader = DataLoader(test_set, batch_size=1)

    # Iteration
    for step, (video, video_path) in enumerate(tqdm(train_loader)):
        video = video.to(device)
        valid = True
        bboxes = []
        for i in range(0, video.shape[1], 12):  # process 10 frames at once
            frames = []
            for frame in video[0][i:i+12]:  # convert to list of tensors
                frames.append(frame)
            pred_boxes, pred_classes = get_prediction(model, frames, threshold=arg.threshold)
            for j, pred_class in enumerate(pred_classes):
                if pred_class.count('person') == 1:  # check if there is only 1 person in the frame, else discard
                    bboxes.append(pred_boxes[j][pred_class.index('person')])
                else:
                    valid = False
                    break
            if not valid:
                break

        if len(bboxes) == video.shape[1]:
            with open(f'labels/{arg.model}_train.txt', 'a+') as f:
                for k, bbox in enumerate(bboxes):
                    f.write(f'{video_path[0]} {k} {bbox} \n')

    for step, (video, video_path) in enumerate(tqdm(test_loader)):
        video = video.to(device)
        valid = True
        bboxes = []
        for i in range(0, video.shape[1], 12):  # process 10 frames at once
            frames = []
            for frame in video[0][i:i+12]:  # convert to list of tensors
                frames.append(frame)
            pred_boxes, pred_classes = get_prediction(model, frames, threshold=0.7)
            for j, pred_class in enumerate(pred_classes):
                if pred_class.count('person') == 1:  # check if there is only 1 person in the frame, else discard
                    bboxes.append(pred_boxes[j][pred_class.index('person')])
                else:
                    valid = False
                    break
            if not valid:
                break

        if len(bboxes) == video.shape[1]:
            with open(f'labels/{arg.model}_test.txt', 'a+') as f:
                for k, bbox in enumerate(bboxes):
                    f.write(f'{video_path[0]} {k} {bbox} \n')


if __name__ == '__main__':
    arg = parse_args()
    main(arg)