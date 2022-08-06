import torch
import torch.nn as nn
from config import parse_args
from dataset import get_dataset
from tqdm import tqdm
from utils import set_random_seed, check_correct
from torch.utils.data import DataLoader
from models import get_model


def main(arg):
    # Set random seed
    set_random_seed()

    # Set device
    device = torch.device('cuda:' + str(arg.gpu[0]) if torch.cuda.is_available() else 'cpu')

    # Setup Dataloader and model
    Dataset = get_dataset(arg.dataset)
    test_dataset = Dataset('test', arg.model, arg.augm, arg.app_augm, num_iterations=3)

    # Initialize model
    Model = get_model(arg.model)
    model = Model(num_classes=test_dataset.num_classes, pretrained=True, freeze=False,
                  keep_head=True, device=device, pre_dataset=arg.pre_dataset).to(device)

    # Definde loss function
    criterion = nn.CrossEntropyLoss()

    # Test loop
    model.eval()
    stats = {}

    for app_augm in ["shuffle_32"]:
        test_dataset.app_augm = app_augm
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
        total_top1_corrects = 0
        total_top5_corrects = 0
        total_counts = 0
        for step, (video, category_idx, label) in enumerate(tqdm(test_loader)):
            with torch.no_grad():
                if arg.model == 'slowfast':
                    video = [vid.to(device)[0] for vid in video]
                else:
                    video = video.to(device)[0]
                label = label.to(device)

                # forward
                output = torch.mean(nn.Softmax(dim=1)(model(video)), dim=0, keepdim=True)
                _, top1 = torch.max(output, 1)
                _, top5 = output.topk(5)

                # statistics
                total_top1_corrects += check_correct(top1, label)
                total_top5_corrects += int(label.item() in top5[0].cpu().detach().numpy().tolist())
                total_counts += 1

        if app_augm:
            stats[app_augm] = {"top 1": total_top1_corrects / total_counts,
                               "top 5": total_top5_corrects / total_counts}
        else:
            stats["None"] = {"top 1": total_top1_corrects / total_counts,
                             "top 5": total_top5_corrects / total_counts}

    test_dataset.app_augm = None
    for augm in []:
        test_dataset.augm = augm
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
        total_top1_corrects = 0
        total_top5_corrects = 0
        total_counts = 0
        for step, (video, category_idx, label) in enumerate(tqdm(test_loader)):
            with torch.no_grad():
                if arg.model == 'slowfast':
                    video = [vid.to(device)[0] for vid in video]
                else:
                    video = video.to(device)[0]
                label = label.to(device)

                # forward
                output = torch.mean(nn.Softmax(dim=1)(model(video)), dim=0, keepdim=True)
                _, top1 = torch.max(output, 1)
                _, top5 = output.topk(5)

                # statistics
                total_top1_corrects += check_correct(top1, label)
                total_top5_corrects += int(label.item() in top5[0].cpu().detach().numpy().tolist())
                total_counts += 1

        stats[augm] = {"top 1": total_top1_corrects / total_counts,
                       "top 5": total_top5_corrects / total_counts}

    print(f"Model: {arg.model}")
    print(" \n {:<20s} {:<20s} {:<20s}".format("Augmentation Method", "Top 1 Accuracy (%)", "Top 5 Accuracy (%)"))
    for k, v in stats.items():
        print("{:<20s} {:<20.2%} {:<20.2%}".format(k, v["top 1"], v["top 5"]))


if __name__ == '__main__':
    arg = parse_args()
    main(arg)