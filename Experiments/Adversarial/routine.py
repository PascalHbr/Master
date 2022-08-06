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
    test_dataset = Dataset('test', arg.model, arg.augm, num_iterations=3)

    # Initialize model
    Model = get_model(arg.model)
    model = Model(num_classes=test_dataset.num_classes, pretrained=True, freeze=False,
                  keep_head=True, device=device, pre_dataset=arg.pre_dataset).to(device)

    # Test loop
    model.eval()
    stats_adverserial = {}
    stats_distribution = {}

    for augm in ["alpha_freeze", "seq_end_freeze", "seq_start_freeze", "random_frame_drop"]:
        test_dataset.augm = augm
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
        total_original_predictions = 0
        total_adverserial_predictions = 0
        total_other_predictions = 0
        total_counts = 0
        for step, (video, category_idx, label, second_labels) in enumerate(tqdm(test_loader)):
            with torch.no_grad():
                if arg.model == 'slowfast':
                    video = [vid.to(device)[0] for vid in video]
                else:
                    video = video.to(device)[0]
                label = label.to(device)
                second_labels = second_labels.to(device)

                # forward
                output = nn.Softmax(dim=1)(model(video))
                _, top1 = torch.max(output, 1)

                # statistics
                for i in range(top1.shape[0]):
                    total_original_predictions += check_correct(top1[i], label[0][i])
                    total_adverserial_predictions += check_correct(top1[i], second_labels[0][i])
                    if check_correct(top1[i], label[0][i]) == 0 and check_correct(top1[i], second_labels[0][i]) == 0:
                        total_other_predictions += 1
                    total_counts += 1

        stats_adverserial[augm] = {"original": total_original_predictions / total_counts,
                                   "adversarial": total_adverserial_predictions / total_counts,
                                   "other": total_other_predictions / total_counts}

    for augm in ["alpha", "seq_end", "seq_start"]:
        test_dataset.augm = augm
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
        total_original_predictions = 0
        total_adverserial_predictions = 0
        total_other_predictions = 0
        total_both_in_top2 = 0
        total_both_in_top5 = 0
        total_counts = 0
        for step, (video, category_idx, label, second_labels) in enumerate(tqdm(test_loader)):
            with torch.no_grad():
                if arg.model == 'slowfast':
                    video = [vid.to(device)[0] for vid in video]
                else:
                    video = video.to(device)[0]
                label = label.to(device)
                second_labels = second_labels.to(device)

                # forward
                output = nn.Softmax(dim=1)(model(video))
                _, top1 = torch.max(output, 1)
                _, top2 = output.topk(2)
                _, top5 = output.topk(5)

                # statistics
                for i in range(top1.shape[0]):
                    total_original_predictions += check_correct(top1[i], label[0][i])
                    total_adverserial_predictions += check_correct(top1[i], second_labels[0][i])
                    if check_correct(top1[i], label[0][i]) == 0 and check_correct(top1[i], second_labels[0][i]) == 0:
                        total_other_predictions += 1
                    if int(label[0][i].item() in top2[i].cpu().detach().numpy().tolist()) \
                            and int(second_labels[0][i].item() in top2[i].cpu().detach().numpy().tolist()):
                        total_both_in_top2 += 1
                    if int(label[0][i].item() in top5[i].cpu().detach().numpy().tolist()) \
                            and int(second_labels[0][i].item() in top5[i].cpu().detach().numpy().tolist()):
                        total_both_in_top5 += 1
                    total_counts += 1

            stats_distribution[augm] = {"original": total_original_predictions / total_counts,
                                       "adversarial": total_adverserial_predictions / total_counts,
                                       "other": total_other_predictions / total_counts,
                                       "top 2": total_both_in_top2 / total_counts,
                                       "top 5": total_both_in_top5 / total_counts}

    print(f"Model: {arg.model}")
    print(f"---------Adverserial---------")
    print(" \n {:<20s} {:<20s} {:<20s} {:<20s}".format("Adversarial Method", "Original", "Adversarial", "Other"))
    for k, v in stats_adverserial.items():
        print("{:<20s} {:<20.1%} {:<20.1%} {:<20.1%} \n".format(k, v["original"], v["adversarial"], v["other"]))
    print(f"---------Distribution---------")
    print(
        " \n {:<20s} {:<20s} {:<20s} {:<20s} {:<20s} {:<20s}".format("Adversarial Method", "Original", "Adversarial", "Other", "Top 2", "Top 5"))
    for k, v in stats_distribution.items():
        print("{:<20s} {:<20.1%} {:<20.1%} {:<20.1%} {:<20.1%} {:<20.1%} \n".format(k, v["original"], v["adversarial"], v["other"],
                                                                       v["top 2"], v["top 5"]))


if __name__ == '__main__':
    arg = parse_args()
    main(arg)
