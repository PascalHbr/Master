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
    if arg.dataset == 'kinetics_':
        test_dataset_subset = torch.utils.data.Subset(test_dataset, range(0, len(test_dataset) // 15))
        test_loader = DataLoader(test_dataset_subset, batch_size=1, shuffle=False, num_workers=8)
        # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=8)
    else:
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    # Initialize model
    Model = get_model(arg.model)
    model = Model(num_classes=test_dataset.num_classes, pretrained=True, freeze=False,
                  keep_head=True, device=device, pre_dataset=arg.pre_dataset).to(device)

    # Definde loss function
    criterion = nn.CrossEntropyLoss()

    # Test loop
    model.eval()
    total_loss = 0.0
    stats = {}
    total_corrects = 0
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
            _, pred = torch.max(output, 1)
            _, top5 = output.topk(5)
            loss = criterion(output, label.view(-1))

            # Debug
            if arg.debug:
                if pred != label.data:
                    print(f"Prediction: {test_dataset.kinetics_labels[pred.item()]}, {pred.item()}")
                    print(f"Label: {test_dataset.kinetics_labels[label.item()]}, {label.item()}")

            # statistics
            correct = check_correct(pred, label)
            total_loss += loss.item() * label.size(0)
            total_corrects += correct
            total_top5_corrects += int(label.item() in top5[0].cpu().detach().numpy().tolist())
            total_counts += 1
            if test_dataset.idx_to_category[category_idx.item()] not in stats:
                stats[test_dataset.idx_to_category[category_idx.item()]] = {"loss": loss.item(),
                                                                          "corrects": correct,
                                                                          "top5_corrects": int(label.item() in top5[0].cpu().detach().numpy().tolist()),
                                                                          "counts": 1}
            else:
                stats[test_dataset.idx_to_category[category_idx.item()]]["loss"] += loss.item()
                stats[test_dataset.idx_to_category[category_idx.item()]]["corrects"] += correct
                stats[test_dataset.idx_to_category[category_idx.item()]]["top5_corrects"] += int(label.item() in top5[0].cpu().detach().numpy().tolist())
                stats[test_dataset.idx_to_category[category_idx.item()]]["counts"] += 1

    # Print stats
    for category in stats.keys():
        stats[category]["top1_accuracy"] = stats[category]["corrects"] / stats[category]["counts"] * 100
        stats[category]["top5_accuracy"] = stats[category]["top5_corrects"] / stats[category]["counts"] * 100
    stats = dict(sorted(stats.items(), key=lambda item: item[1]["top1_accuracy"], reverse=True))
    print(" \n {:<20s} {:<10s} {:<10s} {:<20s} {:<20s}".format("Category", "# Videos", "Loss", "Top 1 Accuracy (%)", "Top 5 Accuracy (%)"))
    print("-" * 60)
    for key in stats.keys():
        print("{:<20s} {:<10d} {:<10.2f} {:<10.2f} {:<10.2f}".format(key, stats[key]["counts"],
                                                           stats[key]["loss"] / stats[key]["counts"],
                                                           stats[key]["top1_accuracy"],
                                                           stats[key]["top5_accuracy"]))
    print("-" * 50)
    print("{:<20s} {:<10d} {:<10.2f} {:<10.2f} {:<10.2f}".format("Total", total_counts, total_loss / total_counts,
                                                       total_corrects / total_counts * 100, total_top5_corrects / total_counts * 100))


if __name__ == '__main__':
    arg = parse_args()
    main(arg)