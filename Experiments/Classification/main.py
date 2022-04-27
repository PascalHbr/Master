import torch.utils.data

from config import *
from dataset import *
from model import *
from tqdm import tqdm


def main(arg):
    # Set random seed
    set_random_seed()

    # Set device
    device = torch.device('cuda:' + str(arg.gpu[0]) if torch.cuda.is_available() else 'cpu')

    # Setup Dataloader and model
    Dataset = get_dataset(arg.dataset)
    test_dataset = Dataset('test', arg.model)
    if arg.dataset == 'kinetics':
        # test_dataset_subset = torch.utils.data.Subset(test_dataset, range(0, len(test_dataset) // 5))
        # test_loader = DataLoader(test_dataset_subset, batch_size=1, shuffle=True, num_workers=8)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=8)
    else:
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=8)

    # Initialize model
    Model = get_model(arg.model)
    model = Model(num_classes=test_dataset.num_classes, pretrained=True, freeze=True,
                  keep_head=True, device=device).to(device)

    # Definde loss function
    criterion = nn.CrossEntropyLoss()

    # Test loop
    model.eval()
    total_loss = 0.0
    stats = {}
    total_corrects = 0
    total_counts = 0

    for step, (video, category_idx, label) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            if arg.model == 'slowfast':
                video = [vid.to(device) for vid in video]
            else:
                video = video.to(device)
            label = label.to(device)

            # forward
            output = model(video)
            _, pred = torch.max(output, 1)
            loss = criterion(output, label.view(-1))

            # statistics
            total_loss += loss.item() * label.size(0)
            total_corrects += torch.sum(pred == label.data)
            total_counts += 1
            if test_dataset.idx_to_category[category_idx.item()] not in stats:
                stats[test_dataset.idx_to_category[category_idx.item()]] = {"loss": loss.item(),
                                                                          "corrects": torch.sum(pred == label.data).item(),
                                                                          "counts": 1}
            else:
                stats[test_dataset.idx_to_category[category_idx.item()]]["loss"] += loss.item()
                stats[test_dataset.idx_to_category[category_idx.item()]]["corrects"] += torch.sum(pred == label.data).item()
                stats[test_dataset.idx_to_category[category_idx.item()]]["counts"] += 1

    # Print stats
    for category in stats.keys():
        stats[category]["accuracy"] = stats[category]["corrects"] / stats[category]["counts"] * 100
    stats = dict(sorted(stats.items(), key=lambda item: item[1]["accuracy"], reverse=True))
    print(" \n {:<20s} {:<10s} {:<10s} {:<20s}".format("Category", "# Videos", "Loss", "Accuracy (%)"))
    print("-" * 60)
    for key in stats.keys():
        print("{:<20s} {:<10d} {:<10.2f} {:<10.2f}".format(key, stats[key]["counts"],
                                                           stats[key]["loss"] / stats[key]["counts"],
                                                           stats[key]["accuracy"]))
    print("-" * 50)
    print("{:<20s} {:<10d} {:<10.2f} {:<10.2f}".format("Total", total_counts, total_loss / total_counts,
                                                       total_corrects / total_counts * 100))


if __name__ == '__main__':
    arg = parse_args()
    main(arg)