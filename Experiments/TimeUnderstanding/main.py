import torch.utils.data

from config import *
from dataset import *
from models import *

import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import wandb


def main(arg):
    # Set random seeds
    set_random_seed()

    # Log with wandb
    if not arg.inference:
        wandb.init(project=arg.task, name=arg.name, config=arg)

    # Set device
    device = torch.device('cuda:' + str(arg.gpu[0]) if torch.cuda.is_available() else 'cpu')

    # Setup Dataloader and model
    Dataset = get_dataset(arg.dataset)
    if not arg.inference:
        train_dataset = Dataset('train', arg.task, arg.model, arg.n_permute, arg.n_blackout)
        train_loader = DataLoader(train_dataset, batch_size=arg.bn, shuffle=True, num_workers=8, pin_memory=True)
    test_dataset = Dataset('test', arg.task, arg.model, arg.n_permute, arg.n_blackout)
    if arg.dataset == 'kinetics':
        test_dataset_subset = torch.utils.data.Subset(test_dataset, range(0, len(test_dataset) // 10))
        test_loader = DataLoader(test_dataset_subset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    else:
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

    # Initialize model
    Model = get_model(arg.model)
    model = Model(num_classes=test_dataset.num_classes, pretrained=arg.pretrained, freeze=arg.freeze,
                  keep_head=arg.keep_head, device=device, pre_dataset=arg.pre_dataset).to(device)

    # Definde loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=arg.lr, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    epochs = arg.epochs

    if not arg.inference:
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch, arg.epochs - 1))
            print('-' * 10)

            # Train loop
            model.train()
            running_loss = 0.0
            running_corrects = 0

            for step, (videos, category_idx, labels) in enumerate(tqdm(train_loader)):
                if arg.model == 'slowfast':
                    videos = [video.to(device) for video in videos]
                else:
                    videos = videos.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(videos)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels.view(-1))

                # backward
                loss.backward()
                optimizer.step()

                # statistics
                running_loss += loss.item() * labels.size(0)
                running_corrects += torch.sum(preds == labels.data)

            scheduler.step()
            epoch_loss = running_loss / len(train_dataset)
            epoch_acc = running_corrects / len(train_dataset)
            wandb.log({"Train Loss": epoch_loss})
            wandb.log({"Train Accuracy": epoch_acc})
            print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

            # Test loop
            model.eval()
            running_loss = 0.0
            running_corrects = 0

            for step, (videos, category_idx, labels) in enumerate(tqdm(test_loader)):
                with torch.no_grad():
                    if arg.model == 'slowfast':
                        videos = [video.to(device) for video in videos]
                    else:
                        videos = videos.to(device)
                    labels = labels.to(device)

                    # forward
                    outputs = model(videos)
                    # outputs = torch.mean(nn.Softmax(dim=1)(outputs), dim=0, keepdim=True)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels.view(-1))

                    # statistics
                    running_loss += loss.item() * labels.size(0)
                    running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(test_dataset)
            epoch_acc = running_corrects / len(test_dataset)
            wandb.log({"Test Loss": epoch_loss})
            wandb.log({"Test Accuracy": epoch_acc})
            print('Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    else:
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