import torch.utils.data

from config import *
from dataset import *
from models import *

import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import wandb


def main(arg):
    # Log with wandb
    if arg.inference and not arg.classification:
        wandb.init(project='Person Segmentation Inference', name=arg.name, config=arg)
    elif not arg.classification:
        wandb.init(project='Person Segmentation', name=arg.name, config=arg)
    # Set device
    device = torch.device('cuda:' + str(arg.gpu[0]) if torch.cuda.is_available() else 'cpu')

    # Setup Dataloader and model
    Dataset = get_dataset(arg.dataset)
    train_dataset = Dataset('train', arg.model, arg.target, arg.classification, arg.blackout)
    train_loader = DataLoader(train_dataset, batch_size=arg.bn, shuffle=True, num_workers=8, pin_memory=True,
                              generator=torch.Generator().manual_seed(42), worker_init_fn=seed_worker)
    test_dataset = Dataset('test', arg.model, arg.target, arg.classification, arg.blackout)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True,
                             generator=torch.Generator().manual_seed(42), worker_init_fn=seed_worker)

    # Initialize model
    Model = get_model(arg.model)
    model = Model(device=device, pre_dataset=arg.pre_dataset).to(device)

    # Make directory
    model_save_dir = 'saved_models/' + arg.name
    if not arg.inference:
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        write_hyperparameters(vars(arg), model_save_dir)

    # Load model for inference
    if arg.inference and not arg.classification:
        model = load_model(model, model_save_dir, device).to(device)

    # Set random seeds
    set_random_seed()

    # Definde loss function
    if arg.classification:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=arg.lr, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    epochs = arg.epochs
    best_loss = 1e12

    if not arg.inference:
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch, arg.epochs - 1))
            print('-' * 10)

            # Train loop
            model.train()
            running_loss = 0.0

            for step, (videos, segmentation_masks, category_label, frame, category_idx) in enumerate(tqdm(train_loader)):
                if arg.model == 'slowfast':
                    videos = [video.to(device) for video in videos]
                else:
                    videos = videos.to(device)
                segmentation_masks = segmentation_masks.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(videos)
                loss = criterion(outputs, segmentation_masks)

                # backward
                loss.backward()
                optimizer.step()

                # statistics
                running_loss += loss.item() * segmentation_masks.size(0)

            scheduler.step()
            epoch_loss = running_loss / len(train_dataset)
            wandb.log({"Train Loss": epoch_loss})
            print('Train Loss: {:.4f}'.format(epoch_loss))

            # Test loop
            model.eval()
            running_loss = 0.0
            frames_to_plot = []
            outputs_to_plot = []
            labels_to_plot = []

            for step, (videos, segmentation_masks, category_labels, frames, category_idxs) in enumerate(tqdm(test_loader)):
                with torch.no_grad():
                    if arg.model == 'slowfast':
                        videos = [video.to(device) for video in videos]
                    else:
                        videos = videos.to(device)
                    segmentation_masks = segmentation_masks.to(device)

                    # forward
                    outputs = model(videos)
                    loss = criterion(outputs, segmentation_masks)

                    # statistics
                    running_loss += loss.item() * segmentation_masks.size(0)

                    # Plot visualizations
                    if epoch % 5 == 0:
                        if step < 9:
                            frames_to_plot.append(torch.cat(frames.unbind()))
                            outputs_to_plot.append(outputs)
                            labels_to_plot.append(segmentation_masks)
                        if step == 9:
                            frames = torch.cat(frames_to_plot, dim=0)
                            outputs = torch.cat(outputs_to_plot, dim=0)
                            labels = torch.cat(labels_to_plot, dim=0)
                            results = make_visualizations(frames, outputs, labels, model_save_dir, 0, False)
                            wandb.log({f"Summary_{epoch}": [wandb.Image(results)]})

            epoch_loss = running_loss / len(test_dataset)
            wandb.log({"Test Loss": epoch_loss})
            print('Test Loss: {:.4f}'.format(epoch_loss))

            # Save model
            if epoch_loss < best_loss:
                save_model(model, model_save_dir)
                best_loss = epoch_loss

    # Make Inference
    else:
        if arg.classification:
            model.eval()
            running_loss = 0.0
            running_corrects = 0
            stats = {}

            for step, (videos, segmentation_masks, category_labels, frames, category_idxs) in enumerate(tqdm(test_loader)):
                with torch.no_grad():
                    if arg.model == 'slowfast':
                        videos = [video.to(device) for video in videos]
                    else:
                        videos = videos.to(device)
                    category_labels = category_labels.to(device)

                    # forward
                    outputs = model(videos)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, category_labels.view(-1))

                    # statistics
                    running_loss += loss.item() * category_labels.size(0)
                    running_corrects += torch.sum(preds == category_labels.data)
                    if test_dataset.category_dict[category_idxs.item()] not in stats:
                        stats[test_dataset.category_dict[category_idxs.item()]] = {"loss": loss.item(),
                                                                                  "corrects": torch.sum(preds == category_labels.data).item(),
                                                                                  "counts": 1}
                    else:
                        stats[test_dataset.category_dict[category_idxs.item()]]["loss"] += loss.item()
                        stats[test_dataset.category_dict[category_idxs.item()]]["corrects"] += torch.sum(preds == category_labels.data).item()
                        stats[test_dataset.category_dict[category_idxs.item()]]["counts"] += 1

            epoch_loss = running_loss / len(test_dataset)
            epoch_acc = running_corrects / len(test_dataset)
            stats = dict(sorted(stats.items(), key=lambda item: item[1]["counts"], reverse=True))
            print(" \n {:<40} {:<10s} {:<10s} {:<20s}".format("Category", "# Videos", "Loss", "Accuracy (%)"))
            print("-" * 80)
            for key in stats.keys():
                print("{:<40} {:<10d} {:<10.2f} {:<10.2f}".format(key, stats[key]["counts"], stats[key]["loss"]/stats[key]["counts"], stats[key]["corrects"]/stats[key]["counts"] * 100))
            print("-" * 70)
            print("{:<20s} {:<10d} {:<10.2f} {:<10.2f}".format(f"Total ({arg.model})", len(test_dataset), epoch_loss, epoch_acc * 100))
        else:
            model.eval()
            running_loss = 0.0
            frames_to_plot = []
            outputs_to_plot = []
            labels_to_plot = []
            stats = {}

            for step, (videos, segmentation_masks, category_labels, frames, category_idxs) in enumerate(tqdm(test_loader)):
                with torch.no_grad():
                    if arg.model == 'slowfast':
                        videos = [video.to(device) for video in videos]
                    else:
                        videos = videos.to(device)
                    segmentation_masks = segmentation_masks.to(device)

                    # forward
                    outputs = model(videos)
                    loss = criterion(outputs, segmentation_masks)

                    # statistics
                    running_loss += loss.item() * segmentation_masks.size(0)
                    if test_dataset.category_dict[category_idxs.item()] not in stats:
                        stats[test_dataset.category_dict[category_idxs.item()]] = {"loss": loss.item(),
                                                                                  "counts": 1}
                    else:
                        stats[test_dataset.category_dict[category_idxs.item()]]["loss"] += loss.item()
                        stats[test_dataset.category_dict[category_idxs.item()]]["counts"] += 1

                    # Plot visualizations
                    if step < 9:
                        frames_to_plot.append(torch.cat(frames.unbind()))
                        outputs_to_plot.append(outputs)
                        labels_to_plot.append(segmentation_masks)
                    if step == 9:
                        frames = torch.cat(frames_to_plot, dim=0)
                        outputs = torch.cat(outputs_to_plot, dim=0)
                        labels = torch.cat(labels_to_plot, dim=0)
                        results = make_visualizations(frames, outputs, labels, model_save_dir, 0, True)
                        wandb.log({f"Summary": [wandb.Image(results)]})

            # epoch_loss = running_loss / len(test_dataset)
            # stats = dict(sorted(stats.items(), key=lambda item: item[1]["counts"], reverse=True))
            # print(" \n {:<20s} {:<10s} {:<10s}".format("Category", "# Videos", "Loss"))
            # print("-" * 40)
            # for key in stats.keys():
            #     print("{:<20s} {:<10d} {:<10.5f}".format(key, stats[key]["counts"], stats[key]["loss"] / stats[key]["counts"]))
            # print("-" * 40)
            # print("{:<20s} {:<10d} {:<10.5f}".format(f"Total ({arg.model})", len(test_dataset), epoch_loss))


if __name__ == '__main__':
    arg = parse_args()
    main(arg)