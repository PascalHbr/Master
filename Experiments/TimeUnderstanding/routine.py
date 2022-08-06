import torch.utils.data

from config import *
from dataset import *
from models import *

import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import wandb


def main(arg):
    for task in []:
        lr = 1e-3 if arg.model == "mae" else 1e-2
        # Set random seeds
        set_random_seed()

        # Log with wandb
        if not arg.inference:
            wandb.init(project=task, name=arg.name, config=arg, reinit=True)

        # Set device
        device = torch.device('cuda:' + str(arg.gpu[0]) if torch.cuda.is_available() else 'cpu')

        # Setup Dataloader and model
        Dataset = get_dataset(arg.dataset)
        if not arg.inference:
            train_dataset = Dataset('train', task, arg.model, arg.n_permute, arg.n_blackout)
            if arg.dataset == 'ssv2':
                train_loader = DataLoader(train_dataset, batch_size=arg.bn, shuffle=False, num_workers=8, pin_memory=True,
                                          sampler=RandomSampler(train_dataset, num_samples=10000))
            else:
                train_loader = DataLoader(train_dataset, batch_size=arg.bn, shuffle=True, num_workers=8, pin_memory=True)
        test_dataset = Dataset('test', task, arg.model, arg.n_permute, arg.n_blackout)
        if arg.dataset == 'ssv2':
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True,
                                     sampler=RandomSampler(test_dataset, num_samples=2000))
        else:
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

        # Initialize model
        Model = get_model(arg.model)
        model = Model(num_classes=test_dataset.num_classes, pretrained=arg.pretrained, freeze=arg.freeze,
                      keep_head=arg.keep_head, device=device, pre_dataset=arg.pre_dataset).to(device)

        # Definde loss function
        # criterion = nn.BCEWithLogitsLoss()
        criterion = nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        epochs = arg.epochs

        if not arg.inference:
            for epoch in range(epochs):
                print('Epoch {}/{}'.format(epoch, arg.epochs - 1))
                print('-' * 10)

                # Train loop
                model.train()
                num_samples = 0
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
                    num_samples += labels.shape[0]
                    running_loss += loss.item() * labels.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                scheduler.step()
                epoch_loss = running_loss / num_samples
                epoch_acc = running_corrects / num_samples
                wandb.log({"Train Loss": epoch_loss})
                wandb.log({"Train Accuracy": epoch_acc})
                print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

                # Test loop
                model.eval()
                num_samples = 0
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
                        num_samples += labels.shape[0]
                        running_loss += loss.item() * labels.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / num_samples
                epoch_acc = running_corrects / num_samples
                wandb.log({"Test Loss": epoch_loss})
                wandb.log({"Test Accuracy": epoch_acc})
                print('Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    for n_permute in [3]:
        lr = 1e-3 if arg.model == "mae" else 1e-2
        task = 'permutation'
        # Set random seeds
        set_random_seed()

        # Log with wandb
        if not arg.inference:
            wandb.init(project=task, name=f'{arg.name}_{n_permute}', config=arg, reinit=True)

        # Set device
        device = torch.device('cuda:' + str(arg.gpu[0]) if torch.cuda.is_available() else 'cpu')

        # Setup Dataloader and model
        Dataset = get_dataset(arg.dataset)
        if not arg.inference:
            train_dataset = Dataset('train', task, arg.model, n_permute, arg.n_blackout)
            if arg.dataset == 'ssv2':
                train_loader = DataLoader(train_dataset, batch_size=arg.bn, shuffle=False, num_workers=8,
                                          pin_memory=True,
                                          sampler=RandomSampler(train_dataset, num_samples=10000))
            else:
                train_loader = DataLoader(train_dataset, batch_size=arg.bn, shuffle=True, num_workers=8,
                                          pin_memory=True)
        test_dataset = Dataset('test', task, arg.model, n_permute, arg.n_blackout)
        if arg.dataset == 'ssv2':
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True,
                                     sampler=RandomSampler(test_dataset, num_samples=2000))
        else:
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

        # Initialize model
        Model = get_model(arg.model)
        model = Model(num_classes=test_dataset.num_classes, pretrained=arg.pretrained, freeze=arg.freeze,
                      keep_head=arg.keep_head, device=device, pre_dataset=arg.pre_dataset).to(device)

        # Definde loss function
        # criterion = nn.BCEWithLogitsLoss()
        criterion = nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        epochs = arg.epochs

        if not arg.inference:
            for epoch in range(epochs):
                print('Epoch {}/{}'.format(epoch, arg.epochs - 1))
                print('-' * 10)

                # Train loop
                model.train()
                num_samples = 0
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
                    num_samples += labels.shape[0]
                    running_loss += loss.item() * labels.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                scheduler.step()
                epoch_loss = running_loss / num_samples
                epoch_acc = running_corrects / num_samples
                wandb.log({"Train Loss": epoch_loss})
                wandb.log({"Train Accuracy": epoch_acc})
                print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

                # Test loop
                model.eval()
                num_samples = 0
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
                        num_samples += labels.shape[0]
                        running_loss += loss.item() * labels.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / num_samples
                epoch_acc = running_corrects / num_samples
                wandb.log({"Test Loss": epoch_loss})
                wandb.log({"Test Accuracy": epoch_acc})
                print('Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))


if __name__ == '__main__':
    arg = parse_args()
    main(arg)