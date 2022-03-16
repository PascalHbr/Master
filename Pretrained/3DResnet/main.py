import torch.utils.data

from config import *
from dataset import *
from model import *

import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import wandb


def main(arg):
    # Set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    rng = np.random.RandomState(42)

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

    # Initialize model
    Model = get_model(arg.model)
    model = Model(num_classes=test_dataset.num_classes, pretrained=arg.pretrained, freeze=arg.freeze,
                  keep_head=arg.keep_head, device=device).to(device)
    if arg.inference and arg.dataset == 'kinetics':
        test_dataset = torch.utils.data.Subset(test_dataset, range(0, len(test_dataset)//10))
    test_loader = DataLoader(test_dataset, batch_size=arg.bn, shuffle=True, num_workers=8, pin_memory=True)

    # Definde loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=arg.lr, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    epochs = arg.epochs if not arg.inference else 1

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, arg.epochs - 1))
        print('-' * 10)

        if not arg.inference:
            # Train loop
            model.train()
            running_loss = 0.0
            running_corrects = 0

            for step, (videos, labels) in enumerate(tqdm(train_loader)):
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

        for step, (videos, labels) in enumerate(tqdm(test_loader)):
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
        if not arg.inference:
            wandb.log({"Test Loss": epoch_loss})
            wandb.log({"Test Accuracy": epoch_acc})

        print('Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))


if __name__ == '__main__':
    arg = parse_args()
    main(arg)