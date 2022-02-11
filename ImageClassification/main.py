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
    wandb.init(project='ImageClassification', config=arg)

    # Set device
    device = torch.device('cuda:' + str(arg.gpu[0]) if torch.cuda.is_available() else 'cpu')

    # Setup Dataloadear
    eval_bn = 1 if arg.avg else arg.bn
    Dataset = get_dataset(arg.dataset)
    train_dataset = Dataset(mode='train', augm=arg.augm, avg=False)
    train_loader = DataLoader(train_dataset, batch_size=arg.bn, shuffle=True, num_workers=8)
    test_dataset = Dataset(mode='test', augm=arg.augm, avg=arg.avg)
    test_loader = DataLoader(test_dataset, batch_size=eval_bn, shuffle=True, num_workers=8)

    # Initialize model
    model = Model(num_classes=train_dataset.num_classes, pretrained=arg.pretrained, freeze=arg.freeze).to(device)

    # Definde loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=arg.lr, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for epoch in range(arg.epochs):
        print('Epoch {}/{}'.format(epoch, arg.epochs - 1))
        print('-' * 10)

        # Train loop
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for step, (frames, labels) in enumerate(tqdm(train_loader)):
            if torch.min(frames) == -1:
                print(f"None returned at item: {labels}")
                continue

            frames = frames.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(frames)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels.view(-1))

            # backward
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * frames.size(0)
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

        for step, (frames, labels) in enumerate(tqdm(test_loader)):
            with torch.no_grad():
                if torch.min(frames) == -1:
                    print(f"None returned at item: {labels}")
                    continue

                frames = frames.to(device)
                labels = labels.to(device)

                if arg.avg:
                    frames = frames.squeeze(0)

                # forward
                outputs = model(frames)
                if arg.avg:
                    outputs = torch.mean(outputs, dim=0, keepdim=True)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels.view(-1))

                # statistics
                if arg.avg:
                    running_loss += loss.item()
                else:
                    running_loss += loss.item() * frames.size(0)
                running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(test_dataset)
        epoch_acc = running_corrects / len(test_dataset)
        wandb.log({"Test Loss": epoch_loss})
        wandb.log({"Test Accuracy": epoch_acc})

        print('Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))


if __name__ == '__main__':
    arg = parse_args()
    main(arg)