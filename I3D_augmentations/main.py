from dataset import *
from model import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import *


def main(arg):
    # Set random seeds
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Use GPU if possible:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Limit Memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[arg.gpu], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

    # Initialize Dataloader
    dataset = UCFDataset('labels.csv', augmentation=arg.augm, factor=arg.factor)
    dataloader = DataLoader(dataset=dataset, batch_size=1)

    with tf.device(f'/device:GPU:{arg.gpu}'):
        # Initialize model
        model = I3D(labels=dataset.kinetics_labels)

        top_1_predictions = []
        top_3_predictions = []
        top_5_predictions = []

        for step, (video, category, labels) in enumerate(tqdm(dataloader)):
            top_1_bool, top_3_bool, top_5_bool = model.predict(video, labels)
            top_1_predictions.append(top_1_bool)
            top_3_predictions.append(top_3_bool)
            top_5_predictions.append(top_5_bool)

        top_1_acc = sum(top_1_predictions) / len(top_1_predictions) * 100
        top_3_acc = sum(top_3_predictions) / len(top_3_predictions) * 100
        top_5_acc = sum(top_5_predictions) / len(top_5_predictions) * 100

        if arg.augm[0] in ['brightness', 'saturation', 'hue']:
            print(f'\n Results for {len(top_1_predictions)} videos with augmentations {arg.augm} ({arg.factor}): \n')
        else:
            print(f'\n Results for {len(top_1_predictions)} videos with augmentations {arg.augm}: \n')
        print(f'Top 1 accuracy: {top_1_acc} %')
        print(f'Top 3 accuracy: {top_3_acc} %')
        print(f'Top 5 accuracy: {top_5_acc} %')


if __name__ == '__main__':
    arg = parse_args()
    main(arg)
