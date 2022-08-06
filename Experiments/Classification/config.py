import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='ucf', type=str)
    parser.add_argument('--model', default='slow', type=str)
    parser.add_argument('--pre_dataset', default='kinetics', type=str)
    parser.add_argument('--gpu', nargs="+", default=[0], type=int)

    parser.add_argument('--augm', type=str)
    parser.add_argument('--n_augm', default=5, type=int)
    parser.add_argument('--app_augm', type=str)

    parser.add_argument('--debug', action='store_true', default=False)

    arg = parser.parse_args()

    return arg