import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='kinetics', type=str)

    parser.add_argument('--gpu', nargs="+", default=[0], type=int)
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--freeze', action='store_true', default=False)
    parser.add_argument('--augm', action='store_true', default=False)
    parser.add_argument('--middle', action='store_true', default=False)
    parser.add_argument('--avg', action='store_true', default=False)

    parser.add_argument('--bn', default=32, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=0.001, type=float)

    arg = parser.parse_args()

    return arg