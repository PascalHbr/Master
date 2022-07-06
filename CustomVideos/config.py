import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='custom', type=str)
    parser.add_argument('--model', default='slow', type=str)
    parser.add_argument('--gpu', nargs="+", default=[0], type=int)

    parser.add_argument('--labels', action='store_true', default=False)
    parser.add_argument('--noise', action='store_true', default=False)
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--stats', action='store_true', default=False)

    arg = parser.parse_args()

    return arg