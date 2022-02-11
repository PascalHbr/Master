import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # Setup
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--augm', nargs='+', default=[None])
    parser.add_argument('--factor', type=float, default=0.)

    arg = parser.parse_args()

    return arg