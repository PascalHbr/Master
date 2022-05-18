import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='ucf', type=str)
    parser.add_argument('--model', default='slowfast', type=str)
    parser.add_argument('--gpu', nargs="+", default=[0], type=int)

    arg = parser.parse_args()

    return arg