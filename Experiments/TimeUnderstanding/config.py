import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', required=True, type=str)
    parser.add_argument('--dataset', default='kinetics', type=str)
    parser.add_argument('--model', default='slow', type=str)
    parser.add_argument('--pre_dataset', default='kinetics_unsupervised', type=str)

    parser.add_argument('--inference', action='store_true', default=False)

    parser.add_argument('--task', default='blackout', type=str)
    parser.add_argument('--n_permute', default=4, type=int)
    parser.add_argument('--n_blackout', default=4, type=int)

    parser.add_argument('--gpu', nargs="+", default=[0], type=int)
    parser.add_argument('--pretrained', action='store_false', default=True)
    parser.add_argument('--freeze', action='store_false', default=True)
    parser.add_argument('--keep_head', action='store_true', default=False)

    parser.add_argument('--bn', default=8, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)

    arg = parser.parse_args()

    return arg