import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', required=True, type=str)
    parser.add_argument('--dataset', default='ucf', type=str)
    parser.add_argument('--model', default='slowfast', type=str)
    parser.add_argument('--pre_dataset', default='kinetics', type=str)

    parser.add_argument('--inference', action='store_true', default=False)
    parser.add_argument('--classification', action='store_true', default=False)
    parser.add_argument('--blackout', default=None, type=str)

    parser.add_argument('--gpu', nargs="+", default=[0], type=int)
    parser.add_argument('--pretrained', action='store_false', default=True)
    parser.add_argument('--freeze', action='store_false', default=True)
    parser.add_argument('--keep_head', action='store_true', default=False)  # TODO add this
    parser.add_argument('--target', type=str, default="first")

    parser.add_argument('--bn', default=8, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.01, type=float)

    arg = parser.parse_args()

    return arg


def write_hyperparameters(r, save_dir):
    filename = save_dir + "/config.txt"
    with open(filename, "w") as input_file:
        for k, v in r.items():
            line = '{}, {}'.format(k, v)
            print(line, file=input_file)