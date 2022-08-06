import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='slowfast', type=str)
    parser.add_argument('--dataset', default='ucf', type=str)
    parser.add_argument('--name', required=True, type=str)
    parser.add_argument('--cfk', action='store_true', default=False)
    parser.add_argument('--plot', action='store_true', default=False)

    arg = parser.parse_args()

    return arg


def write_hyperparameters(r, save_dir):
    filename = save_dir + "/config.txt"
    with open(filename, "w") as input_file:
        for k, v in r.items():
            line = '{}, {}'.format(k, v)
            print(line, file=input_file)