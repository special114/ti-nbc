import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-df', '--dataset-file', required=True)
    parser.add_argument('-dn', '--dataset-name')
    parser.add_argument('-s', '--save-plot', action='store_true')
    parser.add_argument('-k', '--neighborhood-size', type=int, default=3)
    args = parser.parse_args()
    return args.dataset_file, args.dataset_name, args.save_plot, args.neighborhood_size
