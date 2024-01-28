import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-df', '--dataset-file', required=True)
    # parser.add_argument('-dn', '--dataset-name')
    parser.add_argument('-v', '--visualize', action='store_true', default=False)
    # parser.add_argument('-s', '--save-plot', action='store_true', default=False)
    parser.add_argument('-k', '--neighborhood-size', type=int, default=3)
    parser.add_argument('-a', '--algorithm', default='nbc', help='ncb or dbscan')
    parser.add_argument('-e', '--eps', type=float, default=0.5, help='Maximum distance between two point to consider them as neighbours')
    args = parser.parse_args()
    return args
