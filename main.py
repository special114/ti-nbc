from sklearn.metrics import adjusted_rand_score, v_measure_score

from algorithm.get_algorithm import get_algorithm

from util.args import parse_args
from util.util import read_arff
from visualization.visualize import visualize

import time


def score(y_true, y_pred, args):
    s = adjusted_rand_score(y_true, y_pred)
    print(f'Adjusted rand index for dataset {args.dataset_name if args.dataset_name is not None else ""} is', s)
    s = v_measure_score(y_true, y_pred)
    print(f'V-measure score for dataset {args.dataset_name if args.dataset_name is not None else ""} is', s)
    return s


def predict(X, algorithm):
    start = time.time()
    Y = algorithm.fit_predict(X)
    stop = time.time()
    print("Time elapsed: ", stop - start)
    return Y


def process_dataset(algorithm, args):
    X, Y = read_arff(args.dataset_file)
    y_pred = predict(X, algorithm)
    score(Y, y_pred, args)
    visualize(X, Y, y_pred, args)


def fit_predict_score(args):
    algorithm = get_algorithm(args)
    process_dataset(algorithm, args)


if __name__ == '__main__':
    args = parse_args()
    fit_predict_score(args)
