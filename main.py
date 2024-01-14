from sklearn.metrics import adjusted_rand_score, v_measure_score

from algorithm.nbc import TI_NBC
from util.args import parse_args
from util.util import read_arff
from visualization.visualize import visualize


def score(y_true, y_pred, ds_name = None):
    s = adjusted_rand_score(y_true, y_pred)
    print(f'Adjusted rand index for dataset {ds_name if ds_name is not None else ""} is', s)
    s = v_measure_score(y_true, y_pred)
    print(f'V-measure score for dataset {ds_name if ds_name is not None else ""} is', s)
    return s


def process_dataset(path, name, save_plot, k):
    X, Y = read_arff(path)
    y_pred = TI_NBC(k).predict(X)
    score(Y, y_pred, name)
    visualize(X, Y, y_pred, save_plot)


def fit_predict_score():
    file, dataset_name, save_plot, k = parse_args()
    dataset_name = dataset_name if dataset_name is not None else 'unknown'
    process_dataset(file, dataset_name, save_plot, k)


if __name__ == '__main__':
    fit_predict_score()
