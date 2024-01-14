import arff, numpy as np
import pandas as pd


def read_arff(path: str) -> tuple[np.ndarray, np.ndarray]:
    dataset = arff.load(open(path, 'r'))
    data = np.array(dataset['data'], dtype='str')
    labels = pd.get_dummies(data[:, -1]).values.argmax(1)

    return np.array(data[:, :-1], dtype='float64'), labels
