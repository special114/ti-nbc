from sklearn.cluster import DBSCAN

from algorithm.nbc import TI_NBC


def get_algorithm(args):
    if args.algorithm == 'dbscan':
        return DBSCAN(min_samples=args.neighborhood_size, eps=args.eps)
    else:
        return TI_NBC(k=args.neighborhood_size)