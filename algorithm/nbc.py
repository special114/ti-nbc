import numpy as np

from algorithm.neighborhood import calc_ndf, Point


class TI_NBC:
    """
    A Neighborhood-Based Clustering by Means of the Triangle Inequality is a density-based clustering algorithm
    which discovers clusters based on neighborhood characteristics of data. This implementation takes advantage of
    using triangle inequality for determining k-neighbourhoods of all data point in input data.

    :param k: the number of neighbors of each data point to find. It determines the minimal size of the cluster.
    """
    def __init__(self, k: int):
        self.k = k


    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Performs the clustering and returns the predicted clusters for each point.

        :param X: input data, a matrix of flaots
        :return: a vector of predicted cluster number for each input point
        """
        points: list[Point] = calc_ndf(X, self.k)
        for p in points:
            p.clst_no = -1

        cluster_count = 0

        for p in points:
            if p.clst_no != -1 or p.ndf < 1:
                continue

            # setting the same cluster number for the point and all its neighbours
            p.clst_no = cluster_count
            dp_set = list()
            for q in p.neighborhood:
                q.clst_no = cluster_count
                if q.ndf >= 1:
                    dp_set.append(q)

            # expanding cluster
            while len(dp_set) > 0:
                p = dp_set.pop(0)
                for q in p.neighborhood:
                    if q.clst_no != -1:
                        continue
                    q.clst_no = cluster_count
                    if q.ndf >= 1:
                        dp_set.append(q)
            cluster_count += 1

        # print("max cluster" + str(cluster_count -1 ))
        # for p in points:
        #     if p.clst_no == -1:
        #         p.clst_no = cluster_count
        #         cluster_count += 1
        #     print(p.clst_no)

        return np.array([p.clst_no for p in points])
