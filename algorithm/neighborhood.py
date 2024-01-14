from typing import Optional, TypeAlias, Callable

import numpy as np
import math
import bisect

class Point:
    def __init__(self, coords: np.ndarray, original_idx: int):
        self.coords: np.ndarray = coords
        self.original_idx: int = original_idx
        self.dist: Optional[float] = None
        self.idx = None
        self.eps = None
        self.neighborhood: Optional[Neighborhood] = None
        self.r_neighborhood_len = 0
        self.ndf = None
        self.clst_no = None

    def __str__(self):
        return str(self.idx)


Neighbour: TypeAlias = tuple[Point, float]
Neighborhood: TypeAlias = list[Neighbour]


def distance(first: Point, second: Point):
    return math.dist(first.coords, second.coords)


def insorted(points: Neighborhood, p: Point, dist):
    bisect.insort(points, (p, dist), key=lambda x: x[1])


def eps_dist(k_neighborhood: list[Neighbour]) -> float:
    return max([e[1] for e in k_neighborhood])


def as_points(D: np.ndarray) -> list[Point]:
    return [Point(p, i) for i, p in enumerate(D)]


def sort(points: list[Point], f: Callable[[Point], float]) -> list[Point]:
    return sorted(points, key=f)


def calc_ndf(D: np.ndarray, k: int) -> list[Point]:
    """
    Calculate ndf factor of each point in the input dataset D.

    :param D: input data matrix
    :param k: the number of neighbors to find
    :return: list of { Point } objects
    """
    points = ti_k_neighborhood_index(D, as_points(D), k)
    for p in points:
        p.ndf = p.r_neighborhood_len / len(p.neighborhood)

    return points


def ti_k_neighborhood_index(D: np.ndarray, points: list[Point], k: int) -> list[Point]:
    """

    :param D: input data matrix
    :param points:
    :param k:
    :return:
    """
    r = np.zeros(D.shape[1])

    for p in points:
        p.dist = math.dist(p.coords, r)

    points_sorted = sort(points, lambda p: p.dist)

    for i, p in enumerate(points_sorted):
        p.idx = i

    for p in points_sorted:
        p.neighborhood = ti_k_neighborhood(points_sorted, p, k)
        for neighbor in p.neighborhood:
            neighbor.r_neighborhood_len += 1

    return sort(points_sorted, lambda p: p.original_idx)


def ti_k_neighborhood(D: list[Point], p: Point, k: int) -> list[Point]:
    """
    Returns the k+ nearest neighbours of point p using the Euclidean metric.

    :param D: list of input data points ordered by the distance to the reference point r
              ordered in non-decreasing way
    :param p: point to look the neighborhood for
    :param k: the number of neighbors to find
    :return: list of k+ nearest neighbours of p ordered in non-decreasing way
             with respect to point p
    """
    b = p
    f = p
    backward_search, b = preceding_point(D, b)
    forward_search, f = following_point(D, f)
    k_neighborhood = []
    i = 0
    p, b, f, backward_search, forward_search, k_neighborhood, i = find_first_k_candidate_neighbours_fab(D, p, b, f, backward_search, forward_search, k_neighborhood, k, i)
    p, b, backward_search, k_neighborhood, i = find_first_k_candidate_neighbours_b(D, p, b, backward_search, k_neighborhood, k, i)
    p, f, forward_search, k_neighborhood, i = find_first_k_candidate_neighbours_f(D, p, f, forward_search, k_neighborhood, k, i)
    p.eps = eps_dist(k_neighborhood)
    p, b, backward_search, k_neighborhood = verify_k_candidate_neighbours_b(D, p, b, backward_search, k_neighborhood, k)
    p, f, forward_search, k_neighborhood = verify_k_candidate_neighbours_f(D, p, f, forward_search, k_neighborhood, k)

    return [e[0] for e in k_neighborhood]


def preceding_point(D: list[Point], p: Point):
    if p.idx > 0:
        p = D[p.idx - 1]
        backward_search = True
    else:
        backward_search = False
    return backward_search, p


def following_point(D: list[Point], p: Point):
    if p.idx < len(D) - 1:
        p = D[p.idx + 1]
        forward_search = True
    else:
        forward_search = False
    return forward_search, p


def find_first_k_candidate_neighbours_fab(
        D: list[Point],
        p: Point,
        b: Point,
        f: Point,
        backward_search: bool,
        forward_search: bool,
        k_neighborhood: Neighborhood,
        k: int,
        i: int
):
    while backward_search and forward_search and i < k:
        if p.dist - b.dist < f.dist - p.dist:
            dist = distance(b, p)
            i += 1
            insorted(k_neighborhood, b, dist)
            backward_search, b = preceding_point(D, b)
        else:
            dist = distance(f, p)
            i += 1
            insorted(k_neighborhood, f, dist)
            forward_search, f = following_point(D, f)

    return p, b, f, backward_search, forward_search, k_neighborhood, i


def find_first_k_candidate_neighbours_b(
        D: list[Point],
        p: Point,
        b: Point,
        backward_search: bool,
        k_neighborhood: Neighborhood,
        k: int,
        i: int
):
    while backward_search and i < k:
        dist = distance(b, p)
        i += 1
        insorted(k_neighborhood, b, dist)
        backward_search, b = preceding_point(D, b)
    return p, b, backward_search, k_neighborhood, i


def find_first_k_candidate_neighbours_f(
        D: list[Point],
        p: Point,
        f: Point,
        forward_search: bool,
        k_neighborhood: Neighborhood,
        k: int,
        i: int
):
    while forward_search and i < k:
        dist = distance(f, p)
        i += 1
        insorted(k_neighborhood, f, dist)
        forward_search, f = following_point(D, f)
    return p, f, forward_search, k_neighborhood, i


def verify_k_candidate_neighbours_b(
        D: list[Point],
        p: Point,
        b: Point,
        backward_search: bool,
        k_neighborhood: Neighborhood,
        k: int
):
    while backward_search and (p.dist - b.dist) <= p.eps:
        dist = distance(b, p)
        if dist < p.eps:
            i = len([e for e in k_neighborhood if e[1] == p.eps])
            if len(k_neighborhood) - i >= k - 1:
                k_neighborhood = [e for e in k_neighborhood if e[1] != p.eps]
                insorted(k_neighborhood, b, dist)
                p.eps = eps_dist(k_neighborhood)
            else:
                insorted(k_neighborhood, b, dist)
        elif dist == p.eps:
            insorted(k_neighborhood, b, dist)
        backward_search, b = preceding_point(D, b)
    return p, b, backward_search, k_neighborhood


def verify_k_candidate_neighbours_f(
        D: list[Point],
        p: Point,
        f: Point,
        forward_search: bool,
        k_neighborhood: Neighborhood,
        k: int
):
    while forward_search and (f.dist - p.dist) <= p.eps:
        dist = distance(f, p)
        if dist < p.eps:
            i = len([e for e in k_neighborhood if e[1] == p.eps])
            if len(k_neighborhood) - i >= k - 1:
                k_neighborhood = [e for e in k_neighborhood if e[1] != p.eps]
                insorted(k_neighborhood, f, dist)
                p.eps = eps_dist(k_neighborhood)
            else:
                insorted(k_neighborhood, f, dist)
        elif dist == p.eps:
            insorted(k_neighborhood, f, dist)
        forward_search, f = following_point(D, f)
    return p, f, forward_search, k_neighborhood


