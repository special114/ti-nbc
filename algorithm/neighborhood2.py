import numpy as np
import math
import bisect


class Point:
    def __init__(self, coords, dist):
        self.coords = coords
        self.dist = dist
        self.idx = None
        self.eps = None
        self.neighborhood = None
        self.r_neighborhood_len = 0
        self.ndf = None
        self.clst_no = None

    def __str__(self):
        return str(self.idx)


def distance(first: Point, second: Point):
    return math.dist(first.coords, second.coords)


def insorted(points: list, p: Point, dist):
    bisect.insort(points, (p, dist), key=lambda x: x[1])


def eps_dist(k_neighborhood):
    return max([e[1] for e in k_neighborhood])


def calc_ndf(D, k):
    points = ti_k_neighborhood_index(D, k)
    for p in points:
        p.ndf = p.r_neighborhood_len / len(p.neighborhood)

    return points



def ti_k_neighborhood_index(D, k):
    r = np.zeros(D.shape[1])
    dist_func = lambda p: math.dist(p, r)
    distances = np.array([Point(p, dist_func(p)) for p in D])
    sorted_distances = sorted(distances, key=lambda p: p.dist)
    new_D = np.array(sorted_distances)

    for i, p in enumerate(new_D):
        p.idx = i

    for p in new_D:
        p.neighborhood = ti_k_neighborhood(new_D, p, k)
        for neighbor in p.neighborhood:
            neighbor.r_neighborhood_len += 1

    return new_D


def ti_k_neighborhood(D, p, k):
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


def preceding_point(D, p: Point):
    if p.idx > 0:
        p = D[p.idx - 1]
        backward_search = True
    else:
        backward_search = False
    return backward_search, p


def following_point(D, p: Point):
    if p.idx < len(D) - 1:
        p = D[p.idx + 1]
        forward_search = True
    else:
        forward_search = False
    return forward_search, p


def find_first_k_candidate_neighbours_fab(D, p, b, f, backward_search, forward_search, k_neighborhood, k, i):
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


def find_first_k_candidate_neighbours_b(D, p, b, backward_search, k_neighborhood, k, i):
    while backward_search and i < k:
        dist = distance(b, p)
        i += 1
        insorted(k_neighborhood, b, dist)
        backward_search, b = preceding_point(D, b)
    return p, b, backward_search, k_neighborhood, i


def find_first_k_candidate_neighbours_f(D, p, f, forward_search, k_neighborhood, k, i):
    while forward_search and i < k:
        dist = distance(f, p)
        i += 1
        insorted(k_neighborhood, f, dist)
        forward_search, f = following_point(D, f)
    return p, f, forward_search, k_neighborhood, i


def verify_k_candidate_neighbours_b(D, p, b, backward_search, k_neighborhood, k):
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


def verify_k_candidate_neighbours_f(D, p, f, forward_search, k_neighborhood, k):
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


