import numpy as np
from scipy.optimize import linprog


def median_1d(l):
    return np.median(l)


# TODO
def median_kd(l):
    pass


def mean(l):
    return list(np.mean(l, axis=0))


def budget_constraint_min_d1(l):
    """
    l is a list of bliss points
    """
    n = len(l)
    d = len(l[0])

    # Variables are [x0,...,x_{d-1}, z{00}, z{01},...,z{1,d-1}, z{10},...,z{n,d-1}]
    c = np.array([0 for _ in range(d)] + [1 for _ in range(n * d)])
    Aeq = np.array([[1 for _ in range(d)] + [0 for _ in range(n * d)]])
    beq = np.array([1])

    Aub, bub = [], []
    for i in range(n):
        for j in range(d):
            constraint1 = [0 for _ in range(d + n * d)]
            constraint2 = [0 for _ in range(d + n * d)]

            constraint1[j] = 1
            constraint1[d + d * i + j] = -1

            constraint2[j] = -1
            constraint2[d + d * i + j] = -1

            Aub += [constraint1, constraint2]
            bub += [l[i][j], -l[i][j]]

    Aub, bub = np.array(Aub), np.array(bub)

    return list(linprog(c, Aub, bub, Aeq, beq).x[:d])
