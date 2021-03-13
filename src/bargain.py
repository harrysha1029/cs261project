import numpy as np


def median_1d(u, v, alt, metric):
    m = sorted([u, v, alt])[1]  # Finds the median
    if metric(u, m) <= metric(u, alt) and metric(v, m) <= metric(v, alt):
        return m
    else:  # Disagreement
        return alt


def mean(u, v, alt, metric):
    m = np.mean(np.array([u, v, alt]), axis=0)
    if metric(u, m) <= metric(u, alt) and metric(v, m) <= metric(v, alt):
        return m
    else:  # Disagreement
        return alt
