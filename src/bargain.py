import numpy as np

def median_1d(u, v, alt, metric):
    m =  sorted([u, v, alt])[1]
    if metric(u, m) <= metric(u, alt) and metric(v, m) <= metric(v, alt):
        return m
    else: 
        return alt