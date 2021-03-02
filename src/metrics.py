"""Metrics output the 'distance' between
two points
"""
import math

def d_1(x, y):
    if hasattr(x, "__iter__") and hasattr(y, "__iter__"):
        return sum(abs(a-b) for a, b in zip(x, y))
    else:
        return abs(x-y)

def euclidean(x, y):
    if hasattr(x, "__iter__") and hasattr(y, "__iter__"):
        return math.sqrt(sum((a-b)**2 for a, b in zip(x, y)))
    else:
        return math.sqrt((x - y)**2)