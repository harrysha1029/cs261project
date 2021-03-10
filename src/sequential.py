import random
import numpy as np
from src.distortion import distortion


def sequential_deliberation(agents, bargain, metric, T=10):
    """Runs the algorithm from Figure 1 in the paper

    Args:
        agents ([Agent]): List of agents
        bargain (Function): A function that does the bargaining
        metric (Function): A function that takes in two things and returns their distance
        T (int, optional): Number of iterations. Defaults to 10.

    Returns:
        The result of sequential deliberation
    """
    a_t = random.choice(agents)
    for _ in range(T):
        u = random.choice(agents)
        v = random.choice(agents)
        a_t = bargain(u, v, a_t, metric)
    return a_t


def sequential_deliberation_list(agents, bargain, metric, T=10):
    """
    Same as above except it returns a list of solutions at each iteration.
    """
    a_t = random.choice(agents)
    l = [a_t]
    for _ in range(T):
        u = random.choice(agents)
        v = random.choice(agents)
        a_t = bargain(u, v, a_t, metric)
        l.append(a_t)
    return l


def estimate_expected_distortion(agents, bargain, metric, opt, T=10, num_samples=100):
    l = []
    for _ in range(num_samples):
        a_T = sequential_deliberation(agents, bargain, metric, T=10)
        dist = distortion(agents, a_T, opt, metric)
        l.append(dist)
    return np.mean(l)
