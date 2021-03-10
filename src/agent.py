import numpy as np
import pandas as pd


def get_one_dimensional_integral_agents(n, max_val=100):
    """Gets a list of n one-dimensional agents

    Args:
        n (int): Number of Agents
        max_val (int, optional): Max value for any agent. Defaults to 100.

    Returns:
        [Int]: List of generated agents
    """
    return [np.random.randint(1, max_val) for _ in range(n)]


def get_d_dimensional_integral_agents(n, max_val=100, d=3):
    """Gets a list of n d-dimensional agents

    Args:
        n (int): Number of Agents
        max_val (int, optional): Max value for any agent. Defaults to 100.
        d (int, optional): Dimension of vector. Defaults to 3.

    Returns:
        [Agents]: List of generated agents
    """
    return [list(np.random.randint(1, max_val, size=d)) for _ in range(n)]


def get_d_dimensional_agents_summing_to_1(n, d=3):
    """Gets a list of n d-dimensional agents

    Args:
        n (int): Number of Agents
        d (int, optional): Dimension of vector. Defaults to 5.

    Returns:
        [List]: List of generated agents
    """
    l = []
    for _ in range(n):
        arr = np.random.rand(d)
        arr = arr / sum(arr)
        l.append(list(arr))
    return l


if __name__ == "__main__":
    print(get_one_dimensional_integral_agents(10, 100))
    print(get_d_dimensional_integral_agents(10, 100, 4))
    print(get_d_dimensional_agents_summing_to_1(10, 3))
