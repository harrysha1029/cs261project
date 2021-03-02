import numpy as np
import pandas as pd

class Agent:
    def __repr__(self):
        return f"Agent: {self.bliss}"

    def __init__(self, bliss):
        self.bliss = bliss

def extract_bliss_points(agents):
    """Given a list of agents, returns 
    a list of their bliss points
    """
    return [a.bliss for a in agents]

def get_one_dimensional_agents(n, max_val=100):
    """Gets a list of n one-dimensional agents

    Args:
        n (int): Number of Agents
        max_val (int, optional): Max value for any agent. Defaults to 100.

    Returns:
        [List]: List of generated agents
    """
    return [Agent(np.random.randint(1, max_val)) for _ in range(n)]

def get_d_dimensional_agents(n, max_val=100, d=5):
    """Gets a list of n d-dimensional agents

    Args:
        n (int): Number of Agents
        max_val (int, optional): Max value for any agent. Defaults to 100.
        d (int, optional): Dimension of vector. Defaults to 5.

    Returns:
        [List]: List of generated agents
    """
    return [Agent(np.random.randint(1, max_val, size=d)) for _ in range(n)] 

def get_d_dimensional_agents_summing_to_1(n, max_val=100, d=5):
    """Gets a list of n d-dimensional agents

    Args:
        n (int): Number of Agents
        max_val (int, optional): Max value for any agent. Defaults to 100.
        d (int, optional): Dimension of vector. Defaults to 5.

    Returns:
        [List]: List of generated agents
    """
    l = []
    for _ in range(n):
        arr = np.random.randint(1, max_val, size=d)
        arr = arr/sum(arr)
        l.append(Agent(arr))
    return l

if __name__ == "__main__":
    print(get_one_dimensional_agents(10,100))
    print(get_d_dimensional_agents(10,100,4))
    print(get_d_dimensional_agents_summing_to_1(10,100,4))