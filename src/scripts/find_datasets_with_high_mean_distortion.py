import copy

import pandas as pd
from tqdm import tqdm

import src.metrics as metrics
import src.opt as opt
from src.agent import (
    get_d_dimensional_agents_summing_to_1,
    get_d_dimensional_agents_summing_to_1_with_clusters,
)
from src.distortion import distortion, social_cost
from src.plot import plot3d
from src.utils import load_pickle, save_pickle

N_SAMPLES = 1000

parameters_list = [
    # {
    #     "n": 50,
    #     "d": 3,
    #     "n_clusters": 2,
    #     "noise": 0.07,
    #     "weights": [0.7, 0.3],
    # },
    {
        "n": 50,
        "d": 3,
        "n_clusters": 3,
        "noise": 0.07,
        "weights": [0.1, 0.1, 0.8],
        "centers": [[1,0,0], [0,1,0], [0,0,1]],
    },
]

def main():
    results = []
    for i, p in enumerate(parameters_list):
        for _ in range(100):
            agents = get_d_dimensional_agents_summing_to_1_with_clusters(**{k: v for k, v in p.items() if k != "func"})
            optimal = opt.budget_constraint_min_d1(agents)
            dist = distortion(agents, opt.mean(agents), optimal, metrics.d_1)
            print(dist)
            if dist > 1.4:
                print(agents)
                print(optimal)
                print(dist)
                plot3d(agents, optimal_point=optimal, fname=f"figs/high_distortion_agents.html")
                return

