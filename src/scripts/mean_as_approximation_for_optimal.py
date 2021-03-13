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
    {
        "func": "clustered",
        "n": 50,
        "d": 3,
        "n_clusters": 3,
        "noise": 0.06,
        "weights": [0.2, 0.2, 0.6],
        "centers": [[1,0,0], [0,0,1],[0,1,0]],
    },
    {"func": "random", "n": 20, "d": 3},
    {"func": "random", "n": 100, "d": 3},
    {"func": "clustered", "n": 50, "d": 3, "n_clusters": 2, "noise": 0.06},
    {"func": "clustered", "n": 50, "d": 3, "n_clusters": 3, "noise": 0.06},
    {
        "func": "clustered",
        "n": 50,
        "d": 3,
        "n_clusters": 2,
        "noise": 0.06,
        "weights": [0.7, 0.3],
    },
    {
        "func": "clustered",
        "n": 50,
        "d": 3,
        "n_clusters": 3,
        "noise": 0.06,
        "weights": [0.2, 0.2, 0.6],
    },
]

NAME_TO_FUNC = {
    "random": get_d_dimensional_agents_summing_to_1,
    "clustered": get_d_dimensional_agents_summing_to_1_with_clusters,
}


def get_title_from_params(p):
    return "_".join([f"{k}-{v}" for k, v in p.items()])


def main():
    results = []
    for i, p in enumerate(parameters_list):
        f = NAME_TO_FUNC[p["func"]]
        agents = f(**{k: v for k, v in p.items() if k != "func"})
        optimal = opt.budget_constraint_min_d1(agents)
        plot3d(agents, optimal_point=optimal, fname=f"figs/dataset_{i}.html")

        for _ in tqdm(range(N_SAMPLES)):
            agents = f(**{k: v for k, v in p.items() if k != "func"})
            optimal = opt.budget_constraint_min_d1(agents)
            mean = opt.mean(agents)
            this_row = copy.deepcopy(p)
            this_row["dataset_num"] = i
            this_row["title"] = get_title_from_params(p)
            this_row["optimal"] = optimal
            this_row["mean"] = mean
            this_row["optimal_sc"] = social_cost(agents, optimal, metrics.d_1)
            this_row["mean_sc"] = social_cost(agents, mean, metrics.d_1)
            this_row["distortion"] = this_row["mean_sc"] / this_row["optimal_sc"]
            results.append(this_row)

    df = pd.DataFrame(results)
    df.to_csv("results/mean_approx.csv", index=False)
