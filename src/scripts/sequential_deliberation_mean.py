from scipy import optimize
from src.distortion import social_cost, distortion
from src.sequential import (
    sequential_deliberation,
    sequential_deliberation_list,
    sequential_deliberation_many_times
)
import src.bargain as bargain
import src.metrics as metrics
import src.opt as opt
from src.plot import plot3d
from src.agent import get_d_dimensional_agents_summing_to_1, get_d_dimensional_agents_summing_to_1_with_clusters
import numpy as np

NUM_AGENTS = 100
MAX_VAL = 10000
NUM_ITERS = 9
NUM_SAMPLES = 100
DIM = 6

def sequential_deliberation_once():
    agents = get_d_dimensional_agents_summing_to_1_with_clusters(NUM_AGENTS, d=DIM, noise = 0.04, n_clusters=2, weights=[0.2, 0.8])

    optimal = opt.budget_constraint_min_d1(agents)

    a_T = sequential_deliberation(agents, bargain.mean, metrics.d_1, T=NUM_ITERS)

    sc_opt = social_cost(agents, optimal, metrics.d_1)
    sc_mean = social_cost(agents, opt.mean(agents), metrics.d_1)
    sc_seq = social_cost(agents, a_T , metrics.d_1)

    print("Social cost of the optimal: " , sc_opt)
    print("Social cost of the mean: " , sc_mean)
    print("Social cost of the sequential mean: " , sc_seq)

    plot3d(agents, found_point=a_T, optimal_point=optimal) 


def main():
    agents = get_d_dimensional_agents_summing_to_1_with_clusters(NUM_AGENTS, d=DIM, noise = 0.04, n_clusters=2, weights=[0.05, 0.95])

    optimal = opt.budget_constraint_min_d1(agents)
    plot3d(agents, optimal_point=optimal)

    results = sequential_deliberation_many_times(
        agents, bargain.mean, metrics.d_1, NUM_ITERS, NUM_SAMPLES
    )

    plot3d(results, optimal_point=optimal, fname='figs/result.html')
    expected_dist = np.mean([distortion(agents, r, optimal,metrics.d_1) for r in results])

    print(f"Expected distortion over {NUM_SAMPLES} samples: {expected_dist}")
