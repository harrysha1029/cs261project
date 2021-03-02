from src.distortion import social_cost, distortion
from src.sequential import sequential_deliberation, sequential_deliberation_list, estimate_expected_distortion
import src.bargain as bargain 
import src.metrics as metrics
import src.opt as opt
from src.agent import Agent, get_one_dimensional_agents, extract_bliss_points
import numpy as np

NUM_AGENTS = 10
MAX_VAL = 1000
NUM_ITERS = 10
NUM_SAMPLES = 100
DIM = 8

agents = get_one_dimensional_agents(NUM_AGENTS, MAX_VAL)
opt = opt.median_1d(extract_bliss_points(agents))

a_T = sequential_deliberation(
    agents, bargain.median_1d, metrics.d_1, T=NUM_ITERS
)

dist = distortion(agents, a_T, opt, metrics.d_1)

print(f"Optimal: {opt}")
print(f"a_T: {a_T}")
print(f"distortion: {dist}")

expected_dist = estimate_expected_distortion(
    agents, bargain.median_1d, metrics.d_1, opt,
    NUM_ITERS, NUM_SAMPLES
)

print(f"Expected distortion over {NUM_SAMPLES} samples: {expected_dist}")