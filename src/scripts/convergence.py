from scipy import optimize
from src.distortion import social_cost, distortion
from src.sequential import (
    sequential_deliberation,
    sequential_deliberation_list,
    sequential_deliberation_many_times
)
import plotly.express as px
import src.bargain as bargain
import src.metrics as metrics
import src.opt as opt
from src.plot import plot3d
from src.agent import get_d_dimensional_agents_summing_to_1, get_d_dimensional_agents_summing_to_1_with_clusters
import numpy as np

NUM_AGENTS = 100
MAX_VAL = 10000
NUM_ITERS = 100
NUM_SAMPLES = 100
DIM = 6

def main():
    agents = get_d_dimensional_agents_summing_to_1_with_clusters(NUM_AGENTS, d=DIM, noise = 0.04, n_clusters=2, weights=[0.2, 0.8])

    optimal = opt.budget_constraint_min_d1(agents)

    intermediates = sequential_deliberation_list(agents, bargain.mean, metrics.d_1, T=NUM_ITERS)

    distortions = [
        distortion(agents, i, optimal, metrics.d_1) for i in intermediates
    ]

    px.line(distortions).write_html('figs/line.html')

