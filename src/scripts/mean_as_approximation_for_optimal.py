from src.distortion import distortion
from src.agent import get_d_dimensional_agents_summing_to_1
import src.opt as opt
import src.metrics as metrics
import pandas as pd
from tqdm import tqdm

N_SAMPLES = 100
N = 20
d = 10
def main():
    distortions = []
    for _ in tqdm(range(N_SAMPLES)):
        agents = get_d_dimensional_agents_summing_to_1(N, d)
        optimal = opt.budget_constraint_min_d1(agents)
        mean = opt.mean(agents)
        distortions.append(distortion(agents, mean,optimal, metrics.d_1))

    pd.DataFrame(distortions, columns=["DISTORTION"]).to_csv('mean_appox_optimal_10d.csv')

    
    