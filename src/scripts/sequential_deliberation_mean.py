import numpy as np
import pandas as pd
from scipy import optimize

import src.bargain as bargain
import src.metrics as metrics
import src.opt as opt
from src.agent import (
    get_d_dimensional_agents_summing_to_1,
    get_d_dimensional_agents_summing_to_1_with_clusters,
)
from src.distortion import distortion, social_cost
from src.plot import plot3d
from src.sequential import (
    sequential_deliberation,
    sequential_deliberation_list,
    sequential_deliberation_many_times,
)

NOISE = 0.05
NUM_ITERS = 20
NUM_SAMPLES = 1000

DATASETS = [
    (get_d_dimensional_agents_summing_to_1(30),'30 uniform'),
    (get_d_dimensional_agents_summing_to_1_with_clusters(
        50, 3, 2, NOISE, centers=[[0.6, 0.2, 0.2], [0.2, 0.3, 0.5]]
    ), '3 balanced clusters'),
    (get_d_dimensional_agents_summing_to_1_with_clusters(
        50, 3, 2, NOISE, [0.2, 0.8], [[0.6, 0.2, 0.2], [0.2, 0.3, 0.6]]
    ),'3 unbalanced clusters'),
    ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], '3 vertices'),
    ([[0.9347696648938575, 0.0, 0.06523033510614261], [0.9933626354660429, 0.006637364533957144, 0.0], [0.8535565979762921, 0.0, 0.1464434020237078], [0.9668366147590646, 0.013851694150829316, 0.01931169109010615], [0.9938878011613047, 0.0, 0.006112198838695391], [0.0, 0.9727260603041158, 0.027273939695884278], [0.08077058191982653, 0.8724132047280438, 0.04681621335212955], [0.0, 0.9570219785998746, 0.042978021400125355], [0.020577342690628726, 0.9037195125204007, 0.07570314478897053], [0.0, 0.9501087064047704, 0.04989129359522957], [0.029207715658479773, 0.0, 0.9707922843415202], [0.0, 0.06448956056299623, 0.9355104394370037], [0.0, 0.06842416950726855, 0.9315758304927315], [0.016955532664979473, 0.0, 0.9830444673350205], [0.0, 0.03382577519058839, 0.9661742248094117], [0.010793572480236628, 0.01468162894426617, 0.9745247985754971], [0.0, 0.10287348315546412, 0.8971265168445359], [0.018423656181977256, 0.037062874908109736, 0.944513468909913], [0.0, 0.0560698857874479, 0.943930114212552], [0.04232596888823735, 0.0, 0.9576740311117626], [0.0, 0.0384032361803314, 0.9615967638196686], [0.012992596296661316, 0.0, 0.9870074037033386], [0.0, 0.010784536434633292, 0.9892154635653667], [0.03443070726253642, 0.0, 0.9655692927374636], [0.053664456213000276, 0.0, 0.9463355437869997], [0.0, 0.018350980352956667, 0.9816490196470433], [0.007359284330569432, 0.0, 0.9926407156694306], [0.0, 0.059291916598086594, 0.9407080834019134], [0.018201747022623553, 0.05113404026538972, 0.9306642127119867], [0.02279720710388129, 0.09587944511042276, 0.8813233477856959], [0.02123653143295048, 0.04814392934998306, 0.9306195392170665], [0.0213950480359094, 0.058058409404274536, 0.920546542559816], [0.11607574567984304, 0.0, 0.883924254320157], [0.0, 0.0671137164877575, 0.9328862835122425], [0.011236108520484057, 0.04721930990037529, 0.9415445815791407], [0.06477076036192884, 0.045051396944771054, 0.8901778426933], [0.0, 0.028410659863905216, 0.9715893401360949], [0.02150878708438575, 0.014596823008260724, 0.9638943899073535], [0.0, 0.0645950329258359, 0.9354049670741641], [0.0, 0.09668280341737161, 0.9033171965826284], [0.0, 0.12825349339233144, 0.8717465066076685], [0.0, 0.06702194262143192, 0.932978057378568], [0.0, 0.09965076996483062, 0.9003492300351693], [0.09935179993854479, 0.0, 0.9006482000614552], [0.0, 0.13552678708100682, 0.864473212918993], [0.05166628994309425, 0.02296516239594242, 0.9253685476609632], [0.0, 0.16945368110151973, 0.8305463188984803], [0.0, 0.2131226216184798, 0.7868773783815203], [0.030218621061555703, 0.0, 0.9697813789384443], [0.04728662119491853, 0.03759574764820853, 0.9151176311568728]], '3 clusters centered at vertices')
]


def sequential_deliberation_once():
    agents = get_d_dimensional_agents_summing_to_1_with_clusters(
        NUM_AGENTS, d=DIM, noise=0.04, n_clusters=2, weights=[0.2, 0.8]
    )

    optimal = opt.budget_constraint_min_d1(agents)

    a_T = sequential_deliberation(agents, bargain.mean, metrics.d_1, T=NUM_ITERS)

    sc_opt = social_cost(agents, optimal, metrics.d_1)
    sc_mean = social_cost(agents, opt.mean(agents), metrics.d_1)
    sc_seq = social_cost(agents, a_T, metrics.d_1)

    print("Social cost of the optimal: ", sc_opt)
    print("Social cost of the mean: ", sc_mean)
    print("Social cost of the sequential mean: ", sc_seq)

    plot3d(agents, found_point=a_T, optimal_point=optimal)

def plot_datasets():
    for i, agents, name in enumerate(DATASETS):
        optimal = opt.budget_constraint_min_d1(agents)
        plot3d(agents, optimal_point=optimal, fname=f"figs/dataset_sequential_{i}.html")
        print(distortion(agents, opt.mean(agents), optimal, metrics.d_1))


def main():
    rows = []
    for i, (agents, name) in enumerate(DATASETS):
        results = sequential_deliberation_many_times(
            agents, bargain.mean, metrics.d_1, NUM_ITERS, NUM_SAMPLES
        ) # list of results
        optimal = opt.budget_constraint_min_d1(agents)

        for r in results:
            rows.append(
                [i, name, distortion(agents, r, optimal, metrics.d_1)]
            )
    df = pd.DataFrame(rows, columns=['dataset_ind', 'title', 'distortion'])
    df.to_csv("results/sequential.csv", index=False)