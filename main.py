from scipy import optimize
from src.distortion import social_cost, distortion
from src.sequential import (
    sequential_deliberation,
    sequential_deliberation_list,
)
import src.bargain as bargain
import src.metrics as metrics
import src.opt as opt
from src.plot import plot3d
from src.agent import get_d_dimensional_agents_summing_to_1
import src.scripts.mean_as_approximation_for_optimal
import src.scripts.sequential_deliberation_mean
import src.scripts.convergence

src.scripts.convergence.main()
# src.scripts.sequential_deliberation_mean.main()
# src.scripts.mean_as_approximation_for_optimal.main()




# agents_bc = get_d_dimensional_agents_summing_to_1(NUM_AGENTS, d=DIM)

# optimal_bc = opt.budget_constraint_min_d1(agents_bc)
# print(optimal_bc)
# print(sum(optimal_bc))
# print("Social cost of the optimal: " , social_cost(agents_bc, optimal_bc, metrics.d_1))
# print("Social cost of the mean: " , social_cost(agents_bc, opt.mean(agents_bc), metrics.d_1))

# plot3d(agents_bc, optimal_point=optimal_bc) 


# a_T = sequential_deliberation(agents, bargain.median_1d, metrics.d_1, T=NUM_ITERS)

# dist = distortion(agents, a_T, optimal_val, metrics.d_1)

# print(f"Optimal: {optimal_val}")
# print(f"a_T: {a_T}")
# print(f"distortion: {dist}")

# expected_dist = estimate_expected_distortion(
#     agents, bargain.median_1d, metrics.d_1, optimal_val, NUM_ITERS, NUM_SAMPLES
# )

# print(f"Expected distortion over {NUM_SAMPLES} samples: {expected_dist}")
