from scipy import optimize

import src.bargain as bargain
import src.metrics as metrics
import src.opt as opt
import src.scripts.convergence
import src.scripts.mean_as_approximation_for_optimal
import src.scripts.sequential_deliberation_mean
import src.scripts.find_datasets_with_high_mean_distortion
from src.agent import get_d_dimensional_agents_summing_to_1
from src.distortion import distortion, social_cost
from src.plot import plot3d
from src.sequential import sequential_deliberation, sequential_deliberation_list

# src.scripts.convergence.main()
src.scripts.sequential_deliberation_mean.main()
# src.scripts.mean_as_approximation_for_optimal.main()
# src.scripts.find_datasets_with_high_mean_distortion.main()

