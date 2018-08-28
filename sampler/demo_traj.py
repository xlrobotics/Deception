from BayesianLearner import BayesianLearner
from vis import Visualizer
import numpy as np

if __name__ == '__main__':
    config_file_name = "graphconfig"
    pklf = ".pkl"
    data_file_name = "samples"
    csvf = ".csv"

    epsilon = 0.00

    name_config = config_file_name + '{:.2f}'.format(epsilon)[2:] + pklf
    name_data = data_file_name + '{:.2f}'.format(epsilon)[2:] + csvf

    solver = BayesianLearner(name_config, name_data)
    # print(solver.P_g)

    convergence = solver.traj_Learner(early_stop = 2)

    epsilon = 0.27
    name_config_ = config_file_name + '{:.2f}'.format(epsilon)[2:] + pklf
    name_data_ = data_file_name + '{:.2f}'.format(epsilon)[2:] + csvf

    solver_ = BayesianLearner(name_config_, name_data_)
    convergence_ = solver_.traj_Learner(early_stop = 2)

    VIS = Visualizer()
    VIS.plot_trajs_compare(convergence, convergence_, 'test')
