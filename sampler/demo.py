from BayesianLearner import BayesianLearner
from vis import Visualizer
import numpy as np

if __name__ == '__main__':

    epsilons = [i / 100 for i in range(0, 50, 3)]
    config_file_name = "graphconfig"
    pklf = ".pkl"
    data_file_name = "samples"
    csvf = ".csv"

    data1 = []
    data2 = []
    error1_low = []
    error1_high = []
    error2_low = []
    error2_high = []

    variance1 = []
    variance2 = []

    for epsilon in epsilons:
        name_config = config_file_name + '{:.2f}'.format(epsilon)[2:] + pklf
        name_data = data_file_name + '{:.2f}'.format(epsilon)[2:] + csvf

        solver = BayesianLearner(name_config, name_data)
        # print(solver.P_g)

        solver.traj_Learner()
        print (solver.P_g)
        data1.append(solver.P_g['g1'])
        data2.append(solver.P_g['g2'])

        error1_low.append(abs(solver.P_g_min['g1'] - solver.P_g['g1']))
        error1_high.append(abs(solver.P_g_max['g1'] - solver.P_g['g1']))
        error2_low.append(abs(solver.P_g_min['g2'] - solver.P_g['g2']))
        error2_high.append(abs(solver.P_g_max['g2'] - solver.P_g['g2']))

        variance1.append(25*np.var(solver.P_g_history['g1']))
        variance2.append(25*np.var(solver.P_g_history['g2']))

    error1 = [error1_low, error1_high]
    error2 = [error2_low, error2_high]
    # print (variance1, variance2)
    print(error1)
    print(error2)

    VIS = Visualizer()
    # VIS.plot_curve(epsilons, data1, data2, variance1, variance2, 'result')
    VIS.plot_single_curve(epsilons, data1, variance1, 'result_single')