import numpy as np
from copy import deepcopy as dcp

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import *

import csv
import pickle

from BayesianLearner import BayesianLearner
from vis import Visualizer

global_g = 100.0 #0.001
# transition probability setting P(s'|s,a)
# input:
# S: state space,
# wall_cord: boundary states,
# A: action dict,
# epsilon: 1-epsilon = transition probability for P(s+a|s,a)

# !!!the law of grid world is: P(s'|s,a) = 1-epsilon when s' = s+a, otherwise P(s'|s+a) = epsilon/3, if s' exceed
# the state space S, then P(s|s,a) += epsilon/3

def set_P(S, A, epsilon): #wall_cord,
    P = {}
    for state in S:
        s = tuple(state)
        explore = []

        for act in A.keys():
            temp = tuple(np.array(s) + np.array(A[act]))
            explore.append(temp)
        # print s, explore
        for a in A.keys():
            P[s, a] = {}
            P[s, a][s] = 0

            s_ = tuple(np.array(s) + np.array(A[a]))
            unit = epsilon / 3

            if list(s_) in S:

                P[s, a][s_] = 1 - epsilon
                for _s_ in explore:
                    if tuple(_s_) != s_:
                        if list(_s_) in S:
                            P[s, a][tuple(_s_)] = unit
                        else:
                            P[s, a][s] += unit
            else:
                P[s, a][s] = 1 - epsilon
                for _s_ in explore:
                    if _s_ != s_:
                        if list(_s_) in S:
                            P[s, a][tuple(_s_)] = unit
                        else:
                            P[s, a][s] += unit
    return dcp(P)

def set_S(l, w):
    inner = []
    for i in range(1, l):
        for j in range(1, w):
            inner.append([i, j])
    return inner

def Sigma(s, a, P, V_):
    total = 0.0

    for s_ in P[s, a].keys():
        if s_ != s:
            total += P[s, a][s_] * V_[s_]
    return total

def init_V(S, goal, g = 1.0, ng = None):
    V, V_ = {}, {}
    for state in S:
        s = tuple(state)
        if s not in V:
            V[s], V_[s] = 0.0, 0.0
        if s in goal:
            V[s], V_[s] = g, g
        if ng != None and s in ng:
            V[s], V_[s] = 0.0, 0.0
    return dcp(V), dcp(V_)

def init_R(S, goal, P, A, ng = None):
    R = {}
    g = 100.0

    # v = np.log((np.exp(0) * np.exp(0) + np.exp(g) * np.exp(g)) / (np.exp(0) + np.exp(g))) if len(goal) > 1 else g
    v = g
    nongoal = []
    if ng != None:
        nongoal = ng

    for state in S:
        s = tuple(state)
        if s not in R:
            R[s] = {}
        for a in A:
            if (s, a) in P:
                s_ = tuple(np.array(s) + np.array(A[a]))
                if s_ in goal:
                    R[s][a] = v
                elif s_ in nongoal:
                    R[s][a] = 0.0
                else:
                    R[s][a] = 0.0
    return dcp(R)

def Dict2Vec(V, S):
    v = []
    for s in S:
        v.append(V[tuple(s)])
    return np.array(v)

def Softmax_SVI(S, A, P, goal, ng, threshold, gamma, R, init = None):
    Pi = {}
    Q = {}
    V_record = []
    tau = 1
    if init == None:
        V, V_ = init_V(S, goal, global_g, ng)
    else:
        V, V_ = dcp(init), dcp(init)
        for unsafe in ng:
            V[unsafe], V_[unsafe] = 0, 0

    V_current, V_last = Dict2Vec(V, S), Dict2Vec(V_, S)

    it = 1

    while it == 1 or np.inner(V_current - V_last,
                   V_current - V_last) > threshold:  # np.inner(V_current - V_last, V_current - V_last) > threshold

        V_record.append(sum(V.values()))

        for s in S:
            V_[tuple(s)] = V[tuple(s)]

        # plot_heat("SVI_result" + str(it), V, 7, 9)

        for state in S:
            s = tuple(state)
            if s not in goal and s not in ng:

                # max_v, max_a = -1.0 * 99999999, None
                if s not in Pi:
                    Pi[s] = {}
                if s not in Q:
                    Q[s] = {}

                for a in A:
                    if (s, a) in P:
                        s_ = tuple(np.array(s) + np.array(A[a]))
                        if list(s_) not in S:
                            s_ = s
                            # Q[s][a] = 0
                            # continue
                        # Q[s][a] = np.exp((R[s][a] + gamma * Sigma(s, a, P, V_)) / tau)
                        # Q[s][a] = np.exp((R[s][a] + gamma * V_[s_]) / tau)
                        # Q[s][a] = np.exp((gamma * V_[s_]) / tau)

                        core = gamma * Sigma(s, a, P, V_) / tau
                        Q[s][a] = np.exp(core)
                        # Q[s][a] = np.exp((gamma * V_[s_]) / tau)
                        # print Q[tuple(s)]
                # print(sum(Q[s].values()))
                Q_s = sum(Q[s].values())
                for a in A:
                    if (s, a) in P:
                        # print (Q[s][a], Q_s)
                        Pi[s][a] = Q[s][a] / Q_s
                # print Q[tuple(s)].values()
                # V[s] = tau * np.log(np.dot(Q[s].values(), Pi[s].values())) # /len(Q[tuple(s)])
                V[s] = tau * np.log(Q_s)
            else:
                # print s, goal
                # V[s] = 0
                if s not in Pi:
                    Pi[s] = {}
                    for a in A:
                        Pi[s][a] = 0.0
                # Pi[tuple(s)] = []
                # pass
        # print V
        V_current, V_last = Dict2Vec(V, S), Dict2Vec(V_, S)
        it += 1
    # print "softmax iteration num:", it

    return dcp(Pi), dcp(V), dcp(V_record), dcp(Q)

def plot_heat(name, V, l, w):
    # print V
    temp = np.random.random((l-1, w-1))
    for i in range(l-1):
        for j in range(w-1):
            s = tuple([i+1, j+1])
            if s in V:
                temp[i,j] = V[s]
            else:
                temp[i,j] = -1

    fig = plt.figure()
    # x, y = np.mgrid[-1.0:1.0:19j, -1.0:1.0:19j]
    # # #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # #
    # ax.plot_wireframe(x, y, temp)
    # # ax.plot_surface(x, y, temp)
    # plt.show()

    cax = plt.imshow(temp, cmap='hot', interpolation='nearest')
    fig.colorbar(cax)
    plt.savefig(name + ".png") # "../DFA/comparing test/"

def plot_traj(name, V, l, w, traj): #dir,
    temp = np.random.random((l, w))
    for i in range(l):
        for j in range(w):
            s = tuple([i, j])
            if s in V:
                if s in traj:
                    temp[s] = V[s]
                else:
                    temp[s] = -1
            else:
                temp[s] = -1

    x, y = np.mgrid[-1.0:1.0:l, -1.0:1.0:w]

    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # ax.plot_wireframe(x, y, temp)
    # ax.plot_surface(x, y, temp)
    # plt.show()
    plt.imshow(temp, cmap='hot', interpolation='nearest')
    plt.savefig(name + ".png")  # "../DFA/comparing test/"

def plot_curve(trace1, trace2, trace, name):
    plt.figure()

    # print x
    # print trace1
    # print trace2
    l2, = plt.plot(trace2, label="V2")
    l1, = plt.plot(trace1, label="V1")

    l, = plt.plot(trace, label="V")

    plt.legend(handles=[l1, l2, l])

    plt.xlabel('Value iteration episode')
    plt.ylabel('Value iteration function at (9, 11)')
    plt.show()
    # plt.savefig(name + ".png")


def sampling(S, Pi, P, V, goal, S_normal, A, start):
    hit = {}
    # print goal
    for g in goal:
        hit[g] = 0.0

    traj = []

    for i in range(1000):
        # print i
        # k = randint(0, len(S_normal)-1)
        # start = S_normal[k]


        sc = tuple(start)
        # dis = abs(3-start[0]) + abs(3 - start[1])
        t = 0
        # v = 0

        traj_i = [sc]

        while 1:
            # if t > 6:
            #     break
            # v += V[sc]

            if sc in goal:
                if t == (abs(sc[0]-start[0]) + abs(sc[1]-start[1])):
                    hit[sc] += 1
                # traj_i.append(sc)
                break
            # print Pi
            # print (Pi[sc].keys(), Pi[sc].values())
            action = list(np.random.choice(list(Pi[sc].keys()), 1, p=list(Pi[sc].values())))[0]
            # print P[sc, action].keys(), P[sc, action].values()
            index = [i for i in range(len(P[sc, action].keys()))]
            # print len(index), len(P[sc, action].values()), len(P[sc, action].keys())
            nsc = list(np.random.choice(index, 1, p=list(P[sc, action].values())))[0]
            sc = list(P[sc, action].keys())[nsc]
            traj_i.append(sc)
            t += 1
        traj.append(traj_i)
    # for key in hit:
    #     hit[key] /= sum(hit.values())
    return hit, traj


if __name__ == '__main__':

    l, w = 11, 11
    S = set_S(l, w) # set the width and length of grid world
    print(S)

    A = {"N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1)} # initialize action space

    epsilons = [i / 100 for i in range(0, 50, 3)]

    # epsilon = 0.48 # define transition probability: 1-epsilon = P(s+a|s,a)
    gamma = 0.9

    a = "N"
    threshold = 0.000001

    goal1 = [(6, 3)]  # (10, 11)
    goal2 = [(6, 9)]  # (10, 11)
    goal = [(6, 3), (6, 9)]
    start = (1, 6)

    config_file_name = "graphconfig"
    pklf = ".pkl"
    data_file_name = "samples"
    csvf = ".csv"

    # plotter = Visualizer()
    # plotter.plot_heat_map([l,w], S, goal, goal1, start, 'graph')
    # TODO: add goal switching sampling result, control the switching state, and observe if the Bayesian inference react
    # TODO: immediately or delayed.

    for epsilon in epsilons:

        P = set_P(S, A, epsilon) #w

        config = {}
        config['SIZE'] = [l, w]
        config['STATES'] = S
        config['ACTIONS'] = A
        config['EPSILON'] = epsilon
        config['GAMMA'] = gamma
        config['THRESHOLD'] = threshold
        config['DECOYS'] = {'g1': goal1, 'g2': goal2}
        config['BELIEF'] = {'g1': 0.5, 'g2': 0.5}


        with open(config_file_name + '{:.2f}'.format(epsilon)[2:] + pklf, "wb") as config_file:
            pickle.dump(config, config_file)


        S_normal = []
        for state in S:
            s = tuple(state)
            if s not in goal1:
                S_normal.append(s)


        R1 = init_R(S, goal1, P, A)
        R2 = init_R(S, goal2, P, A)


        Pi1, V1, Q1, traj1 = Softmax_SVI(S, A, P, goal1, [], threshold, gamma, R1)
        Pi2, V2, Q2, traj2 = Softmax_SVI(S, A, P, goal2, [], threshold, gamma, R2)

        plot_heat("V1", V1, l, w)
        hit, traj = sampling(S, Pi1, P, V1, goal1, S_normal, A, start)
        myFile = open(data_file_name + '{:.2f}'.format(epsilon)[2:] + csvf, 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(traj)


    # print trajectories
    # for row in solver.trajs:
    #     print (row)


    # fig, ax1 = plt.subplots()
    #
    # # ax1.grid(True)
    # ax2 = ax1.twiny()
    #
    # l1, = ax2.plot(index, collecting, color='r', label=r'Composed $\sigma_1 \wedge \sigma_2$')
    # l2, = ax2.plot(index, truth_plot, color='g', label=r'Original $\sigma_1 \wedge \sigma_2$')
    # l3, = ax1.plot(w_list2, collecting2, color='b', label=r'Composed $\sigma_1 \vee \sigma_2$')
    # l4, = ax1.plot(w_list2, truth_plot2, color='y', label=r'Original $\sigma_1 \vee \sigma_2$')
    #
    # ax1.tick_params(axis='x')
    #
    # # ax1.set_ylim([0, 10])
    # # ax2.set_ylim([0, 10])
    #
    # ax2.set_xlabel(r'Weight: $\omega$')
    # ax1.set_xlabel(r'Weight Reciprocal: $1/\omega$')
    # # ax2.set_xticks(w_list2)
    # # plt.ylabel('Intersection/Nonintersection Ratio')
    # plt.ylabel(r'$P(\sigma_1 \wedge \sigma_2)/P(\sigma_1 \wedge not \sigma_2)$')
    #
    # plt.legend(handles=[l1, l2, l3, l4])
    #
    # ax2.tick_params(axis='x')
    #
    # # plt.setp(l1, color='r', linewidth=2.0)
    # # plt.setp(l2, color='g', linewidth=2.0)
    # # plt.setp(l3, color='b', linewidth=2.0)
    # # plt.setp(l4, color='y', linewidth=2.0)
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.show()
    #
    #
