import numpy as np
from copy import deepcopy as dcp
import csv
import pickle

class BayesianLearner:

    def __init__(self, filename = 'graphconfig.pkl', data = 'samples.csv', reward_constant = 100.0):
        '''
        :param filename: environment configuration, .pkl file
        :param data: sample trajectory
        '''
        config = None
        with open(filename, "rb") as config_file:
            config = pickle.load(config_file)
        if config == None:
            print("ERROR: CONFIGURATION FAILED")

        self.size = config['SIZE']
        self.S = config['STATES']
        self.A = config['ACTIONS']
        self.epsilon = config['EPSILON']
        self.gamma = config['GAMMA']
        self.threshold = config['THRESHOLD']
        self.decoys = config['DECOYS']
        self.goal_reward_value = reward_constant

        # initialize state action transition
        self.P_s1_a_s2 = self.get_P()

        # averaged belief over all samples
        # update rule P(g|h) = alpha*P(g|h) + (1-alpha)*P(g|new) , h <- h , new
        self.P_g = dcp(config['BELIEF'])
        self.P_g_min = dcp(config['BELIEF'])
        self.P_g_max = dcp(config['BELIEF'])
        self.P_g_history = {'g1':[], 'g2':[]}

        self.init_belief = dcp(config['BELIEF'])
        # initialize P(s'|s, g) = sum_a(P(a|s,g)*P(s'|s,a))
        self.P_s_g = dict.fromkeys(self.P_g)
        self.Pi_g = dict.fromkeys(self.P_g)
        for decoy in self.Pi_g.keys():
            self.Pi_g[decoy] = self.get_Policy(self.decoys[decoy])
            self.P_s_g[decoy] = self.get_Transition(decoy)

        # initialize P(|traj(t))
        # update rule P(g|traj(t+1)) propto P(g|traj(t))*P(s(t+1)|g, traj(t)), traj(t+1) <- traj(t).append(s(t+1))
        self.P_h_g = dcp(config['BELIEF'])
        # goals = list(self.P_g.keys())
        #
        # for state in self.S:
        #     s = tuple(state)
        #     self.P_h_g[s] = {}
        #     for goal in goals:
        #         self.P_h_g[s][goal] = 1/len(goals)

        # read training trajectory from the file
        self.trajs = []
        myFile = open(data, 'r', newline='')
        with myFile:
            reader = csv.reader(myFile)
            for row in reader:
                temp = []
                for s in row:
                    temp.append(eval(s))
                self.trajs.append(temp)

    # the stochastic law of grid world is: P(s'|s,a) = 1-epsilon when s' = s+a, otherwise P(s'|s+a) = epsilon/3, if s'
    # is outside the state space S, then P(s|s,a) += epsilon/3
    def get_P(self):
        P = {}
        S = self.S
        A = self.A
        epsilon = self.epsilon

        for state in S:
            s = tuple(state)
            explore = []

            for a in A.keys():
                temp = tuple(np.array(s) + np.array(A[a]))
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

    def get_Policy(self, goal, ng = [], init = None):

        def Dict2Vec(values, states):
            v = []
            for s in states:
                v.append(values[tuple(s)])
            return np.array(v)

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

        Pi = {}
        Q = {}
        V_record = []
        S = self.S
        A = self.A
        P = self.P_s1_a_s2
        threshold = self.threshold
        gamma = self.gamma

        tau = 1
        if init == None:
            V, V_ = init_V(S, goal, self.goal_reward_value, ng)
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

        return dcp(Pi)

    def get_Transition(self, goal):
        T = {}
        Pi = self.Pi_g[goal]
        S = self.S
        P = self.P_s1_a_s2

        for state in S:
            s = tuple(state)
            T[s] = {}
            if s not in self.decoys[goal]:
                for a in Pi[s]:
                    for s_ in P[s, a]:
                        if s_ not in T[s]:
                            T[s][s_] = 0.0
                        T[s][s_] += P[s, a][s_] * Pi[s][a]
            else:
                T[s][s] = 1.0

        return dcp(T)

    def Bayesian_inference(self, traj, return_flag):
        # update rule P(g|traj(t+1)) propto P(g|traj(t))*P(s(t+1)|g, traj(t)), traj(t+1) <- traj(t).append(s(t+1))
        last_s = traj[0]
        flag = False
        result = []
        for s in traj:
            #jump over the first round
            # print(self.P_h_g)
            if not flag:
                flag = True
                continue

            if return_flag:
                result.append(dcp(self.P_h_g['g1']))

            for goal in self.decoys:
                try:
                    self.P_h_g[goal] = self.P_h_g[goal] * self.P_s_g[goal][last_s][s]
                except KeyError:
                    self.P_h_g[goal] = 0.0
                    print (last_s, s, self.epsilon, self.P_s_g[goal][last_s])
            # normalization
            temp_sum = sum(self.P_h_g.values())
            for goal in self.decoys:
                self.P_h_g[goal] /= temp_sum

            last_s = s

        if return_flag:
            return result
        else:
            return None

    def traj_Learner(self, early_stop = 0):

        count = 1
        for traj in self.trajs:

            result = self.Bayesian_inference(traj, early_stop)

            #update belief
            for goal in self.decoys:
                self.P_g[goal] = (self.P_g[goal]*count + self.P_h_g[goal])/(count+1)

            # print(count, self.P_g)
            # print("====================================")

            if count == 1:
                for goal in self.decoys:
                    self.P_g_max[goal] = self.P_h_g[goal]
                    self.P_g_min[goal] = self.P_h_g[goal]

            for goal in self.decoys:
                if self.P_h_g[goal] > self.P_g_max[goal]:
                    self.P_g_max[goal] = self.P_h_g[goal]
                elif self.P_h_g[goal] < self.P_g_min[goal]:
                    self.P_g_min[goal] = self.P_h_g[goal]

                self.P_g_history[goal].append(self.P_h_g[goal])

            self.P_h_g = dcp(self.init_belief)

            if result is not None:
                yield result

            if count == early_stop:
                break

            count += 1

        # if result is not None:
        #     return result
        # else:
        #     return None
