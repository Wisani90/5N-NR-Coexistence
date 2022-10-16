import gym

from wns2.basestation.nrbasestation import NRBaseStation
from wns2.environment.environment import Environment
from wns2.renderer.renderer import CustomRenderer
import numpy.random as random
import logging
from wns2.userequipment.userequipment import UserEquipment
from wns2.environment.ObservationSpace import ObservationSpace
from wns2.environment.OneHotEncoding import OneHotEncoding
import numpy as np
import math

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)


class LEASCHEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def init_env(self, x_lim, y_lim, terr_parm, n_ue, ue_parm):
        #Create Environment con tutti i parametri
        self.env = Environment(x_lim, y_lim, renderer = CustomRenderer())
        self.init_pos = []  # for reset method
        self.LoggedSignals = {}
        for i in range(0, n_ue):
            pos = (
            ue_parm[i]["x"], ue_parm[i]["y"], ue_parm[i]["z"])  # pos = (random.rand()*x_lim, random.rand()*y_lim, 1)
            ue = UserEquipment(self.env, ue_parm[i]["uuid"], 25, pos, speed=0, direction=random.randint(0, 360),
                               _lambda_c=5, _lambda_d=15)
            ue.feed_buffer(ue_parm[i]["buffer"])
            self.env.add_user(ue)
            self.init_pos.append(pos)
        for i in range(len(terr_parm)):
            self.env.add_base_station(
                NRBaseStation(self.env, "001", terr_parm[i]["pos"], terr_parm[i]["freq"], terr_parm[i]["bandwidth"],
                              terr_parm[i]["numerology"], terr_parm[i]["max_bitrate"], terr_parm[i]["power"],
                              terr_parm[i]["gain"], terr_parm[i]["loss"]))

        return

    def __init__(self, x_lim, y_lim, n_ue, terr_parm, ue_parm, queue_parm, load_parm, max_steps):
        super(LEASCHEnv, self).__init__()
        #crea classe
        #definiscie attributi aciton space
        #definisci attributi observation space
        self.len_p = len(terr_parm)
        self.n_ue = n_ue
        self.queue_parm = queue_parm
        self.load_parm = load_parm

        # Define action and observation space
        self.action_space = OneHotEncoding(n_ue)
        # Example for using image as input:
        self.observation_space = ObservationSpace(2 * n_ue)
        self.steps = max_steps
        self.max_steps = max_steps
        self.terr_param = terr_parm
        self.LoggedSignals = {}
        self.ue_parm = ue_parm
        self.init_env(x_lim, y_lim, terr_parm, n_ue, ue_parm)
        return

    def _get_obs(self):
        # metodo per prendere osservation variable dallo stato stato del sistema
        d = self.compute_d()
        f = self.compute_f()
        g = self.compute_g()
        print('G VECTOR:')
        print(g)
        # print('F VECTOR:')
        # print(f)
        print('D VECTOR:')
        print(d)
        return self.compute_obs(d, g, f)


    #Auxiliary functions for state observation
    def compute_g(self):
        # TO DO
        env = self.env
        ue_list = env.ue_list
        g = np.zeros(self.n_ue)
        ue_ids = list(ue_list.keys())
        for i in range(self.n_ue):
            ue_id = ue_ids[i]
            ue = env.ue_by_id(ue_id)
            if (ue.isEligible()):
                g[i] = 1

        return g

    def take_unic_base_station(self):
        env = self.env
        bs_list = env.bs_list
        bs_list_ids = list(bs_list.keys())

        bs_id = bs_list_ids[0]
        bs = self.env.bs_by_id(bs_id)

        return bs

    def compute_d(self):
        # TO DO
        env = self.env
        ue_list = env.ue_list
        d = np.zeros(self.n_ue)
        # bs_list = env.bs_list
        # bs_list_ids = list(bs_list.keys())

        # bs_id = bs_list_ids[0]
        # bs = self.env.bs_by_id(bs_id)
        bs = self.take_unic_base_station()

        ue_rate = bs.ue_data_rate_allocation

        ue_ids = list(ue_list.keys())

        for i in range(self.n_ue):
            ue_id = ue_ids[i]

            if (ue_id not in ue_rate):
                d[i] = 0
            else:
                d[i] = ue_rate[ue_id]
        return d

    def compute_f(self):
        # TO DO
        # TO DO
        env = self.env
        ue_list = env.ue_list
        f = np.zeros(self.n_ue)
        ue_ids = list(ue_list.keys())
        for i in range(self.n_ue):
            ue_id = ue_ids[i]
            ue = env.ue_by_id(ue_id)
            f[i] = ue.fairness
        return f

    def compute_obs(self, d, g, f):
        # TO DO
        s = np.multiply(d, g)

        max_s = np.max(s)

        if (max_s == 0):
            s = s
        else:
            s = np.multiply(s, (1 / max_s))

        max_f = np.max(f)

        if (max_f == 0):
            f = f
        else:
            f = np.multiply(f, (1 / max_f))

        s = np.reshape(s, (5, 1))
        f = np.reshape(f, (5, 1))
        return np.concatenate((s, f), axis=0)

    #######################################

    def _get_info(self):
        # Info ausiliarie ritornate da step e reset
        env = self.env
        ue_list = env.ue_list
        bs_list = env.bs_list

        info = {}
        info["UEs"] = ue_list
        info["AP"] = self.take_unic_base_station()
        info["LoggedSignals"] = self.LoggedSignals

        return info

    def step(self, action):
        # To DO..
        print(action)
        ue_index = action
        # action_array = np.zeros(5)
        # action_array[action-1] = 1
        #
        # action = action_array

        info_bis = self._get_info()

        # s = self._get_obs()

        metrics_matrix = info_bis["LoggedSignals"]["Data_Rate"]
        metrics_array_index = info_bis["LoggedSignals"]["scheduled_RBG"]

        print("observation before action:")
        # print(s)
        g_prev = self.compute_g()
        # print("g_prev:")
        # print(g_prev)

        # Extract d_hat and f data from the state s
        # print("ACTION!!!!")
        # print(action.shape)
        # d_hat = s[1:len(action), 0]
        # f = s[len(action) + 1:2 * len(action), 0]

        # ue_index = -1
        # for u in range(len(action)):
        #     if (action[u] == 1):
        #         ue_index = u

        # print("info base station:")
        # print(info_bis["UEs"])
        # print("ue_index:")
        # print(ue_index)
        # print("AP name:")
        # print(info_bis["AP"])
        self.perform_action(info_bis["UEs"], ue_index, info_bis["AP"], self.LoggedSignals, g_prev)

        print("observation after action:")
        observation = self._get_obs()
        print(observation)

        f = self.compute_f()
        d = self.compute_d()

        data_rate = np.transpose(d)

        metrics_matrix[:, metrics_array_index] = d

        # Compute Reward
        print('d computation')
        print(d)
        print(g_prev)
        d_hat = np.multiply(d, g_prev)
        max_d_hat = np.max(d_hat)
        if (not (max_d_hat == 0)):
            d_hat = np.multiply(d_hat, (1 / max_d_hat))
        # Domani comincia a vedere perché la d computation on funziona, restituisce d = [0 0 0 0]
        print('d hat computation')
        print(d_hat)
        reward = self.compute_reward(ue_index, 0.5, info_bis["UEs"], d_hat, f, g_prev)
        print('f value')
        print(f)
        print('REWARD!!')
        print(reward)
        # Evaluate terminal condition

        cond = 1
        g = self.compute_g()

        # for ue in info_bis["UEs"]:
        #     if (info_bis["UEs"][ue].buffer == 0):
        #         cond = 0

        for elem in g:
            if (elem == 1):
                cond = 0
        if (cond):
            print(
                "Doneeeeeeee !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print()
        done = cond
        self.LoggedSignals["Data_Rate"] = metrics_matrix
        self.LoggedSignals["scheduled_RBG"] = self.LoggedSignals["scheduled_RBG"] + 1
        info = {}
        # print("REWARD!!!!")
        # print(reward)
        return observation, reward, done, info

    def perform_action(self, UEs, ue_index, AP, LoggedSignals, g):
        env = self.env
        ue_list = env.ue_list
        ue_ids = list(ue_list.keys())

        ue = UEs[ue_ids[ue_index]]

        ris = np.zeros((1, len(ue_ids)))

        ue_selected = ''

        for i in range(len(ue_ids)):
            ue_id = ue_ids[i]
            if (ue_index == i):
                selected = 1
                ue_selected = ue_id
            else:
                selected = 0
            UEs[ue_id].update_fairness(selected)

        # print(g)
        self.LoggedSignals['count_schedule'][ue_index] = self.LoggedSignals['count_schedule'][ue_index] + 1
        # print(ue_index)
        if (g[ue_index] == 1):
            # print("BS ID")
            # print(AP.get_position())
            # print(ue_selected)
            # print("UE POSITION")
            # print(UEs[ue_selected].get_position())
            self.LoggedSignals['good_schedule'][ue_index] = self.LoggedSignals['good_schedule'][ue_index] + 1
            print(UEs[ue_selected].connect_bs(AP.bs_id))

            schedule_result = AP.schedule(UEs[ue_selected], 2)
            BUFFER_REDUCTION = 2 * AP.get_buffer_reduction(UEs[ue_selected].ue_id, 2)
            print('BUFFER EDUCTION:')
            print(BUFFER_REDUCTION)
            UEs[ue_selected].reduce_buffer(BUFFER_REDUCTION)

        if (AP.reset_condition()):
            AP.disconnect_all()

        return

    def compute_reward(self, ue_index, k, UEs, d_hat, f, g):
        env = self.env
        ue_list = env.ue_list
        ue_ids = list(ue_list.keys())
        ue = UEs[ue_ids[ue_index]]

        if (not (g[ue_index])):
            reward = -k
        else:
            if (np.max(f) == 0):
                norm_f = 0
            else:
                norm_f = np.min(f[f > 0]) / np.max(f)
                print('norm f')
                print(norm_f)
                reward = d_hat[ue_index] * norm_f

        return reward

    def reset(self, seed=None, return_info=False, options=None):
        # To DO...
        print(
            "Resetttttt #################################################################################################################")

        env = self.env
        ue_list = env.ue_list
        ue_ids = list(ue_list.keys())
        info_bis = self._get_info()
        UEs = info_bis["UEs"]

        for i in range(len(ue_ids)):
            ue_id = ue_ids[i]
            UEs[ue_id].feed_buffer(self.ue_parm[i]["buffer"])

        # for i in range(0, self.n_ue):
        #     pos = (
        #         self.ue_parm[i]["x"], self.ue_parm[i]["y"],
        #         self.ue_parm[i]["z"])  # pos = (random.rand()*x_lim, random.rand()*y_lim, 1)
        #     ue = UserEquipment(self.env, self.ue_parm[i]["uuid"], 25, pos, speed=0, direction=random.randint(0, 360),
        #                        _lambda_c=5, _lambda_d=15)
        #     ue.feed_buffer(self.ue_parm[i]["buffer"])
        #     self.env.add_user(ue)
        #     self.init_pos.append(pos)

        x_lim = self.env.get_x_limit()
        y_lim = self.env.get_y_limit()

        # self.init_env(x_lim, y_lim, self.terr_param, self.n_ue, self.ue_param)
        observation = self._get_obs()

        f = np.zeros((1, self.n_ue))
        d = np.ones((1, self.n_ue))
        observation = np.concatenate((d, f), axis=1)
        observation = np.transpose(observation)

        # clean the render collection and add the initial frame
        # self.render.reset()
        # self.render.render_step()

        self.LoggedSignals["Data_Rate"] = np.zeros((self.n_ue, 100000))
        self.LoggedSignals["count_schedule"] = np.zeros(self.n_ue)
        self.LoggedSignals["good_schedule"] = np.zeros(self.n_ue)
        self.LoggedSignals["scheduled_RBG"] = 1

        # observation = self._get_obs()
        return observation

    def render(self, mode='human'):
        # return self.env.render()
        return

    def close(self):
        return