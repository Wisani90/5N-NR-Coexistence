import gym
from gym import spaces
from wns2.basestation.nrbasestation import NRBaseStation
from wns2.environment.environment import Environment
from wns2.renderer.renderer import CustomRenderer
import numpy.random as random
import logging
from wns2.userequipment import userequipment
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
        for i in range(0, n_ue):
            pos = (ue_parm[i]["x"], ue_parm[i]["y"], ue_parm[i]["z"]) #pos = (random.rand()*x_lim, random.rand()*y_lim, 1)
            self.env.add_user(userequipment(self.env, ue_parm[i]["uuid"], 25, pos, speed = 0, direction = random.randint(0, 360), _lambda_c=5, _lambda_d = 15))
            self.init_pos.append(pos)
        for i in range(len(terr_parm)):
            self.env.add_base_station(NRBaseStation(self.env, i, terr_parm[i]["pos"], terr_parm[i]["freq"], terr_parm[i]["bandwidth"], terr_parm[i]["numerology"], terr_parm[i]["max_bitrate"], terr_parm[i]["power"], terr_parm[i]["gain"], terr_parm[i]["loss"]))

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
        self.observation_space = ObservationSpace(2*n_ue)
        self.steps = max_steps
        self.max_steps = max_steps
        self.terr_param = terr_parm

        self.init_env(x_lim, y_lim, terr_parm, n_ue, ue_parm)
        return

    def _get_obs(self):
        # metodo per prendere osservation variable dallo stato stato del sistema
        d = self.compute_d()
        f = self.compute_f()
        g = self.compute_g()
        return self.compute_obs(d, g, f)

    #Auxiliary functions for state observation
    def compute_g(self):
        #TO DO
        env = self.env
        ue_list = env.ue_list
        g = np.zeros((1, self.n_ue))
        for i in range(self.n_ue):
            ue_id = ue_list[i]
            ue = env.ue_by_id(ue_id)
            if(ue.isEligible()):
                g[1, i] = 1
        return g

    def compute_d(self):
        # TO DO
        env = self.env
        ue_list = env.ue_list
        d = np.zeros((1, self.n_ue))
        bs_list = env.bs_list
        bs_list_ids = list(bs_list.keys())

        bs_id = bs_list_ids[0]
        bs = self.env.bs_by_id(bs_id)

        ue_rate = bs.ue_data_rate_allocation

        for i in range(self.n_ue):
            ue_id = ue_list[i]
            d[1, i] = ue_rate[ue_id]
        return d

    def compute_f(self):
        # TO DO
        # TO DO
        env = self.env
        ue_list = env.ue_list
        f = np.zeros((1, self.n_ue))
        for i in range(self.n_ue):
            ue_id = ue_list[i]
            ue = env.ue_by_id(ue_id)
            f[1, i] = ue.fairness
        return f

    def compute_obs(self, d, g, f):
        # TO DO
        s = np.multiply(d, g)
        max_s = max(s)
        if (max_s == 0):
            s = s
        else:
            s = np.multiply(s, (1 / max_s))

        max_f = max(f)

        if (max_f == 0):
            f = f
        else:
            f = np.multiply(f, (1 / max_f))

        return np.vstack((np.transpose(s), np.transpose(f)))

    #######################################

    def _get_info(self):
        # Info ausiliarie ritornate da step e reset
        return

    def step(self):
        # To DO...
        return


    def reset(self):
        #To DO...
        return

    def render(self, mode='human'):
        return self.env.render()
        return

    def close(self):
        return