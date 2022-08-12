import numpy as np
from DRL_Utils import LEASCHEnv
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

import keras
import tensorflow

import rl
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

# Get the environment and extract the number of actions available in the Cartpole problem
n_ue = 5
bs_parm = [{"pos": (0, 0, 30),
            "freq": 800,
            "numerology": 1,
            "power": 20,
            "gain": 16,
            "loss": 3,
            "bandwidth": 20,
            "max_bitrate": 1000}]

ue_parm = [{"x": 50,
            "y": 50,
            "z": 5,
            "uuid": "001",
            "buffer": 1024},
           {"x": 50,
            "y": 50,
            "z": 5,
            "uuid": "002",
            "buffer": 1024},
           {"x": 50,
            "y": 50,
            "z": 5,
            "uuid": "003",
            "buffer": 1024},
           {"x": 50,
            "y": 50,
            "z": 5,
            "uuid": "004",
            "buffer": 1024},
           {"x": 50,
            "y": 50,
            "z": 5,
            "uuid": "005",
            "buffer": 1024}
           ]

x_lim = 500
y_lim = 500
z_lim = 100
max_steps = 500
# --------------------------------------------
env = LEASCHEnv(x_lim, y_lim, n_ue, bs_parm, ue_parm, 0, 0, max_steps)

np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.size

# Lets build the neural network
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.size))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
# model.add(Flatten(input_shape=(1,) + env.observation_space.size))
print(model.summary())
#############################

policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this slows down training quite a lot.
dqn.fit(env, nb_steps=5000, visualize=True, verbose=2)

dqn.test(env, nb_episodes=5, visualize=True)
