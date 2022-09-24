import numpy as np
from DRL_Utils import LEASCHEnv
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, CategoryEncoding
from keras.optimizers import Adam

import keras
import tensorflow

import rl
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from environment.OneHotPolicy import OneHotPolicy

# Get the environment and extract the number of actions available in the Cartpole problem
n_ue = 5
bs_parm = [{"pos": (0, 0, 30),
            "freq": 1700,
            "numerology": 2,
            "power": 20,
            "gain": 16,
            "loss": 10,
            "bandwidth": 20,
            "max_bitrate": 1000}]

ue_parm = [{"x": 450,
            "y": 700,
            "z": 5,
            "uuid": "001",
            "buffer": 1024000},
           {"x": 50,
            "y": 30,
            "z": 5,
            "uuid": "002",
            "buffer": 1024000},
           {"x": 250,
            "y": 150,
            "z": 5,
            "uuid": "003",
            "buffer": 1024000},
           {"x": 500,
            "y": 650,
            "z": 5,
            "uuid": "004",
            "buffer": 1024000},
           {"x": 850,
            "y": 800,
            "z": 5,
            "uuid": "005",
            "buffer": 1024000}
           ]

x_lim = 500
y_lim = 500
z_lim = 100
max_steps = 100000
# --------------------------------------------
env = LEASCHEnv(x_lim, y_lim, n_ue, bs_parm, ue_parm, 0, 0, max_steps)

np.random.seed(123)
env.seed(123)
nb_actions = 5
print(nb_actions)
obs_shape = ()  # env.action_space.size

print("Observation")
print(env._get_obs())
# Lets build the neural network
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
# model.add(Flatten(input_shape=(1,2,5)))
# print(model.output_shape)
model.add(Dense(16))
# print(model.output_shape)
model.add(Activation('relu'))
# print(model.output_shape)
model.add(Dense(nb_actions))
# print(model.output_shape)
model.add(Activation('linear'))
print(model.output_shape)
# model.add(Dense(1))
# model.add(CategoryEncoding(num_tokens=nb_actions, output_mode="one_hot"))
# model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
print(model.summary())
#############################

# model = generate_dense_model((window_length,) + env.observation_space.shape, layers, nb_actions)

policy = EpsGreedyQPolicy()
print(policy)
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this slows down training quite a lot.
dqn.fit(env, nb_steps=5000, visualize=True, verbose=2)

dqn.test(env, nb_episodes=5, visualize=True)
