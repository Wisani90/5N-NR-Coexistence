import numpy as np
from DRL_Utils import LEASCHEnv
import gym
from MTB import ModifiedTensorBoard
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, CategoryEncoding
from keras.optimizers import Adam
import os
import datetime
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import rl
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy

import tf_agents
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

ue_parm = [{"x": 1500,
            "y": 1000,
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
           {"x": 900,
            "y": 850,
            "z": 5,
            "uuid": "004",
            "buffer": 1024000},
           {"x": 1500,
            "y": 2000,
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

training = 1

test_policy = GreedyQPolicy()
policy = EpsGreedyQPolicy()

print(policy)

memory = SequentialMemory(limit=50000, window_length=1)

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy, test_policy=test_policy)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

# Test tensorflow training ######################

# Convert from gym spaces to tf spaces
# tf_agents.environments.gym_wrapper.spec_from_gym_space(space: gym.Space)
# tf_env = tf_agents.environments.suite_gym.wrap_env(env)
# train_env = tf_agents.environments.TFPyEnvironment(tf_env)
################################################


# cb1 = keras.callbacks.TensorBoard(log_dir='./logs')
# cb1 = tf.compat.v1.keras.callbacks.TensorBoard(
#     log_dir='./logs',
#     histogram_freq=0,
#     batch_size=32,
#     write_graph=True,
#     write_grads=False,
#     write_images=False,
#     embeddings_freq=0,
#     embeddings_layer_names=None,
#     embeddings_metadata=None,
#     embeddings_data=None,
#     update_freq='epoch',
#     profile_batch=2
# )


# cb1 = ModifiedTensorBoard("model", log_dir="./logs")
# callback_list = [cb1]
# Okay, now it's time to learn something! We visualize the training here for show, but this slows down training quite a lot.
# history = dqn.fit(env, nb_steps=500000, visualize=True, verbose=2) ci mette tanto


if (training):
    # Training
    history = dqn.fit(env, nb_steps=10000, visualize=False, verbose=2, nb_max_episode_steps=3000)
    date = datetime.datetime.now().strftime("%m%d%Y%H%M")
    model.save("./saved_models/model" + date + ".h5")
    dqn.save_weights("./saved_models/weights" + date + ".h5")
    print(history)
    print("ciaoooooooooooo")
    x = range(len(history.history['episode_reward']))
    y = history.history['episode_reward']

    plt.plot(x, y)
    plt.show()

    np.save('my_history.npy', history.history)
    os.system("CLS")
    env.reset()
    print("#################\n")
    print("#               #\n")
    print("#    Start      #\n")
    print("#    Testing    #\n")
    print("#               #\n")
    print("#################\n")
    test_history = dqn.test(env)

    LoggedSignals = env.LoggedSignals
    ################################
    # Plots of data rates

    x = range(np.size(env.LoggedSignals["Data_Rate"], 1))
    # for i in range( 1, len(LoggedSignals["Data_Rate"].size(1)) ):

    y = np.transpose(LoggedSignals["Data_Rate"])
    plt.plot(x, y)
    plt.xlim([0, 500])
    plt.show()
    ################################
    # Histogram plot
    x = [1, 2, 3, 4, 5]

    plt.bar(x, height=LoggedSignals["good_schedule"])
    plt.show()

    plt.bar(x, height=LoggedSignals["count_schedule"])
    plt.show()


else:
    model = keras.models.load_model("./saved_models/model101620221904.h5")
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=test_policy, test_policy=test_policy)
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    # Testing
    dqn.load_weights("./saved_models/weights101620221904.h5")
    history = dqn.test(env, verbose=1, visualize=True, nb_max_episode_steps=20000)
    LoggedSignals = env.LoggedSignals
    ################################
    # Plots of data rates

    x = range(np.size(env.LoggedSignals["Data_Rate"], 1))
    # for i in range( 1, len(LoggedSignals["Data_Rate"].size(1)) ):

    y = np.transpose(LoggedSignals["Data_Rate"])
    plt.plot(x, y)
    plt.xlim([0, 5000])
    plt.show()
    ################################
    # Histogram plot
    x = [1, 2, 3, 4, 5]

    plt.bar(x, height=LoggedSignals["good_schedule"])
    plt.show()

    plt.bar(x, height=LoggedSignals["count_schedule"])
    plt.show()

# print(history)

# dqn.test(env, nb_episodes=5, visualize=True)
