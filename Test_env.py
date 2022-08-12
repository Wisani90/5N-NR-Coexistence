from DRL_Utils import LEASCHEnv
import gym
import numpy as np

# Da spostare ---------------------------
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

state = env.reset()

print(env.observation_space.shape)
num_steps = 99
for s in range(num_steps + 1):
    print(f"step: {s} out of {num_steps}")

    # sample a random action from the list of available actions
    action = env.action_space.sample()

    # perform this action on the environment
    env.step(action)

    # print the new state
    # env.render()

# end this instance of the taxi environment
env.close()
