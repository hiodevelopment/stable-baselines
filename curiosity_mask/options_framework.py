import os
import sys
import numpy as np

from gym.spaces import MultiDiscrete
from stable_baselines.common.vec_env import DummyVecEnv
import stable_baselines.common.walker_action_mask as walker

# Run through all the actions to generate the resulting state vectors.
left_hip = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
left_knee = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
right_hip = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
right_knee = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

possible_actions = []
possible_states = []
for x_index, x_value in enumerate(list(filter(lambda x: x == 0.2, left_hip))):
    for y_index, y_value in enumerate(list(filter(lambda x: x == -1, left_knee))):
        for z_index, z_value in enumerate(list(filter(lambda x: x >= -0.4 and x <= 0.5, right_hip))):
            for j_index, j_value in enumerate(list(filter(lambda x: x == 0.1, right_knee))):
                possible_actions.append((x_value, y_value, z_value, j_value))
                #possible_states(walker().simulate(x_value, y_value, z_value, j_value))
print('possible states: ', len(possible_actions), possible_states)
sys.exit()