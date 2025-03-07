import gym
import numpy as np
from gym.spaces import Discrete, MultiDiscrete


class DiscreteActionMaskEnv(gym.Env):
    metadata = {'render.modes': ['human', 'system', 'none']}

    def __init__(self):
        self.action_space = Discrete(3)

        self.observation_shape = (1, 10, 10)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=np.float16)

        self.counter = 0
        self.valid_actions = [1, 1, 1]

    def reset(self):
        self.counter = 0
        self.valid_actions = [1, 1, 1]
        return self.state()

    def step(self, action: int):
        valid_actions = [1, 1, 1]
        if self.valid_actions[action] == 0:
            raise Exception("Invalid action was selected! Valid actions: {}, "
                            "action taken: {}".format(self.valid_actions, action))
        valid_actions[action] = 0

        self.counter += 1
        self.valid_actions = valid_actions
        return self.state(), 0, self.finish(), {'action_mask': self.valid_actions}

    def render(self, mode='human'):
        pass

    def finish(self):
        return self.counter == 128

    def state(self):
        tmp = np.reshape(np.array([*range(100)]), self.observation_shape)
        obs = tmp / 100
        return obs


class MultiDiscreteActionMaskEnv(gym.Env):
    metadata = {'render.modes': ['human', 'system', 'none']}

    def __init__(self):
        self.action_space = MultiDiscrete([2, 3, 4])

        self.observation_shape = (1, 10, 10)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=np.float16)

        self.counter = 0
        self.valid_actions1 = [1, 1]
        self.valid_actions2 = [[1, 1, 1],
                               [1, 1, 1]]
        self.valid_actions3 = [[[1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1]],
                               [[1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1]]]
        self.valid_actions = [self.valid_actions1, self.valid_actions2, self.valid_actions3]

    def generate_mask(a,b,c,d):

        action_mask = []

        # Two elements
        action_mask1 = [1 for x in range(a)]

        # Two lists of 3 elements each. 
        action_mask2 = [[1]*b for x in range(a)]

        #Two lists with 3  lists with 4 elements each.
        action_mask3 = [[[1]*c for y in range(b)] for x in range(a)]

        #Two lists, with 3 lists, with 4 lists with 5 elements each
        action_mask4 = [[[1]*d for z in range(c)]for y in range(b) for x in range(a)]
        
        return [action_mask1, action_mask2, action_mask3, action_mask4]


    def reset(self):
        self.counter = 0
        self.valid_actions1 = [1, 1]
        self.valid_actions2 = [[1, 1, 1],
                               [1, 1, 1]]
        self.valid_actions3 = [[[1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1]],
                               [[1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1]]]
        self.valid_actions = [self.valid_actions1, self.valid_actions2, self.valid_actions3]
        return self.state()

    def step(self, actions):
        valid_actions1 = [1, 1]
        valid_actions2 = [[1, 1, 1],
                          [1, 1, 1]]
        valid_actions3 = [[[1, 1, 1, 1],
                           [1, 1, 1, 1],
                           [1, 1, 1, 1]],
                          [[1, 1, 1, 1],
                           [1, 1, 1, 1],
                           [1, 1, 1, 1]]]

        if self.valid_actions[0][actions[0]] == 0:
            raise Exception("Invalid action was selected! Valid actions: {}, "
                            "action taken: {}".format(self.valid_actions, actions))
        else:
            valid_actions1[actions[0]] = 0
        if self.valid_actions[1][actions[0]][actions[1]] == 0:
            raise Exception("Invalid action was selected! Valid actions: {}, "
                            "action taken: {}".format(self.valid_actions, actions))
        else:
            valid_actions2[0][actions[1]] = 0
            valid_actions2[1][actions[1]] = 0
        if self.valid_actions[2][actions[0]][actions[1]][actions[2]] == 0:
            raise Exception("Invalid action was selected! Valid actions: {}, "
                            "action taken: {}".format(self.valid_actions, actions))
        else:
            valid_actions3[0][0][actions[2]] = 0
            valid_actions3[0][1][actions[2]] = 0
            valid_actions3[0][2][actions[2]] = 0
            valid_actions3[1][0][actions[2]] = 0
            valid_actions3[1][1][actions[2]] = 0
            valid_actions3[1][2][actions[2]] = 0

        self.valid_actions = [valid_actions1, valid_actions2, valid_actions3]
        self.counter += 1

        return self.state(), 0, self.finish(), {'action_mask': self.valid_actions}

    def render(self, mode='human'):
        pass

    def finish(self):
        return self.counter == 250

    def state(self):
        tmp = np.reshape(np.array([*range(100)]), self.observation_shape)
        obs = tmp / 100
        return obs
