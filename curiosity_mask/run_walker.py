import gym
import numpy as np
import imageio

from gym.spaces import MultiDiscrete
import stable_baselines.common.walker_action_mask as walker
from stable_baselines import PPO2, A2C
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import DummyVecEnv
from curiosity_mask.util import create_dummy_action_mask as mask

# Create environment.
env = DummyVecEnv([walker.BipedalWalker])

brain = PPO2.load("walking_brain_1-1")

# Evaluate the agent.
#mean_reward, n_steps = evaluate_policy(brain, brain.get_env(), n_eval_episodes=1)
#print(mean_reward)

action_space = MultiDiscrete([3, 21, 21, 21, 21])
action_mask = mask(action_space)

# Predict from the trained agent and record animated gif. 
#images = []
obs, done, action_masks = env.reset(), [False], []
for i in range(1000):
    action, _states = brain.predict(obs, action_mask=action_masks)
    obs, rewards, dones, info = env.step(action)
    env.render()
    #images.append(env.render(mode='rgb_array'))

#imageio.mimsave('./logs/walker-ppo2-curiosity-mask.gif', images, fps=29)