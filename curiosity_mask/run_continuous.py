import gym
import numpy as np
import stable_baselines.common.vec_env.bipedal_heuristic as bipedal

from stable_baselines import SAC
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

env = DummyVecEnv([bipedal.BipedalWalker])

model = SAC(MlpPolicy, env, verbose=1, tensorboard_log="run/")
model.learn(total_timesteps=50000)
#model.save("continuous_curiosity_mask")

#del model # remove to demonstrate saving and loading

#model = SAC.load("sac_pendulum")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
