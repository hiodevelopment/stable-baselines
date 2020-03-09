import gym
import numpy as np
import imageio

from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import DummyVecEnv
import stable_baselines.common.vec_env.bipedal_heuristic_selector as bipedal

# Create environment.
env = DummyVecEnv([bipedal.BipedalWalker])

# Initialize and train agent.
model = PPO2('MlpPolicy', env, verbose=1, tensorboard_log="./logs/")
model.learn(total_timesteps=100000000, tb_log_name="Walker PPO2 - Discrete Curiosity Mask, Run 1")

# Evaluate the agent.
mean_reward, n_steps = evaluate_policy(model, model.get_env(), n_eval_episodes=1)
print(mean_reward)

# Predict from the trained agent and record animated gif. 
images = []
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    images.append(env.render(mode='rgb_array'))

imageio.mimsave('./logs/walker-ppo2-curiosity-mask.gif', images, fps=29)