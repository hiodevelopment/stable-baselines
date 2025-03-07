import stable_baselines.common.vec_env.bipedal_heuristic as bipedal
#import stable_baselines.common.vec_env.acrobot as acrobot
from stable_baselines import PPO2 as PPO
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

env = DummyVecEnv([bipedal.BipedalWalker])
# env = DummyVecEnv([acrobot.AcrobotEnv])

model = PPO(MlpPolicy, env, verbose=1, tensorboard_log="run/")
model.learn(10000)
model.learn(250)
# model.save("expert_model")

# Enjoy trained agent
for _ in range(25):
    obs, done, action_masks = env.reset(), [False], []
    for i in range(1000):
        action, _states = model.predict(obs, action_mask=action_masks)
        obs, _, done, infos = env.step(action)

        action_masks.clear()
        for info in infos:
            env_action_mask = info.get('action_mask')
            action_masks.append(env_action_mask)
        env.render()
