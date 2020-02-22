import stable_baselines.common.vec_env.chess as chess
from stable_baselines import PPO2 as PPO
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

env = DummyVecEnv([chess.ChessEnv])
# env = DummyVecEnv([acrobot.AcrobotEnv])

n_steps = 0

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps
    # Print stats every 1000 calls
    if (n_steps + 1) % 5 == 0:
        # Set Masks
        piece_mask = [1] * 16
        position_mask = [1] * 64
        updated_masks = {'piece_mask' : piece_mask, 'position_mask' : position_mask}
        env.infos.update(updated_masks)
    n_steps += 1
    return True

model = PPO(MlpPolicy, env, verbose=1, tensorboard_log="run/")
model.learn(250000)

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
