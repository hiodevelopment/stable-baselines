import os
import sys
import copy
import csv
import math
import time
import numpy as np
import tensorflow as tf

import gym
from gym.spaces import MultiDiscrete
from stable_baselines import PPO2, A2C
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.callbacks import BaseCallback, CheckpointCallback, EveryNTimesteps
from curiosity_mask.util import create_dummy_action_mask as mask
from curiosity_mask.util import set_action_mask_gait as gait_mask
from curiosity_mask.ui import UI

from transitions import Machine

env = gym.make('Acrobot-v1')

class Balance(object):

    def __init__(self):
        self.action_mask = []
        self.num_timesteps = None

    def is_sufficient_torque (self, event): 
        #return False # abs(event.kwargs.get('torque')) > 9.8
        #if math.cos(event.kwargs.get('angle_1')) < -0.7:  # Torso link is lifted high
        #    sys.exit()
        return math.cos(event.kwargs.get('angle_1')) < -0.8

    def back_to_swing (self, event): 
        if self.state == 'lift' and self.num_timesteps > 8:
            return True

    def reset_iter_counter(self, event):
        self.num_timesteps = 0

    def reset_step_counter(self, event):
        self.step_count = 0

    def log_iteration(self, event):
        if self.log: 
            print(self.num_timesteps, self.state, 'activated: ', self.action, round(event.kwargs.get('angle_1'), 4), round(event.kwargs.get('angle_2'), 4))

    def set_mask(self, event):

        if self.state == 'swing': 

            if event.kwargs.get('velocity_1') < 0:
                self.action_mask = [0,0,1]

            if event.kwargs.get('velocity_1') > 0:
                self.action_mask = [1,0,0]
        
        if self.state == 'lift':

            if self.action == 0:
                self.action_mask = [1,0,0]
            if self.action == 1:
                self.action_mask = [0,1,0]
            if self.action == 2:
                self.action_mask = [0,0,1]
               
class ActionMaskCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    
    def __init__(self, verbose=2, locals=locals, globals=globals):
        super(ActionMaskCallback, self).__init__(verbose)
        
        self.model = None  # type: BaseRLModel
        self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        self.n_calls = 0  # Number of time the callback was called # type: int
        self.num_timesteps = 0  # type: int
        self.locals = None  # type: Dict[str, Any]
        self.globals = None  # type: Dict[str, Any]

        # Set all actions to valid.
        self.action_mask = [1, 1, 1]

        states = ['swing', 'lift']
        self.gait = Balance()
        self.gait.num_timesteps = self.num_timesteps
        self.gait.action_mask = [1, 1, 1]
        self.gait.action = None
        self.gait.log = True

        machine = Machine(self.gait, states=states, send_event=True, initial='swing')

        # Setup state machine. 
        machine.add_transition('torque', 'swing', 'lift', conditions='is_sufficient_torque', prepare=['log_iteration', 'set_mask'], before=['reset_iter_counter'])
        machine.add_transition('torque', 'lift', 'swing', conditions='back_to_swing', prepare=['log_iteration', 'set_mask'], before=['reset_iter_counter'])

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        #print('train start')
        
        self.action_mask = self.gait.action_mask
        self.gait.num_timesteps = self.num_timesteps
            
        return True

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        #print('callback rollout')
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        
        #print('callback step')
        # Declare facts from environment.  
        self.gait.num_timesteps += 1
        sim = self.training_env.envs[0].state
        self.gait.action = self.training_env.envs[0].action

        # Torque = r x f x sin(theta)
        torque = 1*1*9.81*math.sin(sim[0])

        self.gait.torque(angle_1=sim[0], angle_2=sim[1], velocity_1=sim[2], velocity_2=sim[3], torque=torque)
        
        """
        if self.gait.state == 'swing':
            time.sleep(0.5)
        if self.gait.state == 'lift':
            time.sleep(1.5)
        """
        self.action_mask = self.gait.action_mask
        self.training_env.env_method('render')

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

# Callbacks for the Brains
callback = ActionMaskCallback()
  
brain = PPO2(MlpPolicy, env, verbose=2, tensorboard_log="walker_training_graphs/") # , tensorboard_log="acrobot_training_graphs/"
#brain.load("acrobot_strategy_brain")
brain.learn(100000, callback=callback)

brain.save("acrobot_strategy_brain")


