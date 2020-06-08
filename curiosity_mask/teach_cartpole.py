import os
import sys
import copy
import csv
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

env = gym.make('CartPole-v1')

class Balance(object):

    def __init__(self):
        self.action_mask = []
        self.num_timesteps = None

    def is_cart_near_edge (self, event): 
        if event.kwargs.get('cart_position') < -2.0 or event.kwargs.get('cart_position') > 2.0:
            return True

    def is_pole_falling (self, event): 
        if event.kwargs.get('pole_angle') < -0.15 or event.kwargs.get('pole_angle') > 0.15: # and event.kwargs.get('cart_position') >= -2.0 and event.kwargs.get('cart_position') <= 2.0:
            return True

    def is_balancing (self, event): 
        if event.kwargs.get('cart_position') >= -2.0 and event.kwargs.get('cart_position') <= 2.0 \
            and event.kwargs.get('pole_angle') >= -0.15 and event.kwargs.get('pole_angle') <= 0.15:
            return True

    def reset_iter_counter(self, event):
        self.num_timesteps = 0

    def reset_step_counter(self, event):
        self.step_count = 0

    def log_iteration(self, event):
        if self.log and (self.state == 'pole_falling' or self.state == 'cart_near_edge'): 
            print(self.num_timesteps, self.state, 'activated: ', round(event.kwargs.get('cart_position'), 4), round(event.kwargs.get('cart_velocity'), 4), round(event.kwargs.get('pole_angle'), 4), round(event.kwargs.get('pole_velocity'), 4))

    def set_mask(self, event):

        if self.state == 'balancing': # Alternate forces

            self.action_mask = [1,1]
        
        if self.state == 'cart_near_edge':

            if event.kwargs.get('cart_position') < 0:  # Straying left. 
                self.action_mask = [1, 0] # Push right. 
            if event.kwargs.get('cart_position') > 0:  # Straying right. 
                self.action_mask = [0, 1] # Push left.

        if self.state == 'pole_falling': 

            if event.kwargs.get('pole_angle') < 0:  # Falling left. 
                self.action_mask = [0, 1] # Push right. 
            if event.kwargs.get('pole_angle') > 0:  # Falling right. 
                self.action_mask = [1, 0] # Push left.
               
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
        self.action_mask = [1, 1]

        states = ['cart_near_edge', 'pole_falling', 'balancing']
        self.gait = Balance()
        self.gait.num_timesteps = self.num_timesteps
        self.gait.action_mask = [1, 1]
        self.gait.log = True

        machine = Machine(self.gait, states=states, send_event=True, initial='balancing')

        # Setup state machine. 
        machine.add_transition('force', 'balancing', 'pole_falling', conditions='is_pole_falling', prepare=['log_iteration', 'set_mask'], before=['reset_iter_counter'])
        machine.add_transition('force', 'balancing', 'cart_near_edge', conditions='is_cart_near_edge', unless='is_pole_falling', prepare=['log_iteration', 'set_mask'], before=['reset_iter_counter'])
        #machine.add_transition('force', 'cart_near_edge', 'pole_falling', conditions='is_pole_falling', unless='is_cart_near_edge', prepare=['log_iteration', 'set_mask'], before=['reset_iter_counter'])
        machine.add_transition('force', 'cart_near_edge', 'balancing', conditions='is_balancing', prepare=['log_iteration', 'set_mask'], before=['reset_iter_counter'])
        #machine.add_transition('force', 'pole_falling', 'cart_near_edge', conditions='is_cart_near_edge', unless='is_pole_falling', prepare=['log_iteration', 'set_mask'], before=['reset_iter_counter'])
        machine.add_transition('force', 'pole_falling', 'balancing', conditions='is_balancing', prepare=['log_iteration', 'set_mask'], before=['reset_iter_counter'])

        """
        machine.add_transition('reset', 'balancing', 'balancing', before=['reinstate', 'reset_step_counter', 'set_mask'])
        machine.add_transition('reset', 'cart_near_edge', 'balancing', before=['reinstate', 'reset_step_counter', 'set_mask'])
        machine.add_transition('reset', 'pole_falling', 'balancing', before=['reinstate', 'reset_step_counter', 'set_mask'])
        """

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

        self.gait.force(cart_position=sim[0], cart_velocity=sim[1], pole_angle=sim[2], pole_velocity=sim[3])
        
        self.action_mask = self.gait.action_mask
        #self.training_env.env_method('render')

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
  
brain = PPO2(MlpPolicy, env, verbose=2, tensorboard_log="walker_training_graphs/") # , tensorboard_log="cartpole_training_graphs/"
#brain.load("cartpole_strategy_step_brain")
brain.learn(100000, callback=callback)

brain.save("cartpole_strategy_step_brain")


