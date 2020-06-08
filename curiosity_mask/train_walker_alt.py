import os
import sys
import copy
import csv
import numpy as np
import tensorflow as tf

import gym
from gym.spaces import MultiDiscrete
import stable_baselines.common.walker_action_mask as walker
from stable_baselines import PPO2, A2C
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.callbacks import BaseCallback, CheckpointCallback, EveryNTimesteps
from curiosity_mask.util import create_dummy_action_mask as mask
from curiosity_mask.util import set_action_mask_gait as gait_mask
from curiosity_mask.ui import UI

from transitions import Machine

env = DummyVecEnv([walker.BipedalWalker])

class Walk(object):

    def __init__(self):
        self.action_mask = []
        self.num_timesteps = None

    def brain_transition_spring_forward(self, event): 
        if event.kwargs.get('hull_angle') < 0:
            return True

    def brain_transition_plant_leg(self, event): 
        #if event.kwargs.get('velocity') > 0: 
        if event.kwargs.get('velocity') > 0 and event.kwargs.get('hull_angle') < 0:
            self.action_mask[0] = [1, 1, 0]
        if event.kwargs.get('action')[0] == 1:
            self.action_mask[0] = [0, 1, 0]
            self.start = False
            return True

    def is_swinging_leg_planted(self, event): 
        if self.step_flag and event.kwargs.get('leg_force') > 58.528643843829634:
            self.action_mask[0] = [0, 0, 1]
            return True

    def brain_transition_lift_leg(self, event):
        if self.swinging_leg == 'left' and self.num_timesteps > 5 and event.kwargs.get('right_position') > event.kwargs.get('left_position'): 
            self.action_mask[0] = [1, 0, 1]
        if self.swinging_leg == 'right' and self.num_timesteps > 5 and event.kwargs.get('left_position') > event.kwargs.get('right_position'): 
            self.action_mask[0] = [1, 0, 1]
        if event.kwargs.get('action')[0] == 0: 
            self.action_mask[0] = [1, 0, 0] 
            return True
     
    def switch_legs(self, event):
        self.swinging_leg = 'right' if self.swinging_leg == 'left' else 'left'
        #print('switching legs: ', self.swinging_leg)

    def reset_iter_counter(self, event):
        self.num_timesteps = 0
        self.start_spring_forward_reward = 0

    def reset_step_counter(self, event):
        self.step_count = 0

    def is_start(self, event):
        return self.start
    
    def terminate(self, event):
        self.terminal = True

    def reinstate(self, event):
        self.terminal = False

    def increment_step_count(self, event):
        self.step_count += 1
        self.step_flag = True
        print('completed step: ', self.step_count)

    def log_iteration(self, event):
        if self.log:
            if event.kwargs.get('action') is not None:
                print(self.state, 'activated: ', event.kwargs.get('action'), self.action_mask[0], self.num_timesteps, self.swinging_leg, round(event.kwargs.get('right_hip_angle'), 4), event.kwargs.get('left_contact'), event.kwargs.get('right_contact'))
            else:
                print('Episode Start', 'no action', self.action_mask[0], self.num_timesteps, self.swinging_leg, event.kwargs.get('left_contact'), event.kwargs.get('right_contact'))
        else:
            pass

    def set_mask(self, event):

        if event.kwargs.get('action') is not None and len(event.kwargs.get('action')) == 5:
            gait = event.kwargs.get('action')[0]
        else:
            gait = 0

        teaching = self.teaching # event.kwargs.get('teaching')
        gait_ref = str(gait+1)
        gait_action_mask = self.action_mask[0]
        self.action_mask = mask(MultiDiscrete([3, 21, 21, 21, 21]))
        self.action_mask[0] = gait_action_mask

        # Flip bits to teach the strategy for this gait. (Left Hip, Left Knee, Right, Hip, Right Knee)
        if self.swinging_leg == 'left':

            ranges = {'left_hip': {'min': teaching['gait-' + gait_ref + '-swinging-hip-min'], 'max': teaching['gait-' + gait_ref + '-swinging-hip-max']},
                    'left_knee': {'min': teaching['gait-' + gait_ref + '-swinging-knee-min'], 'max': teaching['gait-' + gait_ref + '-swinging-knee-max']}, 
                    'right_hip': {'min': teaching['gait-' + gait_ref + '-planted-hip-min'], 'max': teaching['gait-' + gait_ref + '-planted-hip-max']}, 
                    'right_knee': {'min': teaching['gait-' + gait_ref + '-planted-knee-min'], 'max': teaching['gait-' + gait_ref + '-planted-knee-max']}
                }
            """
            if self.state == 'start_crouch':

                ranges['right_hip']['min'] = 10

            if self.state == 'start_spring_forward':

                ranges['right_hip']['max'] = 10
            """
            self.action_mask = gait_mask(gait, ranges, self.action_mask)

        if self.swinging_leg == 'right':

            ranges = {'left_hip': {'min': teaching['gait-' + gait_ref + '-planted-hip-min'], 'max': teaching['gait-' + gait_ref + '-planted-hip-max']},
                'left_knee': {'min': teaching['gait-' + gait_ref + '-planted-knee-min'], 'max': teaching['gait-' + gait_ref + '-planted-knee-max']}, 
                'right_hip': {'min': teaching['gait-' + gait_ref + '-swinging-hip-min'], 'max': teaching['gait-' + gait_ref + '-swinging-hip-max']}, 
                'right_knee': {'min': teaching['gait-' + gait_ref + '-swinging-knee-min'], 'max': teaching['gait-' + gait_ref + '-swinging-knee-max']}
            }
            """
            if self.state == 'start_crouch':

                ranges['left_hip']['min'] = 10

            if self.state == 'start_spring_forward':

                ranges['left_hip']['max'] = 10
            """
            #print('gait mask details: ', gait, ranges, self.action_mask[1])
            self.action_mask = gait_mask(gait, ranges, self.action_mask)   
        
class ActionMaskCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def remap(self, x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def translate_ui(self, ui):
        for key, value in ui.items():
            if not isinstance(value, str) and value is not None:
                ui[key] = int(self.remap(value, -10, 10, 0, 20))
        return ui

    def __init__(self, verbose=2, locals=locals, globals=globals):
        super(ActionMaskCallback, self).__init__(verbose)
        
        self.model = None  # type: BaseRLModel
        self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        self.n_calls = 0  # Number of time the callback was called # type: int
        self.num_timesteps = 0  # type: int
        self.locals = None  # type: Dict[str, Any]
        self.globals = None  # type: Dict[str, Any]

        # Set all actions to valid.
        self.action_space = MultiDiscrete([3, 21, 21, 21, 21])
        self.action_mask = mask(self.action_space)
        self.action_mask[0] = [1, 0, 0]

        #states = ['start_crouch', 'start_spring_forward', 'lift_leg', 'plant_leg', 'switch_leg']
        states = ['start', 'start_spring_forward', 'lift_leg', 'plant_leg', 'switch_leg']
        self.gait = Walk()
        self.gait.num_timesteps = self.num_timesteps
        self.gait.action_mask = mask(self.action_space)
        self.gait.action_mask[0] = [1, 0, 0]
        self.gait.swinging_leg = 'right'
        self.gait.terminal = False
        self.gait.start_spring_forward_reward = 0
        self.gait.start = True
        self.gait.step_count = 0
        self.gait.step_flag = False
        self.gait.log = True
        self.gait.starting_position = 4

        machine = Machine(self.gait, states=states, send_event=True, initial='start')

        # Setup for Pure Selector Orchestration
        machine.add_transition('move', 'start_crouch', 'start_spring_forward', conditions='brain_transition_spring_forward', prepare=['log_iteration', 'set_mask'], before=['reset_iter_counter'])
        #machine.add_transition('move', 'start_crouch', 'plant_leg', conditions='brain_transition_plant_leg', prepare=['log_iteration', 'set_mask'], before=['reset_iter_counter'])
        #machine.add_transition('move', 'start_spring_forward', 'plant_leg', conditions='brain_transition_plant_leg', prepare=['log_iteration', 'set_mask'], before=['reset_iter_counter'])
        machine.add_transition('move', 'start', 'plant_leg', conditions='brain_transition_plant_leg', prepare=['log_iteration', 'set_mask'], before=['reset_iter_counter'])
        machine.add_transition('move', 'lift_leg', 'plant_leg', conditions='brain_transition_plant_leg', unless='is_start', prepare=['log_iteration', 'set_mask'], before=['reset_iter_counter'])
        machine.add_transition('move', 'plant_leg', 'switch_leg', conditions='is_swinging_leg_planted', prepare=['log_iteration', 'set_mask'], before=['reset_iter_counter', 'switch_legs', 'increment_step_count'])
        machine.add_transition('move', 'switch_leg', 'lift_leg', conditions='brain_transition_lift_leg', prepare=['log_iteration', 'set_mask'], before=['reset_iter_counter'])

        machine.add_transition('reset', 'start', 'start', before=['reinstate', 'reset_step_counter', 'set_mask'])
        machine.add_transition('reset', 'start_spring_forward', 'start', before=['reinstate', 'reset_step_counter', 'set_mask'])
        machine.add_transition('reset', 'plant_leg', 'start', before=['reinstate', 'reset_step_counter', 'set_mask'])
        machine.add_transition('reset', 'switch_leg', 'start', before=['reinstate', 'reset_step_counter', 'set_mask'])
        machine.add_transition('reset', 'lift_leg', 'start', before=['reinstate', 'reset_step_counter', 'set_mask'])

        """    
        machine.add_transition('reset', 'start_crouch', 'start_crouch', before=['reinstate', 'reset_step_counter', 'set_mask'])
        machine.add_transition('reset', 'start_spring_forward', 'start_crouch', before=['reinstate', 'reset_step_counter', 'set_mask'])
        machine.add_transition('reset', 'plant_leg', 'start_crouch', before=['reinstate', 'reset_step_counter', 'set_mask'])
        machine.add_transition('reset', 'switch_leg', 'start_crouch', before=['reinstate', 'reset_step_counter', 'set_mask'])
        machine.add_transition('reset', 'lift_leg', 'start_crouch', before=['reinstate', 'reset_step_counter', 'set_mask'])
        """

        """
        machine.add_transition('reset', 'start_crouch', 'start_spring_forward', before=['reinstate', 'reset_step_counter', 'set_mask'])
        machine.add_transition('reset', 'start_spring_forward', 'start_spring_forward', before=['reinstate', 'reset_step_counter', 'set_mask'])
        machine.add_transition('reset', 'plant_leg', 'start_spring_forward', before=['reinstate', 'reset_step_counter', 'set_mask'])
        machine.add_transition('reset', 'switch_leg', 'start_spring_forward', before=['reinstate', 'reset_step_counter', 'set_mask'])
        machine.add_transition('reset', 'lift_leg', 'start_spring_forward', before=['reinstate', 'reset_step_counter', 'set_mask'])
        """

        #print('callback init')
        self.ui = UI()
        self.ui.event, self.ui.values = self.ui.window.read()
        if self.ui.event == 'Cancel':
            self.ui.window.close()
        elif self.ui.event == 'Reset':
           self.training_env.env_method('reset')
        elif self.ui.event in (None, 'Submit'):
           self.gait.teaching = self.translate_ui(self.ui.values) 

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        #print('train start')
        
        self.action_mask = self.gait.action_mask
        self.training_env.env_method('set_state_machine', self.gait) # Pass state machine to environment.
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
        sim = self.training_env.env_method('get_state')
        
        hull_angle = sim[0]['state'][0]
        velocity = sim[0]['state'][2]
        leg_force = self.training_env.envs[0].leg_force
        left_leg = {'side': 'left', 'contact': sim[0]['state'][8], 'position': sim[0]['legs'][0].position[0], 'hip_angle': sim[0]['state'][4], 'knee_angle': sim[0]['state'][6], 'hip_height': sim[0]['state'][26], 'knee_height': sim[0]['state'][24]}
        right_leg = {'side': 'right', 'contact': sim[0]['state'][13], 'position': sim[0]['legs'][1].position[0], 'hip_angle': sim[0]['state'][9], 'knee_angle': sim[0]['state'][11], 'hip_height': sim[0]['state'][27], 'knee_height': sim[0]['state'][25]}

        # Read from Machine Teaching UI
        #"""
        self.ui.event, self.ui.values = self.ui.window.read(timeout=0)
        self.gait.teaching = self.translate_ui(self.ui.values)
        self.gait.teaching['radio-1'] = self.ui.window.FindElement('radio-1').get()
        self.gait.teaching['radio-2'] = self.ui.window.FindElement('radio-2').get()
        self.ui.window.FindElement('teaching-button-1').Update(value=self.gait.state)
        if self.ui.event == 'Cancel':
            self.ui.window.close()
        elif self.ui.event == 'Reset':
           self.training_env.env_method('reset')
        #"""
        
        self.gait.move(action=sim[0]['action'], leg_force=leg_force, velocity=velocity, hull_angle=hull_angle, left_position=left_leg['position'], left_contact=left_leg['contact'], \
            right_position=right_leg['position'], right_contact=right_leg['contact'], \
                left_hip_angle=left_leg['hip_angle'], left_knee_angle=left_leg['knee_angle'], \
                    right_hip_angle=right_leg['hip_angle'], right_knee_angle=right_leg['knee_angle'], \
                        left_hip_height=left_leg['hip_height'], right_hip_height=right_leg['hip_height'], \
                            left_knee_height=left_leg['knee_height'], right_knee_height=right_leg['knee_height'], teaching=self.gait.teaching)
        
        self.action_mask = self.gait.action_mask
        self.training_env.env_method('render')
        if self.gait.terminal:
            self.training_env.env_method('set_terminal', self.gait.terminal)

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
  
brain = PPO2(MlpPolicy, env, verbose=2, tensorboard_log="walker_training_graphs/") # , tensorboard_log="walker_training_graphs/"
#brain.load("crouch_spring_first_step_brain")
brain.learn(250000, callback=callback)

brain.save("crouch_spring_first_step_brain")


