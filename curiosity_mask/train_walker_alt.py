import os
import sys
import warnings
import numpy as np
import tensorflow as tf

from gym.spaces import MultiDiscrete
import stable_baselines.common.walker_action_mask as walker
from stable_baselines import PPO2, A2C
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.callbacks import BaseCallback, CheckpointCallback, EveryNTimesteps
from curiosity_mask.util import create_dummy_action_mask as mask
from curiosity_mask.ui import UI

from transitions import Machine

""""
from experta import *
from experta.fact import *
import schema
"""

env = DummyVecEnv([walker.BipedalWalker])

class Walk(object):

    # Define Gait Phases
    """
    conditions = ['is_swinging_leg_lifted', 'is_swinging_leg_planted', 'is_swinging_leg_leading']
    conditions[0] = # lifting leg knee is above threshold.
    conditions[1] = # lifting leg makes contact with the ground. 
    condisions[2] = # new swinging leg is now in front of the other leg and needs to be lifted. 
    """
    def __init__(self):
        self.action_mask = []
        self.num_timesteps = None

    def is_swinging_leg_lifted(self, event): 
        if abs(event.kwargs.get('left_hip_angle') - event.kwargs.get('right_hip_angle')) > 0.15: # angle between legs
            self.start = False
            return True
        """"
        if self.swinging_leg == 'right' and self.num_timesteps > 5:
            self.start = False # No longer in launch phase. 
            return event.kwargs.get('left_knee_height') > 0.85*event.kwargs.get('right_hip_height') 
        if self.swinging_leg == 'left' and self.num_timesteps > 5:
            self.start = False # No longer in launch phase.
            return event.kwargs.get('right_knee_height') > 0.85*event.kwargs.get('left_hip_height') 
        """

    def done_lifting_swinging_leg(self, event): 
        #return event.kwargs.get('action')[0] == 1  # The brain decided to switch.
        #"""
        if self.num_timesteps > 10 and event.kwargs.get('action')[0] != 1: # Brain hasn't changed yet, but teaching limit reached.
            self.action_mask[0] = [0, 0, 1]
            return True
        else:
            return event.kwargs.get('action')[0] == 1  # The brain decided to switch. 
        #"""

    def is_swinging_leg_planted(self, event): 
        if self.swinging_leg == 'right':
            return bool(event.kwargs.get('right_contact')) #or self.num_timesteps > 5 
        if self.swinging_leg == 'left':
            return bool(event.kwargs.get('left_contact')) #or self.num_timesteps > 5

    def is_swinging_leg_leading(self, event):
        if self.swinging_leg == 'right' and self.num_timesteps > 5:  #  and self.num_timesteps > 30
            return event.kwargs.get('right_position') > event.kwargs.get('left_position') # based on height of knee
        if self.swinging_leg == 'left' and self.num_timesteps > 5:  #   and self.num_timesteps > 30
            return event.kwargs.get('left_position') > event.kwargs.get('right_position') # based on height of knee
        

    def start_lifting_swinging_leg(self, event): 
        if self.swinging_leg == 'right': 
            if event.kwargs.get('right_position') > event.kwargs.get('left_position'):
                if self.num_timesteps > 5 or event.kwargs.get('action')[0] == 2:  # The brain decided to start planting the swinging leg.
                    self.action_mask[0] = [1, 0, 0]
                    return True
        if self.swinging_leg == 'left': 
            if event.kwargs.get('left_position') > event.kwargs.get('right_position'): 
                if self.num_timesteps > 5 or event.kwargs.get('action')[0] == 2:  # The brain decided to start planting the swinging leg.
                    self.action_mask[0] = [1, 0, 0]
                    return True
        
        
 
    def switch_legs(self, event):
        self.swinging_leg = 'right' if self.swinging_leg == 'left' else 'left'
        #print('switching legs: ', self.swinging_leg)

    def set_fuzzy_decision1(self, event):
        self.action_mask[0] = [1, 1, 0]

    def set_fuzzy_decision2(self, event):
        self.action_mask[0] = [0, 1, 1]

    def reset_iter_counter(self, event):
        self.num_timesteps = 0

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

    def set_mask_start(self, event): 
        """
        if event.kwargs.get('action') is not None:
            print('Start Activated', event.kwargs.get('action'), self.action_mask[0], self.num_timesteps, round(event.kwargs.get('left_hip_angle'), 2), round(event.kwargs.get('right_hip_angle'), 2), self.swinging_leg, event.kwargs.get('left_contact'), event.kwargs.get('right_contact'))
        else:
            print(self.teaching)
            print('Start Activated ', 'no action', self.action_mask[0], self.num_timesteps, self.swinging_leg, event.kwargs.get('left_contact'), event.kwargs.get('right_contact'))
        """
        gait = 1
        teaching = self.teaching
        """
        # Rules
        if self.swinging_leg == 'right' and self.num_timesteps > 2 and not bool(event.kwargs.get('left_contact')): # Planted leg not on the ground. 
            self.terminal = True
        if self.swinging_leg == 'left' and self.num_timesteps > 2 and not bool(event.kwargs.get('right_contact')): # Planted leg not on the ground.
            self.terminal = True
        """
        # Each concept rule needs to mask the gait and flip bits for control concept.
        self.action_mask[0] = [1, 0, 0]  # Limit selection to gait phase 1.

        # Flip bits for joint modifications. (Left Hip, Left Knee, Right, Hip, Right Knee)
        if self.swinging_leg == 'left':
            self.action_mask[1][gait-1] = [1 if (action_index >= 20 and action_index <= 20) else 0 for action_index in range(21)] # Positive hip motion for planted leg.
            for joint1_value in range(21):
                self.action_mask[2][gait-1][joint1_value] = [1 if (action_index >= teaching['slider1-2a'] and action_index <= teaching['slider1-2b']) else 0 for action_index in range(21)] # Negative hip motion for swinging leg.
            for joint1_value in range(21):
                    for joint2_value in range(21):
                        self.action_mask[3][gait-1][joint1_value][joint2_value] = [1 if (action_index == teaching['slider1-3a'] and action_index == teaching['slider1-3b']) else 0 for action_index in range(21)] # Slight Positive knee motion for planted leg.
            for joint1_value in range(21):
                for joint2_value in range(21):
                    for joint3_value in range(21):
                        self.action_mask[4][gait-1][joint1_value][joint2_value][joint3_value] = [1 if (action_index >= teaching['slider1-4a'] and action_index <= teaching['slider1-4b']) else 0 for action_index in range(21)] # Negative knee motion for swinging leg.
            
        #print('swinging hip: ', self.action_mask[1][gait-1], )
        #print('swinging knee: ', self.action_mask[2][gait-1][joint1_value])
        #print('planted hip: ', self.action_mask[3][gait-1][joint1_value][joint2_value])
        #print('planted knee: ', self.action_mask[4][gait-1][joint1_value][joint2_value][joint3_value])

        if self.swinging_leg == 'right':
            self.action_mask[1][gait-1] = [1 if (action_index >= teaching['slider1-3a'] and action_index <= teaching['slider1-3b']) else 0 for action_index in range(21)] # Positive hip motion for planted leg.
            for joint1_value in range(21):
                self.action_mask[2][gait-1][joint1_value] = [1 if (action_index >= teaching['slider1-4a'] and action_index <= teaching['slider1-4b']) else 0 for action_index in range(21)] # Negative hip motion for swinging leg.
            for joint1_value in range(21):
                    for joint2_value in range(21):
                        self.action_mask[3][gait-1][joint1_value][joint2_value] = [1 if (action_index >= 20 and action_index <= 20) else 0 for action_index in range(21)] # Slight Positive knee motion for planted leg.
            for joint1_value in range(21):
                for joint2_value in range(21):
                    for joint3_value in range(21):
                        self.action_mask[4][gait-1][joint1_value][joint2_value][joint3_value] = [1 if (action_index >= teaching['slider1-2a'] and action_index <= teaching['slider1-2b']) else 0 for action_index in range(21)] # Negative knee motion for swinging leg.
        pass

    def set_mask_gait_1(self, event): 
        """
        if event.kwargs.get('action') is not None:
            if self.state == 'fuzzy_transition_1':
                print('Fuzzy Transition 1 Activated', event.kwargs.get('action'), self.action_mask[0], self.num_timesteps, self.swinging_leg, round(event.kwargs.get('left_hip_angle'), 2), round(event.kwargs.get('right_hip_angle'), 2), event.kwargs.get('left_contact'), event.kwargs.get('right_contact'))
            else:
                print('Concept 1 Activated', event.kwargs.get('action'), self.action_mask[0], self.num_timesteps, self.swinging_leg, round(event.kwargs.get('left_hip_angle'), 2), round(event.kwargs.get('right_hip_angle'), 2), event.kwargs.get('left_contact'), event.kwargs.get('right_contact'))
        else:
            print(self.teaching)
            print('Concept 1 Activated ', 'no action', self.action_mask[0], self.num_timesteps, self.swinging_leg, event.kwargs.get('left_contact'), event.kwargs.get('right_contact'))
        """
        gait = 1
        teaching = self.teaching
        # Each concept rule needs to mask the gait and flip bits for control concept.
        self.action_mask[0] = [1, 0, 0]  # Limit selection to gait phase 1.

        # Flip bits for joint modifications. (Left Hip, Left Knee, Right, Hip, Right Knee)
        if self.swinging_leg == 'left':
            self.action_mask[1][gait-1] = [1 if (action_index >= teaching['slider1-1a'] and action_index <= teaching['slider1-1b']) else 0 for action_index in range(21)] # Positive hip motion for planted leg.
            for joint1_value in range(21):
                self.action_mask[2][gait-1][joint1_value] = [1 if (action_index >= teaching['slider1-2a'] and action_index <= teaching['slider1-2b']) else 0 for action_index in range(21)] # Negative hip motion for swinging leg.
            for joint1_value in range(21):
                    for joint2_value in range(21):
                        self.action_mask[3][gait-1][joint1_value][joint2_value] = [1 if (action_index == teaching['slider1-3a'] and action_index == teaching['slider1-3b']) else 0 for action_index in range(21)] # Slight Positive knee motion for planted leg.
            for joint1_value in range(21):
                for joint2_value in range(21):
                    for joint3_value in range(21):
                        self.action_mask[4][gait-1][joint1_value][joint2_value][joint3_value] = [1 if (action_index >= teaching['slider1-4a'] and action_index <= teaching['slider1-4b']) else 0 for action_index in range(21)] # Negative knee motion for swinging leg.
            
        #print('swinging hip: ', self.action_mask[1][gait-1], )
        #print('swinging knee: ', self.action_mask[2][gait-1][joint1_value])
        #print('planted hip: ', self.action_mask[3][gait-1][joint1_value][joint2_value])
        #print('planted knee: ', self.action_mask[4][gait-1][joint1_value][joint2_value][joint3_value])

        if self.swinging_leg == 'right':
            self.action_mask[1][gait-1] = [1 if (action_index >= teaching['slider1-3a'] and action_index <= teaching['slider1-3b']) else 0 for action_index in range(21)] # Positive hip motion for planted leg.
            for joint1_value in range(21):
                self.action_mask[2][gait-1][joint1_value] = [1 if (action_index >= teaching['slider1-4a'] and action_index <= teaching['slider1-4b']) else 0 for action_index in range(21)] # Negative hip motion for swinging leg.
            for joint1_value in range(21):
                    for joint2_value in range(21):
                        self.action_mask[3][gait-1][joint1_value][joint2_value] = [1 if (action_index >= teaching['slider1-1a'] and action_index <= teaching['slider1-1b']) else 0 for action_index in range(21)] # Slight Positive knee motion for planted leg.
            for joint1_value in range(21):
                for joint2_value in range(21):
                    for joint3_value in range(21):
                        self.action_mask[4][gait-1][joint1_value][joint2_value][joint3_value] = [1 if (action_index >= teaching['slider1-2a'] and action_index <= teaching['slider1-2b']) else 0 for action_index in range(21)] # Negative knee motion for swinging leg.
        pass

    def set_mask_gait_2(self, event): 
        """
        print('Concept 2 Activated', event.kwargs.get('action'), self.action_mask[0], round(event.kwargs.get('left_position'), 2), \
           round(event.kwargs.get('right_position'), 2), self.num_timesteps, self.swinging_leg, round(event.kwargs.get('left_hip_angle'), 2), round(event.kwargs.get('right_hip_angle'), 2), event.kwargs.get('left_contact'), event.kwargs.get('right_contact'))
        """
        gait = 2
        teaching = event.kwargs.get('teaching')
        # Each concept rule needs to mask the gait and flip bits for control concept.
        self.action_mask[0] = [0, 1, 0]  # Limit selection to gait phase 2.

        # Flip bits for joint modifications. (Left Hip, Left Knee, Right, Hip, Right Knee)
        if self.swinging_leg == 'left':
            self.action_mask[1][gait-1] = [1 if (action_index >= teaching['slider2-1a'] and action_index <= teaching['slider2-1b']) else 0 for action_index in range(21)] # Positive hip motion for planted leg.
            for joint1_value in range(21):
                self.action_mask[2][gait-1][joint1_value] = [1 if (action_index >= teaching['slider2-2a'] and action_index <= teaching['slider2-2b']) else 0 for action_index in range(21)] # Negative hip motion for swinging leg.
            for joint1_value in range(21):
                    for joint2_value in range(21):
                        self.action_mask[3][gait-1][joint1_value][joint2_value] = [1 if (action_index >= teaching['slider2-3a'] and action_index <= teaching['slider2-3b']) else 0 for action_index in range(21)] # Slight Positive knee motion for planted leg.
            for joint1_value in range(21):
                for joint2_value in range(21):
                    for joint3_value in range(21):
                        self.action_mask[4][gait-1][joint1_value][joint2_value][joint3_value] = [1 if (action_index >= teaching['slider2-4a'] and action_index <= teaching['slider2-4b']) else 0 for action_index in range(21)] # Negative knee motion for swinging leg.
            
        if self.swinging_leg == 'right':
            self.action_mask[1][gait-1] = [1 if (action_index >= teaching['slider2-3a'] and action_index <= teaching['slider2-3b']) else 0 for action_index in range(21)] # Positive hip motion for planted leg.
            for joint1_value in range(21):
                self.action_mask[2][gait-1][joint1_value] = [1 if (action_index >= teaching['slider2-4a'] and action_index <= teaching['slider2-4b']) else 0 for action_index in range(21)] # Negative hip motion for swinging leg.
            for joint1_value in range(21):
                    for joint2_value in range(21):
                        self.action_mask[3][gait-1][joint1_value][joint2_value] = [1 if (action_index >= teaching['slider2-1a'] and action_index <= teaching['slider2-1b']) else 0 for action_index in range(21)] # Slight Positive knee motion for planted leg.
            for joint1_value in range(21):
                for joint2_value in range(21):
                    for joint3_value in range(21):
                        self.action_mask[4][gait-1][joint1_value][joint2_value][joint3_value] = [1 if (action_index >= teaching['slider2-2a'] and action_index <= teaching['slider2-2b']) else 0 for action_index in range(21)] # Negative knee motion for swinging leg.
        pass

    def set_mask_gait_3(self, event): 
        """
        if self.state == 'fuzzy_transition_2':
            print('Fuzzy Transition 2 Activated', self.action_mask[0], round(event.kwargs.get('left_position'), 2), \
                round(event.kwargs.get('right_position'), 2), self.num_timesteps, self.swinging_leg, round(event.kwargs.get('left_hip_angle'), 2), round(event.kwargs.get('right_hip_angle'), 2), event.kwargs.get('left_contact'), event.kwargs.get('right_contact'))
        else:
            print('Concept 3 Activated', event.kwargs.get('action'), self.action_mask[0], round(event.kwargs.get('left_position'), 2), \
                round(event.kwargs.get('right_position'), 2), self.num_timesteps, self.swinging_leg, round(event.kwargs.get('left_hip_angle'), 2), round(event.kwargs.get('right_hip_angle'), 2), event.kwargs.get('left_contact'), event.kwargs.get('right_contact'))
        """
        gait = 3
        teaching = event.kwargs.get('teaching')
        # Each concept rule needs to mask the gait and flip bits for control concept.
        self.action_mask[0] = [0, 0, 1]  # Limit selection to gait phase 3.

        # Flip bits for joint modifications.  (Left Hip, Left Knee, Right, Hip, Right Knee)
        if self.swinging_leg == 'left':
            self.action_mask[1][gait-1] = [1 if (action_index >= teaching['slider3-1a'] and action_index <= teaching['slider3-1b']) else 0 for action_index in range(21)] # Positive hip motion for planted leg.
            for joint1_value in range(21):
                self.action_mask[2][gait-1][joint1_value] = [1 if (action_index >= teaching['slider3-2a'] and action_index <= teaching['slider3-2b']) else 0 for action_index in range(21)] # Negative hip motion for swinging leg.
            for joint1_value in range(21):
                    for joint2_value in range(21):
                        self.action_mask[3][gait-1][joint1_value][joint2_value] = [1 if (action_index >= teaching['slider3-3a'] and action_index <= teaching['slider3-3b']) else 0 for action_index in range(21)] # Slight Positive knee motion for planted leg.
            for joint1_value in range(21):
                for joint2_value in range(21):
                    for joint3_value in range(21):
                        self.action_mask[4][gait-1][joint1_value][joint2_value][joint3_value] = [1 if (action_index >= teaching['slider3-4a'] and action_index <= teaching['slider3-4b']) else 0 for action_index in range(21)] # Negative knee motion for swinging leg.
        
        #print('swinging hip: ', self.action_mask[1][gait-1])
        #print('swinging knee: ', self.action_mask[2][gait-1][joint1_value])
        #print('planted hip: ', self.action_mask[3][gait-1][joint1_value][joint2_value])
        #print('planted knee: ', self.action_mask[4][gait-1][joint1_value][joint2_value][joint3_value])

        if self.swinging_leg == 'right':
            self.action_mask[1][gait-1] = [1 if (action_index >= teaching['slider3-3a'] and action_index <= teaching['slider3-3b']) else 0 for action_index in range(21)] # Positive hip motion for planted leg.
            for joint1_value in range(21):
                self.action_mask[2][gait-1][joint1_value] = [1 if (action_index >= teaching['slider3-4a'] and action_index <= teaching['slider3-4b']) else 0 for action_index in range(21)] # Negative hip motion for swinging leg.
            for joint1_value in range(21):
                    for joint2_value in range(21):
                        self.action_mask[3][gait-1][joint1_value][joint2_value] = [1 if (action_index >= teaching['slider3-1a'] and action_index <= teaching['slider3-1b']) else 0 for action_index in range(21)] # Slight Positive knee motion for planted leg.
            for joint1_value in range(21):
                for joint2_value in range(21):
                    for joint3_value in range(21):
                        self.action_mask[4][gait-1][joint1_value][joint2_value][joint3_value] = [1 if (action_index >= teaching['slider3-2a'] and action_index <= teaching['slider3-2b']) else 0 for action_index in range(21)] # Negative knee motion for swinging leg.
        pass

class ActionMaskCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def remap(self, x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def translate_ui(self, ui):
        for key, value in ui.items():
            ui[key] = self.remap(value, -10, 10, 0, 20)
        return ui

    def __init__(self, verbose=2, locals=locals, globals=globals):
        super(ActionMaskCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        self.n_calls = 0  # type: int
        self.num_timesteps = 0  # type: int
        # local and global variables
        self.locals = None  # type: Dict[str, Any]
        self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

        # Set all actions to valid.
        self.action_space = MultiDiscrete([3, 21, 21, 21, 21])
        self.action_mask = mask(self.action_space)
        self.action_mask[0] = [1, 0, 0]

        states = ['lift_leg', 'fuzzy_transition_1', 'plant_leg', 'switch_leg', 'fuzzy_transition_2']
        # Fuzzy transition 1 determines how long to lift swinging leg before planting it. 
        # Fuzzy transition 2 determines how long to swing switched leg before lifing it. 
        self.gait = Walk()
        self.gait.num_timesteps = self.num_timesteps
        self.gait.action_mask = mask(self.action_space)
        self.gait.action_mask[0] = [1, 0, 0]
        self.gait.swinging_leg = 'right'
        self.gait.terminal = False
        self.gait.start = True
        self.gait.step_count = 0
        self.gait.step_flag = False
        machine = Machine(self.gait, states=states, send_event=True, initial='start')
        
        machine.add_transition('move', 'start', 'fuzzy_transition_1', conditions='is_swinging_leg_lifted', prepare='set_mask_start', before=['set_mask_start', 'reset_iter_counter', 'set_fuzzy_decision1'])
        machine.add_transition('move', 'lift_leg', 'fuzzy_transition_1', conditions='is_swinging_leg_lifted', unless='is_start', prepare='set_mask_gait_1', before=['set_mask_gait_1', 'reset_iter_counter', 'set_fuzzy_decision1'])
        machine.add_transition('move', 'fuzzy_transition_1', 'plant_leg', conditions='done_lifting_swinging_leg', prepare=['set_mask_gait_1', 'set_fuzzy_decision1'], before=['reset_iter_counter', 'set_mask_gait_2'])
        machine.add_transition('move', 'plant_leg', 'switch_leg', conditions='is_swinging_leg_planted', prepare='set_mask_gait_2', before=['reset_iter_counter', 'switch_legs', 'set_mask_gait_3', 'increment_step_count'])
        machine.add_transition('move', 'switch_leg', 'fuzzy_transition_2', conditions='is_swinging_leg_leading', prepare='set_mask_gait_3', before=['set_mask_gait_3', 'set_fuzzy_decision2'])
        machine.add_transition('move', 'fuzzy_transition_2', 'lift_leg', conditions='start_lifting_swinging_leg', prepare=['set_mask_gait_3', 'set_fuzzy_decision2'], before=['reset_iter_counter', 'set_mask_gait_1'])
        #machine.add_transition('move', 'fuzzy_transition_2', 'plant_leg', conditions='start_planting_swinging_leg', prepare=['set_mask_gait_3', 'set_fuzzy_decision2'], before=['reset_iter_counter', 'set_mask_gait_2', 'increment_step_count'])

        machine.add_transition('reset', 'fuzzy_transition_1', 'start', before=['reinstate', 'reset_step_counter', 'set_mask_gait_1'])
        machine.add_transition('reset', 'plant_leg', 'start', before=['reinstate', 'reset_step_counter', 'set_mask_gait_1'])
        machine.add_transition('reset', 'switch_leg', 'start', before=['reinstate', 'reset_step_counter', 'set_mask_gait_1'])
        machine.add_transition('reset', 'fuzzy_transition_2', 'start', before=['reinstate', 'reset_step_counter', 'set_mask_gait_1'])
        machine.add_transition('reset', 'lift_leg', 'start', before=['reinstate', 'reset_step_counter', 'set_mask_gait_1'])
        #print('callback init')
        self.ui = UI()
        self.ui.event, self.ui.values = self.ui.window.read()
        if self.ui.event == 'Cancel':
            self.ui.window.close()
        elif self.ui.event == 'Reset':
           self.training_env.env_method('reset')
        elif self.ui.event in (None, 'Submit'):
           self.gait.teaching = self.translate_ui(self.ui.values) 
           #self.ui.window.close()
    

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        print('train start')
        
        # Query the state to pass to def facts.
        sim = self.training_env.env_method('get_state')
        """
        #print(sim[0]['state'][8], sim[0]['state'][13], sim[0]['action'])
        """
        left_leg = {'side': 'left', 'contact': sim[0]['state'][8], 'position': sim[0]['legs'][1].position[0], 'hip_angle': sim[0]['state'][4], 'knee_angle': sim[0]['state'][6], 'hip_height': sim[0]['state'][26], 'knee_height': sim[0]['state'][24]}
        right_leg = {'side': 'right', 'contact': sim[0]['state'][13], 'position': sim[0]['legs'][3].position[0], 'hip_angle': sim[0]['state'][9], 'knee_angle': sim[0]['state'][11], 'hip_height': sim[0]['state'][27], 'knee_height': sim[0]['state'][25]}

        # Read from Machine Teaching UI
        self.ui.event, self.ui.values = self.ui.window.read(timeout=0)
        self.gait.teaching = self.translate_ui(self.ui.values)
        if self.ui.event == 'Cancel':
            self.ui.window.close()
        elif self.ui.event == 'Reset':
           self.training_env.env_method('reset')
        """
        elif self.ui.event in (None, 'Submit'):
           self.gait.teaching = self.translate_ui(self.ui.values) 
        """

        self.gait.move(action=sim[0]['action'], left_position=left_leg['position'], left_contact=left_leg['contact'], \
            right_position=right_leg['position'], right_contact=right_leg['contact'], \
                left_hip_angle=left_leg['hip_angle'], left_knee_angle=left_leg['knee_angle'], \
                    right_hip_angle=right_leg['hip_angle'], right_knee_angle=right_leg['knee_angle'], \
                        left_hip_height=left_leg['hip_height'], right_hip_height=right_leg['hip_height'], \
                            left_knee_height=left_leg['knee_height'], right_knee_height=right_leg['knee_height'], teaching=self.gait.teaching)
        
        self.action_mask = self.gait.action_mask
        #self.action_mask = mask(self.action_space)
        #self.action_mask[0] = [1, 0, 0]
        #self.gait.action_mask = self.action_mask
        #self.training_env.env_method('set_infos', self.gait.action_mask)
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
        
        # Declare facts from environment.  
        self.gait.num_timesteps += 1
        sim = self.training_env.env_method('get_state')
        
        #print(sim[0]['state'][8], sim[0]['state'][13], sim[0]['action'])
        left_leg = {'side': 'left', 'contact': sim[0]['state'][8], 'position': sim[0]['legs'][0].position[0], 'hip_angle': sim[0]['state'][4], 'knee_angle': sim[0]['state'][6], 'hip_height': sim[0]['state'][26], 'knee_height': sim[0]['state'][24]}
        right_leg = {'side': 'right', 'contact': sim[0]['state'][13], 'position': sim[0]['legs'][1].position[0], 'hip_angle': sim[0]['state'][9], 'knee_angle': sim[0]['state'][11], 'hip_height': sim[0]['state'][27], 'knee_height': sim[0]['state'][25]}
        #print(left_leg, right_leg)
        #sys.exit()

        # Read from Machine Teaching UI
        #"""
        self.ui.event, self.ui.values = self.ui.window.read(timeout=0)
        self.gait.teaching = self.translate_ui(self.ui.values)
        if self.ui.event == 'Cancel':
            self.ui.window.close()
        elif self.ui.event == 'Reset':
           self.training_env.env_method('reset')
        #"""
        """
        elif self.ui.event in (None, 'Submit'):
           self.gait.teaching = self.translate_ui(self.ui.values) 
        """
        
        self.gait.move(action=sim[0]['action'], left_position=left_leg['position'], left_contact=left_leg['contact'], \
            right_position=right_leg['position'], right_contact=right_leg['contact'], \
                left_hip_angle=left_leg['hip_angle'], left_knee_angle=left_leg['knee_angle'], \
                    right_hip_angle=right_leg['hip_angle'], right_knee_angle=right_leg['knee_angle'], \
                        left_hip_height=left_leg['hip_height'], right_hip_height=right_leg['hip_height'], \
                            left_knee_height=left_leg['knee_height'], right_knee_height=right_leg['knee_height'], teaching=self.gait.teaching)
        self.action_mask = self.gait.action_mask
        #self.training_env.env_method('set_infos', self.gait.action_mask)
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
        """
        self.action_mask = self.gait.action_mask = []
        self.training_env.env_method('set_infos', self.gait.action_mask)
        if self.gait.state != 'lift_leg':
            self.gait.reset()
        print('training end: ', self.gait.state)
        """
        pass

# Callbacks for the Brains
callback = ActionMaskCallback()
  
brain = PPO2(MlpPolicy, env, verbose=2, tensorboard_log="walker3/") # A2C is the other option: tensorboard_log="walker/", nminibatches=1 , tensorboard_log="walker2/"
#brain.load("position_reward_brain")
brain.learn(100000, callback=callback)


brain.save("position_reward_brain")
"""
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
"""

