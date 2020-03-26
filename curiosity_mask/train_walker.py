import os
import sys
import warnings
import numpy as np
import tensorflow as tf

import stable_baselines.common.walker_action_mask as walker
from stable_baselines import PPO2, A2C
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.callbacks import BaseCallback

from experta import *
from experta.fact import *
import schema

env = DummyVecEnv([walker.BipedalWalker])

class Leg(Fact):
    side = Field(schema.Or("left", "right"), mandatory=True)
    position = Field(float)
    orientation = Field(schema.Or("leading", "trailing"))
    contact = Field(schema.Or("1.0", "0.0", "blank"))
    function = Field(schema.Or("planted", "swinging", "lifting"))
    pass

class Machine_Teaching(KnowledgeEngine):

    @DefFacts()
    def enivronment_definitions(self, left_position, left_contact, right_position, right_contact):
        print('defining initial facts')
        yield Leg(side='left', position=left_position, contact=str(left_contact))
        yield Leg(side='right', position=right_position, contact=str(right_contact))
        pass
            
    ##### Heuristic Machine Teaching Strategies
    # The leg that is further ahead is leading.
    @Rule(AS.leg <<
        Leg(side=L('left'), position=MATCH.p1),
        Leg(side=L('right'), position=MATCH.p2,),
        TEST(lambda p1, p2: p1 if (p1>p2) else p2)
        )
    #@Rule(AS.leg << Leg(side=L('left')))
    def determine_leading_leg(self, leg, p1, p2):
        self.duplicate(leg, orientation='leading')
        #print('leading leg set: ', leg['side'], leg['position'], p1, p2)
        pass
    
    # The leg that is further behind is trailing.
    @Rule(AS.leg << 
        Leg(side=L('right'), position=MATCH.p1),
        Leg(side=L('left'), position=MATCH.p2,),
        TEST(lambda p1, p2: p1 if (p1>p2) else p2)
        )
    #@Rule(AS.leg << Leg(side=L('right')))
    def determine_trailing_leg(self, leg, p1, p2):
        self.duplicate(leg, orientation='trailing')
        #print('trailing leg set: ', leg['side'], leg['position'], p1, p2)
        pass

    # Lift leading leg (more than swinging) when both feet are on the ground.. 
    @Rule(AS.leg << Leg(orientation=L('leading')) & Leg(contact=L('1.0')))
    def define_lifting(self, leg):
        self.duplicate(leg, function='lifting')
        #print('lifting leg set: ', leg['side'], leg['position'], leg['orientation'])
        pass
    """
    # Swinging leg is leading and not touching the ground. 
    @Rule(AS.leg << Leg(orientation=L('leading')) & Leg(contact=L('0.0')))
    def define_swinging(self, leg):
        self.duplicate(leg, function='swinging')
        #print('swinging leg set: ', leg['side'], leg['position'], leg['orientation'])
        pass
    """
    # Planted leg is trailing and in contact with the ground. , contact=1.0
    @Rule(AS.leg << Leg(orientation=L('trailing')))
    def define_planted(self, leg):
        self.duplicate(leg, function='planted')
        #print('planted leg set: ', leg['side'], leg['position'], leg['orientation'])
        pass

    # Plant right leg, lift left leg.
    @Rule(Leg(function='planted', side='right') & Leg(function='lifting', side='left'))
    def left_plant_right_lift(self):
        """
        Right leg planted, left leg lifting. 
        """
        # act on mask
        a = b = c = d = 21
        self.mask_right_hip = [[[1 if action_index > 10 else 0 for action_index in range(c)] for y in range(b)] for x in range(a)]
        self.valid_actions[2] = self.mask_right_hip

        self.mask_left_hip = [1 if action_index <= 6 else 0 for action_index in range(a)] # Equivalent of lift leg higher (than swinging)
        self.valid_actions[0] = self.mask_left_hip
        
        print('left leg planted, right leg lifting')
        return self.valid_actions

    @Rule(Leg(function='planted', side='left'))
    def planted_leg(self):
        """
        Modify the mask for the planted leg.  Hip rotates forward.  Left hip action mask. 
        """
        # act on mask
        a = b = c = d = 21
        self.mask_left_hip = [1 if action_index > 10 else 0 for action_index in range(a)]
        self.valid_actions[0] = self.mask_left_hip
        print('planted left rule')
        return self.valid_actions

    @Rule(Leg(function='planted', side='right'))
    def planted_leg(self):
        """
        Modify the mask for the planted leg.  Hip rotates forward.  Right hip action mask. 
        """
        # act on mask
        a = b = c = d = 21
        self.mask_right_hip = [[[1 if action_index > 10 else 0 for action_index in range(c)] for y in range(b)] for x in range(a)]
        self.valid_actions[2] = self.mask_right_hip
        print('planted right rule')
        return self.valid_actions

    @Rule(Leg(function='swinging', side='left'))
    def swinging_leg(self):
        """
        Modify the mask for the swinging leg.  Hip rotates backward. For now, raise.  Left hip action mask.
        """
        # act on mask
        a = b = c = d = 21
        self.mask_left_hip = [1 if action_index <= 10 else 0 for action_index in range(a)]
        self.valid_actions[0] = self.mask_left_hip

        return self.valid_actions

    @Rule(Leg(function='swinging', side='right'))
    def swinging_leg(self):
        """
        Modify the mask for the swinging leg.  Hip rotates backward. For now, raise.  Right hip action mask.
        """
        # act on mask
        a = b = c = d = 21
        self.mask_right_hip = [[[1 if action_index <= 10 else 0 for action_index in range(c)] for y in range(b)] for x in range(a)]
        self.valid_actions[2] = self.mask_right_hip

        return self.valid_actions

    @Rule(Leg(function='lifting', side='left'))
    def lifting_leg(self):
        """
        Modify the mask for the lifting leg.  Hip rotates backward. For now, raise.  Left hip action mask.
        """
        # act on mask
        a = b = c = d = 21
        self.mask_left_hip = [1 if action_index <= 6 else 0 for action_index in range(a)] # Equivalent of lift leg higher (than swinging)
        self.valid_actions[0] = self.mask_left_hip
        print('lifting left rule')
        return self.valid_actions

    @Rule(Leg(function='lifting', side='right'))
    def lifting_leg(self):
        """
        Modify the mask for the planted leg.  Hip rotates backward. For now, raise.  Right hip action mask.
        """
        # act on mask
        a = b = c = d = 21
        self.mask_right_hip = [[[1 if action_index <= 6 else 0 for action_index in range(c)] for y in range(b)] for x in range(a)] # Equivalent of lift leg higher (than swinging)
        self.valid_actions[2] = self.mask_right_hip
        print('lifting right rule')
        return self.valid_actions

class ActionMaskCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def reset_mask(self):

        # Set all actions to valid.
        def generate_mask(a,b,c,d):
            """
            action_mask = []

            # Two elements
            action_mask1 = [1 for x in range(a)]

            # Two lists of 3 elements each. 
            action_mask2 = [[1]*b for x in range(a)]

            #Two lists with 3  lists with 4 elements each.
            action_mask3 = [[[1]*c for y in range(b)] for x in range(a)]

            #Two lists, with 3 lists, with 4 lists with 5 elements each
            action_mask4 = [[[[1]*d for z in range(c)]for y in range(b)] for x in range(a)]
            
            return [action_mask1, action_mask2, action_mask3, action_mask4]
            """
            action_mask = []

            # Two elements
            action_mask1 = [1 if action_index == 0 else 0 for action_index in range(a)]

            # Two lists of 3 elements each. 
            action_mask2 = [[1 if action_index == 0 else 0 for action_index in range(b)] for x in range(a)]

            #Two lists with 3  lists with 4 elements each.
            action_mask3 = [[[1 if action_index == 0 else 0 for action_index in range(c)] for y in range(b)] for x in range(a)]

            #Two lists, with 3 lists, with 4 lists with 5 elements each
            action_mask4 = [[[[1 if action_index == 0 else 0 for action_index in range(d)] for z in range(c)]for y in range(b)] for x in range(a)]

            return [action_mask1, action_mask2, action_mask3, action_mask4]

        self.engine.valid_actions = generate_mask(21,21,21,21)
        self.engine.mask_left_hip = self.engine.valid_actions[0]
        self.engine.mask_left_knee = self.engine.valid_actions[1]
        self.engine.mask_right_hip = self.engine.valid_actions[2]
        self.engine.mask_right_knee = self.engine.valid_actions[3]

        pass

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
        self.engine = Machine_Teaching()
        self.action_mask = None

        # Set all actions to valid.
        self.reset_mask()
        self.engine.mask_width = 1
        self.engine.sim = {}
        self.engine.possible_states = []
        #print('callback init')

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        #print('train start')
        
        # Query the state to pass to def facts.
        sim = self.training_env.env_method('get_state')
        print(sim[0]['state'][8], sim[0]['state'][13], sim[0]['action'])
        left_leg = {'side': 'left', 'contact': sim[0]['state'][8], 'position': sim[0]['legs'][0].position[0]}
        right_leg = {'side': 'right', 'contact': sim[0]['state'][13], 'position': sim[0]['legs'][1].position[0]}
        
        self.engine.reset(left_position=left_leg['position'], left_contact=left_leg['contact'], right_position=right_leg['position'], right_contact=right_leg['contact'])
        #self.reset_mask()
        self.engine.run()
        self.action_mask = self.engine.valid_actions
        self.training_env.env_method('set_infos', self.engine.valid_actions)

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
        # Widen the mask based on reward conditions. 
        """
        # Reward based criteria for widening the mask"
        reward = self.training_env.env_method('get_reward')[0]
        if reward > 60: # 60
            self.engine.mask_width = 2
        elif reward > 62: # 70
            self.engine.mask_width = 3
        elif reward > 65:
            self.engine.mask_width = 4
        """

        # Declare facts from environment. 
        sim = self.training_env.env_method('get_state')
        print(sim[0]['state'][8], sim[0]['state'][13], sim[0]['action'])
        left_leg = {'side': 'left', 'contact': sim[0]['state'][8], 'position': sim[0]['legs'][0].position[0]}
        right_leg = {'side': 'right', 'contact': sim[0]['state'][13], 'position': sim[0]['legs'][1].position[0]}
        #print(left_leg, right_leg)
        #sys.exit()

        # Filter the actions based on the states that will result from taking each of them. 
        """
        # Run through all the actions to generate the resulting state vectors.
        left_hip = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        left_knee = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        right_hip = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        right_knee = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

        self.possible_states = [[[[self.training_env.env_method('simulate', x, y, z, j) for j in right_knee] for z in right_hip]for y in left_knee] for x in left_hip]
        print('possible states: ', possible_states[0][0][0][0][0])  # x, y, z, j then [0]
        sys.exit()
        """
        def gen(a,b,c,d):

            action_mask = []

            # Two elements
            action_mask1 = [1 if action_index == 0 else 0 for action_index in range(a)]

            # Two lists of 3 elements each. 
            action_mask2 = [[1 if action_index == 0 else 0 for action_index in range(b)] for x in range(a)]

            #Two lists with 3  lists with 4 elements each.
            action_mask3 = [[[1 if action_index == 0 else 0 for action_index in range(c)] for y in range(b)] for x in range(a)]

            #Two lists, with 3 lists, with 4 lists with 5 elements each
            action_mask4 = [[[[1 if action_index == 0 else 0 for action_index in range(d)] for z in range(c)]for y in range(b)] for x in range(a)]

            return [action_mask1, action_mask2, action_mask3, action_mask4]

        #test_mask = gen(21,21,21,21)
        #print(test_mask)
        #print(self.engine.facts)
        self.engine.reset(left_position=left_leg['position'], left_contact=left_leg['contact'], right_position=right_leg['position'], right_contact=right_leg['contact'])
        #self.reset_mask()
        #print(self.engine.facts)
        self.engine.run()
        print(self.engine.facts)
        #print('done run')
        #sys.exit()
        #print('engine run', tf.convert_to_tensor(np.expand_dims(self.engine.valid_actions[3], 0)).get_shape().as_list())
        #self.action_mask = test_mask #self.engine.valid_actions
        self.training_env.env_method('set_infos', self.engine.valid_actions)

        # Log variables to Tensorboard as needed. 
        """
        # Log scalar value to Tensorboard
        co2 = o['CO2']
        energy = h['energy_elec']
        comfort = h['error_T_real']
        summary = tf.Summary(value=[tf.Summary.Value(tag='air_quality', simple_value=co2), tf.Summary.Value(tag='energy', simple_value=energy), tf.Summary.Value(tag='comfort', simple_value=comfort)])
        #summary2 = tf.Summary(value=[tf.Summary.Value(tag='energy', simple_value=energy)])
        #summary3 = tf.Summary(value=[tf.Summary.Value(tag='comfort', simple_value=comfort)])
        self.locals['writer'].add_summary(summary, self.num_timesteps)
        """
        
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

callback = ActionMaskCallback()

#model = PPO2(MlpPolicy, env, verbose=2, tensorboard_log="walker/", nminibatches=1)
#model = PPO2(MlpPolicy, env, verbose=2, nminibatches=1)  
model = A2C(MlpPolicy, env, verbose=2)
model.learn(250000, callback=callback)

"""
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
"""