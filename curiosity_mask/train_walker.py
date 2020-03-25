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

n_steps = 0

class Leg(Fact):
    side = Field(schema.Or("left", "right"), mandatory=True)
    position = Field(float, mandatory=True)
    orientation = Field(schema.Or("leading", "trailing"))
    contact = Field(schema.Or("1.0", "0.0", "blank"), mandatory=True)
    function = Field(schema.Or("planted", "swinging"))

class Machine_Teaching(KnowledgeEngine):
    """
    @DefFacts()
    def enivronment_definitions(self):
        pass
    """
            
    # Heuristic Machine Teaching Strategies

    # The leg that is further ahead is leading.
    @Rule(
        Leg(position=MATCH.p1),
        Leg(position=MATCH.p2,),
        TEST(lambda p1, p2: p1 if (p1>p2) else p2)
        )
    def determine_leading_leg(self):
        self.modify(leg, orientation='leading')
        print('leading leg set: ', self.side, self.orientation, self.function)
        pass

    # The leg that is further behind is trailing.
    @Rule(
        Leg(position=MATCH.p1),
        Leg(position=MATCH.p2,),
        TEST(lambda p1, p2: p1 if (p1<p2) else p2)
        )
    def determine_trailing_leg(self):
        self.modify(leg, orientation='trailing')
        print('trailing leg set: ', self.side, self.orientation, self.function)
        pass

    # Swinging leg is leading and not touching the ground. 
    @Rule(Fact(orientation='leading', contact=0.0))
    def define_swinging(self):
        self.modify(leg, function='swinging')
        print('swinging leg set: ', self.side, self.orientation, self.function)
        pass

    # Planted leg is trailing and in contact with the ground. 
    @Rule(Fact(orientation='trailing', contact=1.0))
    def define_planted():
        self.modify(leg, function='planted')
        print('planted leg set: ', self.side, self.orientation, self.function)
        pass

    @Rule(Fact(function='planted', side='left'))
    def planted_leg(direction):
        """
        Modify the mask for the planted leg.  Hip rotates forward.  Left hip action mask. 
        """

        # act on mask
        a = 21
        self.mask_left_hip = [1 if x > 10 else 0 for x in range(a)]
        self.valid_actions[0] = self.mask_left_hip

        return self.valid_actions

    @Rule(Fact(function='planted', side='right'))
    def planted_leg(direction):
        """
        Modify the mask for the planted leg.  Hip rotates forward.  Right hip action mask. 
        """
        # act on mask
        # valid actions =

        return self.valid_actions

    @Rule(Fact(function='swinging', side='left'))
    def swinging_leg():
        """
        Modify the mask for the planted leg.  Hip rotates backward. For now, raise.  Left hip action mask.
        """

        # act on mask
        a = 21
        self.mask_left_hip = [1 if x <= 10 else 0 for x in range(a)]
        self.valid_actions[0] = self.mask_left_hip

        return self.valid_actions

    @Rule(Fact(function='swinging', side='right'))
    def swinging_leg():
        """
        Modify the mask for the planted leg.  Hip rotates backward. For now, raise.  Right hip action mask.
        """

        # act on mask
        a, b = 21
        self.mask_right_hip = [[1 if y > 10 else 0 for y in range(b)] for x in range(a)]
        self.valid_actions[2] = self.mask_left_hip

        return self.valid_actions

class ActionMaskCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=2, locals=locals, globals=globals):
        super(ActionMaskCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
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

        # Set all actions to valid.
        def generate_mask(a,b,c,d):

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

        self.engine.valid_actions = generate_mask(21,21,21,21)
        self.engine.mask_left_hip = self.engine.valid_actions[0]
        self.engine.mask_left_knee = self.engine.valid_actions[1]
        self.engine.mask_right_hip = self.engine.valid_actions[2]
        self.engine.mask_right_knee = self.engine.valid_actions[3]
        self.engine.mask_width = 1
        self.engine.sim = {}
        self.engine.possible_states = []


    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        
        print('train start')
        """
        sim = self.training_env.env_method('get_state')
        """

        # Query the state to pass to def facts.
        #sim = self.training_env.env_method('get_state')
        #print(sim[0]['state'][8], sim[0]['state'][13], sim[0]['action'])

        self.engine.reset()

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """

        #print('rollout')
        pass


    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        
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
        
        self.engine.declare(*[Fact(side=x['side'], contact=x['contact'], position=x['position']) for x in (left_leg, right_leg)])
        self.engine.run()
        #print('engine run', tf.convert_to_tensor(np.expand_dims(self.engine.valid_actions[3], 0)).get_shape().as_list())
        self.model.policy.set_action_mask(self.model.policy, self.engine.valid_actions) # Set Action Mask


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