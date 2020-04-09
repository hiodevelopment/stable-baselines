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
from stable_baselines.common.callbacks import BaseCallback
from curiosity_mask.util import create_dummy_action_mask as mask

from experta import *
from experta.fact import *
import schema

env = DummyVecEnv([walker.BipedalWalker])

class Leg(Fact):
    side = Field(schema.Or("left", "right"), mandatory=True)
    position = Field(float)
    knee = Field(float)  # Angle
    hip = Field(float)  # Angle
    orientation = Field(schema.Or("leading", "trailing"))
    contact = Field(bool)
    function = Field(schema.Or("planted", "swinging", "lifting"))
    gait = Field(schema.Or(1, 2, 3), mandatory=True)
    pass

class Orchestration(Fact):
    condition = Field(schema.Or("deterministic", "fuzzy"), default='deterministic', mandatory=True)
    event = Field(schema.Or("selection", "blank"))
    gait = Field(schema.Or(1, 2, 3), mandatory=True, default=1)
    concept = Field(schema.Or(1, 2, 3))
    pass

class Machine_Teaching(KnowledgeEngine):

    @DefFacts()
    def enivronment_definitions(self, action, left_position, left_contact, right_position, right_contact, left_hip_angle, left_knee_angle, right_hip_angle, right_knee_angle, left_hip_height, right_hip_height, left_knee_height, right_knee_height):
        #Sprint('defining initial facts')
        yield Leg(side='left', gait=1, position=left_position, contact=bool(left_contact), hip=left_hip_angle, knee_angle=left_knee_angle, hip_height=left_hip_height, knee_height=left_knee_height)
        yield Leg(side='right', gait=1, position=right_position, contact=bool(right_contact), hip=right_hip_angle, knee_angle=right_knee_angle, hip_height=right_hip_height, knee_height=left_knee_height)
        #yield Orchestration(gait=1, event='blank', condition='deterministic')
        #print(left_position, right_position, bool(left_contact), bool(right_contact))
        yield Fact(concept=1) # self.action
        print('deffacts, action = ', action, self.valid_actions[0], self.action)
        pass
    
    ##### Heuristic Machine Teaching Strategies
    
    @Rule(Fact(concept=1), AS.left_leg << Leg(side=L('left'), position=MATCH.p1),
        AS.right_leg << Leg(side=L('right'), position=MATCH.p2, contact=MATCH.contact, hip_height=MATCH.hip_height, knee_height=MATCH.knee_height)
        )
    def set_legs(self, left_leg, right_leg, p1, p2, hip_height, knee_height, contact):   # The leg that is further ahead is leading.
 
        """
        if p1 > p2:
            self.declare(Fact(side='left', orientation='leading'))
            self.declare(Fact(side='right', orientation='trailing'))
            self.declare(Fact(side='left', function='swinging', contact=contact, hip_height=hip_height, knee_height=knee_height))
            self.declare(Fact(side='right', function='planted'))
        else:
            self.declare(Fact(side='right', orientation='leading'))
            self.declare(Fact(side='left', orientation='trailing'))
            self.declare(Fact(side='right', function='swinging', contact=contact, hip_height=hip_height, knee_height=knee_height))
            self.declare(Fact(side='left', function='planted'))
        """

        self.declare(Fact(side='right', orientation='leading'))
        self.declare(Fact(side='left', orientation='trailing'))
        self.declare(Fact(side='right', function='swinging', contact=contact, hip_height=hip_height, knee_height=knee_height))
        self.declare(Fact(side='left', function='planted'))
        self.declare(Fact(concept=1))
        
        #print('Set Legs Activated')
        pass


    """
    @Rule(AS.leg <<
        Leg(side=L('left'), position=MATCH.p1),
        Leg(side=L('right'), position=MATCH.p2,),
        TEST(lambda p1, p2: p1 if (p1>p2) else p2)
        )
    def determine_leading_leg(self, leg, p1, p2):   # The leg that is further ahead is leading.
        self.duplicate(leg, orientation='leading')
        #print('leading leg set: ', leg['side'], leg['position'], p1, p2)
        pass
    
    @Rule(AS.leg << 
        Leg(side=L('right'), position=MATCH.p1),
        Leg(side=L('left'), position=MATCH.p2,),
        TEST(lambda p1, p2: p1 if (p1>p2) else p2)
        )
    def determine_trailing_leg(self, leg, p1, p2):   # The leg that is further behind is trailing.
        self.duplicate(leg, orientation='trailing')
        #print('trailing leg set: ', leg['side'], leg['position'], p1, p2)
        pass

    @Rule(AS.planted_leg << Leg(orientation=L('trailing')), AS.swinging_leg << Leg(orientation=L('leading')))
    def define_planted_vs_swinging(self, planted_leg, swinging_leg):    # Planted leg is [set conditions based on heuristic].
        self.duplicate(planted_leg, function='planted')
        self.duplicate(swinging_leg, function='swinging')
        #print(self.agenda)
        #print('planted leg set: ', planted_leg['side'], planted_leg['position'], planted_leg['orientation'])
        #print('swinging leg set: ', swinging_leg['side'], swinging_leg['position'], swinging_leg['orientation'])
        pass
    """
    
    ##### Transition: Gait Phase 1 -> 2: After knee is raised, bring leg back to the ground.
    @Rule(AS.orchestration << Fact(concept=1), NOT(Fact(concept=2)), NOT(Fact(concept=3)), Fact(function='swinging', side=MATCH.side_swinging, hip_height=MATCH.hip, knee_height=MATCH.knee),
           TEST(lambda knee, hip: hip > 0.85*knee), salience=1)
    def transition_1_2(self, orchestration):
        self.valid_actions[0] = [0, 1, 0]
        self.declare(Fact(concept=2))
        self.action = 2
        self.retract(orchestration)
        """
        if self.action[0] == 1: # Transition
            print(self.facts)
            self.valid_actions[0] = [0, 1, 0]
            self.declare(Fact(concept=2))
            self.retract(orchestration)
            print('gait 1 -> 2 transition complete', self.facts)
            #sys.exit()
        else: # Agent decided to stay in gait phase 1.  Give it the option to transition next. 
            self.valid_actions[0] = [1, 1, 0]
            pass
        """
        print('Fuzzy condition between gait phase 1 and 2: ', self.action)
    
    ###### Concept: Gait Phase 1 #####
    @Rule(Fact(concept=1), Fact(function='swinging', side=MATCH.side_swinging, hip_height=MATCH.hip_height, knee_height=MATCH.knee_height)) # Deterministic Phase 1
    def concept_1(self, side_swinging, hip_height, knee_height):   
        gait = 1
        # Each concept rule needs to mask the gait and flip bits for control concept.
        self.valid_actions[0] = [1, 0, 0]  # Limit selection to gait phase 1.

        # Flip bits for joint modifications. (Left Hip, Right Hip, Left Knee, Right Knee)
        self.mask_left_hip = self.valid_actions[1][gait-1] = [1 if action_index == 12 else 0 for action_index in range(21)] # Positive hip motion for planted leg.
        for joint1_value in range(21):
            self.mask_left_knee = self.valid_actions[2][gait-1][joint1_value] = [1 if action_index >= 10 else 0 for action_index in range(21)] # Negative hip motion for swinging leg.
        for joint1_value in range(21):
                for joint2_value in range(21):
                    self.mask_right_hip = self.valid_actions[3][gait-1][joint1_value][joint2_value] = [1 if action_index == 10 else 0 for action_index in range(21)] # Slight Positive knee motion for planted leg.
        for joint1_value in range(21):
            for joint2_value in range(21):
                for joint3_value in range(21):
                    self.mask_right_knee = self.valid_actions[4][gait-1][joint1_value][joint2_value][joint3_value] = [1 if action_index == 0 else 0 for action_index in range(21)] # Negative knee motion for swinging leg.
    
        print('Concept 1 Activated')
        pass

    ##### Transition: Gait Phase 2 -> 3: When swinging leg makes contact with the ground, plant leg and push off.
    @Rule(AS.orchestration << Fact(concept=2), Fact(function='swinging', contact=MATCH.contact),
            AS.swinging_right << Fact(function='swinging', side='right'), AS.planted_left << Fact(function='planted', side='left'),
            TEST(lambda contact: contact), salience=2)
    def transition_2_3(self, orchestration, swinging_right, planted_left):
        self.valid_actions[0] = [0, 0, 1]
        self.action = 3
        self.retract(orchestration)
        self.retract(swinging_right)
        self.retract(planted_left)
        self.declare(Fact(side='left', function='swinging'))
        self.declare(Fact(side='right', function='planted'))
        self.declare(Fact(concept=3))
        print('Fuzzy condition between gait phase 2 and 3: ', self.action)

    ###### Concept: Gait Phase 2 #####
    @Rule(Fact(concept=2), salience=2) # Deterministic Phase 2
    def concept_2(self): 
        gait = 2
        # Each concept rule needs to mask the gait and flip bits for control concept.
        #self.valid_actions[0] = [0, 1, 0]  # Limit selection to gait phase 2.

        # Flip bits for joint modifications. (Left Hip, Right Hip, Left Knee, Right Knee)
        self.mask_left_hip = self.valid_actions[1][gait-1] = [1 if action_index == 12 else 0 for action_index in range(21)] # Positive hip motion for planted leg.
        for joint1_value in range(21):
            self.mask_left_knee = self.valid_actions[2][gait-1][joint1_value] = [1 if action_index == 6 else 0 for action_index in range(21)] # Positive hip motion for swinging leg.
        for joint1_value in range(21):
                for joint2_value in range(21):
                    self.mask_right_hip = self.valid_actions[3][gait-1][joint1_value][joint2_value] = [1 if action_index == 12 else 0 for action_index in range(21)] # Slight Positive knee motion for planted leg.
        for joint1_value in range(21):
            for joint2_value in range(21):
                for joint3_value in range(21):
                    self.mask_right_knee = self.valid_actions[4][gait-1][joint1_value][joint2_value][joint3_value] = [1 if action_index == 0 else 0 for action_index in range(21)] # Negative knee motion for swinging leg.
    
        print('Concept 2 Activated')
        pass

    ###### Concept: Gait Phase 3 ##### 
    @Rule(Fact(concept=3), salience=3) # Deterministic Phase 3
    def concept_3(self): 
        gait = 3
        # Each concept rule needs to mask the gait and flip bits for control concept.
        self.valid_actions[0] = [0, 0, 1]  # Limit selection to gait phase 3.

        # Flip bits for joint modifications. (Left Hip, Right Hip, Left Knee, Right Knee)
        self.mask_left_hip = self.valid_actions[1][gait-1] = [1 if action_index == 5 else 0 for action_index in range(21)] # Positive hip motion for planted leg.
        for joint1_value in range(21):
            self.mask_left_knee = self.valid_actions[2][gait-1][joint1_value] = [1 if action_index == 16 else 0 for action_index in range(21)] # Positive hip motion for planted leg.
        for joint1_value in range(21):
                for joint2_value in range(21):
                    self.mask_right_hip = self.valid_actions[3][gait-1][joint1_value][joint2_value] = [1 if action_index == 12 else 0 for action_index in range(21)] # Slight Positive knee motion for planted leg.
        for joint1_value in range(21):
            for joint2_value in range(21):
                for joint3_value in range(21):
                    self.mask_right_knee = self.valid_actions[4][gait-1][joint1_value][joint2_value][joint3_value] = [1 if action_index == 10 else 0 for action_index in range(21)] # Negative knee motion for swinging leg.
    
        print('Concept 3 Activated')
        pass

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
        self.action_space = MultiDiscrete([3, 21, 21, 21, 21])
        self.engine.action = 1 # gait phase instead of reading list, starts with 1, [] 
        self.engine.valid_actions = mask(self.action_space)
        self.engine.valid_actions[0] = [1, 0, 0] # Set to start in gait phase 1.
        self.engine.value_list = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        #self.engine.valid_actions[0] = [1, 0, 0]
        self.engine.mask_left_hip = []
        self.engine.mask_left_knee = []
        self.engine.mask_right_hip = []
        self.engine.mask_right_knee = []
        self.engine.mask_width = 1
        self.engine.sim = {}
        self.engine.possible_states = []
        #print('callback init')

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        #print('train start')
        self.engine.valid_actions = mask(self.action_space)
        self.engine.valid_actions[0] = [1, 0, 0] # Set to start in gait phase 1.
        self.engine.action = 1
        
        # Query the state to pass to def facts.
        sim = self.training_env.env_method('get_state')
        #print(sim[0]['state'][8], sim[0]['state'][13], sim[0]['action'])
        left_leg = {'side': 'left', 'contact': sim[0]['state'][8], 'position': sim[0]['legs'][1].position[0], 'hip_angle': sim[0]['state'][4], 'knee_angle': sim[0]['state'][6], 'hip_height': sim[0]['state'][26], 'knee_height': sim[0]['state'][24]}
        right_leg = {'side': 'right', 'contact': sim[0]['state'][13], 'position': sim[0]['legs'][3].position[0], 'hip_angle': sim[0]['state'][9], 'knee_angle': sim[0]['state'][11], 'hip_height': sim[0]['state'][27], 'knee_height': sim[0]['state'][25]}
        
        self.engine.reset(action=sim[0]['action'], left_position=left_leg['position'], left_contact=left_leg['contact'], \
            right_position=right_leg['position'], right_contact=right_leg['contact'], \
                left_hip_angle=left_leg['hip_angle'], left_knee_angle=left_leg['knee_angle'], \
                    right_hip_angle=right_leg['hip_angle'], right_knee_angle=right_leg['knee_angle'], \
                        left_hip_height=left_leg['hip_height'], right_hip_height=right_leg['hip_height'], \
                            left_knee_height=left_leg['knee_height'], right_knee_height=right_leg['knee_height'])
        #self.mask(self.action_space)()
        #self.engine.action = sim[0]['action']
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
        
        # Declare facts from environment. 
        sim = self.training_env.env_method('get_state')
        #print(sim[0]['state'][8], sim[0]['state'][13], sim[0]['action'])
        left_leg = {'side': 'left', 'contact': sim[0]['state'][8], 'position': sim[0]['legs'][0].position[0], 'hip_angle': sim[0]['state'][4], 'knee_angle': sim[0]['state'][6], 'hip_height': sim[0]['state'][26], 'knee_height': sim[0]['state'][24]}
        right_leg = {'side': 'right', 'contact': sim[0]['state'][13], 'position': sim[0]['legs'][1].position[0], 'hip_angle': sim[0]['state'][9], 'knee_angle': sim[0]['state'][11], 'hip_height': sim[0]['state'][27], 'knee_height': sim[0]['state'][25]}
        #print(left_leg, right_leg)
        #sys.exit()

        self.engine.reset(action=sim[0]['action'], left_position=left_leg['position'], left_contact=left_leg['contact'], \
            right_position=right_leg['position'], right_contact=right_leg['contact'], \
                left_hip_angle=left_leg['hip_angle'], left_knee_angle=left_leg['knee_angle'], \
                    right_hip_angle=right_leg['hip_angle'], right_knee_angle=right_leg['knee_angle'], \
                        left_hip_height=left_leg['hip_height'], right_hip_height=right_leg['hip_height'], \
                            left_knee_height=left_leg['knee_height'], right_knee_height=right_leg['knee_height'])
        #print('joint status: ', right_leg['hip_height'], right_leg['knee_height'], left_leg['hip_height'], left_leg['knee_height'], right_leg['contact'])
        #self.valid_actions = mask(self.action_space)
        #print('Before engine run: ', self.engine.facts)
        #print('Agenda: ', self.engine.agenda)
        #self.engine.action = sim[0]['action']
        self.engine.run()
        #print('After engine run: ', self.engine.facts)
        #print('Agenda: ', self.engine.agenda)
        #print('done run')
        #sys.exit()
        #print('engine run', tf.convert_to_tensor(np.expand_dims(self.engine.valid_actions[3], 0)).get_shape().as_list())
        #self.action_mask = self.engine.valid_actions
        self.training_env.env_method('set_infos', self.engine.valid_actions)

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
control_brain_callback = ActionMaskCallback()
#selector_brain_callback = SelectionMaskCallback()
  
control_brain = A2C(MlpPolicy, env, verbose=2) # PPO2 is the other option: tensorboard_log="walker/", nminibatches=1
control_brain.learn(250000, callback=control_brain_callback)

#selector_brain = A2C(MlpPolicy, env, verbose=2) # PPO2 is the other option: tensorboard_log="walker/", nminibatches=1
#selector_brain.learn(250000, callback=selector_brain_callback)

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