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
    def enivronment_definitions(self, left_position, left_contact, right_position, right_contact, left_hip_angle, left_knee_angle, right_hip_angle, right_knee_angle, left_hip_height, right_hip_height, left_knee_height, right_knee_height):
        #Sprint('defining initial facts')
        yield Leg(side='left', gait=1, position=left_position, contact=bool(left_contact), hip=left_hip_angle, knee_angle=left_knee_angle, hip_height=left_hip_height, knee_height=left_knee_height)
        yield Leg(side='right', gait=1, position=right_position, contact=bool(right_contact), hip=right_hip_angle, knee_angle=right_knee_angle, hip_height=right_hip_height, knee_height=left_knee_height)
        #yield Orchestration(gait=1, event='blank', condition='deterministic')
        print(left_position, right_position, left_contact, right_contact)
        sys.exit()
        pass
    
    ##### Heuristic Machine Teaching Strategies
    @Rule(AS.left_leg << Leg(side=L('left'), position=MATCH.p1),
        AS.right_leg << Leg(side=L('right'), position=MATCH.p2, hip_height=MATCH.hip_height, knee_height=MATCH.knee_height)
        )
    def set_legs(self, left_leg, right_leg, p1, p2, hip_height, knee_height):   # The leg that is further ahead is leading.
        
        if p1 > p2:
            self.declare(Fact(side='left', orientation='leading'))
            self.declare(Fact(side='right', orientation='trailing'))
            self.declare(Fact(side='left', function='swinging', hip_height=hip_height, knee_height=knee_height))
            self.declare(Fact(side='right', function='planted'))
        else:
            self.declare(Fact(side='right', orientation='leading'))
            self.declare(Fact(side='left', orientation='trailing'))
            self.declare(Fact(side='right', function='swinging'))
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
    
    @Rule(AS.orchestration << Orchestration(gait=L(1)) & NOT(Orchestration(event=L('selection'))), salience=5)
    def test(self, orchestration):   
        self.duplicate(orchestration, concept=1)
        print('Concept 1a Activated')
        pass

    @Rule(Orchestration(concept=L(1)))
    def test(self):   
        print('Concept 1b Activated')
        pass
    
    @Rule(Orchestration(gait=1), salience=1)
    def test(self):   
        self.declare(Fact(concept=1))
        print('Concept 1b Activated')
        pass
    """
    
    @Rule(AS.orchestration << Fact(concept=1), Fact(function='swinging', side=MATCH.side_swinging, hip_height=MATCH.hip, knee_height=MATCH.knee),
           TEST(lambda knee, hip: hip > 0.85*knee), salience=1)
    def test1(self, orchestration):
        self.declare(Fact(concept=2))
        self.retract(orchestration)
        #print('Transition', self.facts)
    
    
    @Rule(Fact(concept=1), Fact(function='swinging', side=MATCH.side_swinging, hip_height=MATCH.hip_height, knee_height=MATCH.knee_height))
    def test(self, side_swinging, hip_height, knee_height):   
        gait = 1
        self.valid_actions[0] = [1, 0, 0]  # gait phase 1
        self.mask_left_hip = self.valid_actions[1][gait-1] = [1 if action_index == 12 else 0 for action_index in range(21)] # Positive hip motion for planted leg.
        #print(self.mask_left_hip)
        #sys.exit()
        for joint1_value in range(21):
            self.mask_left_knee = self.valid_actions[2][gait-1][joint1_value] = [1 if action_index >= 10 else 0 for action_index in range(21)] # Negative hip motion for swinging leg.
        for joint1_value in range(21):
                for joint2_value in range(21):
                    self.mask_right_hip = self.valid_actions[3][gait-1][joint1_value][joint2_value] = [1 if action_index == 10 else 0 for action_index in range(21)] # Slight Positive knee motion for planted leg.
        for joint1_value in range(21):
            for joint2_value in range(21):
                for joint3_value in range(21):
                    self.mask_right_knee = self.valid_actions[4][gait-1][joint1_value][joint2_value][joint3_value] = [1 if action_index == 0 else 0 for action_index in range(21)] # Negative knee motion for swinging leg.
    
        #print('Concept 1 Activated')
        pass
    
    @Rule(Fact(concept=2), Fact(function='swinging'))
    def test2(self):   
        gait = 2
        self.valid_actions[0] = [0, 1, 0]  # gait phase 1
        self.mask_left_hip = self.valid_actions[1][gait-1] = [1 if action_index == 12 else 0 for action_index in range(21)] # Positive hip motion for planted leg.
        #print(self.mask_left_hip)
        #sys.exit()
        for joint1_value in range(21):
            self.mask_left_knee = self.valid_actions[2][gait-1][joint1_value] = [1 if action_index == 6 else 0 for action_index in range(21)] # Positive hip motion for swinging leg.
        for joint1_value in range(21):
                for joint2_value in range(21):
                    self.mask_right_hip = self.valid_actions[3][gait-1][joint1_value][joint2_value] = [1 if action_index == 12 else 0 for action_index in range(21)] # Slight Positive knee motion for planted leg.
        for joint1_value in range(21):
            for joint2_value in range(21):
                for joint3_value in range(21):
                    self.mask_right_knee = self.valid_actions[4][gait-1][joint1_value][joint2_value][joint3_value] = [1 if action_index == 0 else 0 for action_index in range(21)] # Negative knee motion for swinging leg.
    
        #print('Concept 2 Activated')
        pass

        @Rule(AS.orchestration << Fact(concept=2), Fact(function='swinging', contact=MATCH.contact),
           TEST(lambda contact: contact == True), salience=1)
        def test3(self, orchestration):
            self.declare(Fact(concept=3))
            self.retract(orchestration)
            print('Transition', self.facts)
            sys.exit()
        
    ###### Concept: Gait Phase 1 ##### 
    @Rule(Fact(concept=1), AS.swinging_leg << Fact(function='swinging', side=MATCH.side_swinging),
            AS.planted_leg << Fact(function='planted', side=MATCH.side_planted))  # Not fuzzy.
    def gait_phase_1(self, swinging_leg, planted_leg, side_swinging, side_planted):  # Deterministic Phase 1.
        # Each concept rule needs to mask the gait and flip bits for control concept.
        self.valid_actions[0] = [1, 0, 0]  # gait phase 1
        # Flip bits for joint modifications. (Left Hip, Right Hip, Left Knee, Right Knee)
        
        # action_space[0] is the gait
        # action_space[1] is joint 1 for each gait
        # action_space[2] is joint 2 for each gait, then for each possible value of joint 1
        # action_space[3] is joint 3 for each gait, then then for each possible value of joint 1, then for joint 2
        # action_space[4] is joint 4 for each gait, then then for each possible value of joint 1, then for joint 2, then for joint 3
        gait = 1
        
        """
        if swinging_leg['side'] == 'left':
            self.mask_left_hip = self.valid_actions[1][gait] = [1 if action_index >= 10 else 0 for action_index in range(21)] # Positive hip motion for swinging leg.
            for joint1_value in range(20):
                self.mask_left_knee = self.valid_actions[2][gait][joint1_value] = [1 if action_index < 9 else 0 for action_index in range(21)] # Negative knee motion for swinging leg.
        elif swinging_leg['side'] == 'right':
            for joint1_value in range(20):
                for joint2_value in range(20):
                    self.mask_right_hip = self.valid_actions[3][gait][joint1_value][joint2_value] = [1 if action_index >= 10 else 0 for action_index in range(21)] # Positive hip motion for swinging leg.
            for joint1_value in range(20):
                for joint2_value in range(20):
                    for joint3_value in range(20):
                        self.mask_right_knee = self.valid_actions[4][gait][joint1_value][joint2_value][joint3_value] = [1 if action_index < 9 else 0 for action_index in range(21)] # Negative knee motion for swinging leg.
        """
        #print('Concept 1 Activated', self.valid_actions[0])
        pass
    
    ####### Transtion: Gait Phase 1 -> 2, If planted leg is behind -> put swinging leg down: plant swinging leg
    @Rule(Leg(gait=L(1)) & Leg(function=L('planted')) & Leg(orientation=L('trailing')))
    def define_swinging(self):
        orchestration = self.declare(Orchestration(gait=1, event='selection', condition='fuzzy'))
        print('Fuzzy condition between gait phase 1 and 2: ', orchestration['event'])
        pass

    @Rule(AS.orchestration << Orchestration(gait=L(1)) & Orchestration(event=L('selection')) & Orchestration(condition=L('fuzzy')), salience=1)   # Fuzzy, gait choice only. Let it learn what to do in transition. 
    def transition_phase_2(self, orchestration):
        # Each concept rule needs to mask the gait and flip bits for control concept.
        self.valid_actions[0] = [1, 1, 0]  # gait phase 1 or gait phase 2
        self.retract(orchestration)
        print('Gait Transition -> Phase 2')
        pass

    ###### Concept: Gait Phase 2 #####  
    @Rule(Orchestration(gait=L(2)) & NOT(Orchestration(event=L('selection'))), AS.swinging_leg << Leg(gait=L(2)) & Leg(function=L('swinging')),
            AS.planted_leg << Leg(gait=L(2)) & Leg(function=L('planted'), side=MATCH.side), salience=0)  # Not fuzzy.
    def gait_phase_2(self, swinging_leg, planted_leg):  # Deterministic Phase 1.
        # Each concept rule needs to mask the gait and flip bits for control concept.
        self.valid_actions[0] = [1, 0, 0]  # gait phase 1
        # Flip bits for joint modifications. (Left Hip, Left Knee, Right Hip, Right Knee) 

        if swinging_leg[side] == 'left':
            for mask in range(20):
                self.valid_actions[4][mask][0][0][0] = [1 if action_index >= 10 else 0 for action_index in range(21)] # Negative hip motion for swinging leg.
            for mask in range(20):
                self.valid_actions[4][0][0][mask][0] = [1 if action_index < 9 else 0 for action_index in range(21)] # Positive knee motion for swinging leg.
        elif swinging_leg[side] == 'right':
            for mask in range(20):
                self.valid_actions[4][0][mask][0][0] = [1 if action_index < 9 else 0 for action_index in range(21)] # Negative hip motion for swinging leg.
            for mask in range(20):
                self.valid_actions[4][0][0][0][mask] = [1 if action_index >= 10 else 0 for action_index in range(21)] # Positive knee motion for swinging leg.
        print('Concept 2 Activated')
        pass

    
    ##### Transition: Gait Phase 2 -> 3: Push off planted leg, if swinging leg makes contact with the ground.
    @Rule(Leg(gait=L(2)) & Leg(function=L('swinging')) & Leg(contact=L(True)))
    def define_swinging(self):
        orchestration = self.declare(Orchestration(gait=2, event='selection', condition='fuzzy'))
        print('Fuzzy condition between gait phase 2 and 3: ', orchestration['event'])
        pass
 
    @Rule(AS.swinging_leg << Leg(gait=L(2)) & Leg(function=L('swinging')) & Leg(contact=L(True)),
            AS.planted_leg << Leg(gait=L(2)) & Leg(function=L('planted')), salience=1)
    def transition_phase_3(self, swinging_leg, planted_leg):
        # Each concept rule needs to mask the gait and flip bits for control concept.
        self.valid_actions[0] = [0, 1, 1]  # gait phase 2 or 3
        self.modify (swinging_leg, function='planted', gait='3') # Remember to check for modify effects. 
        self.modify (planted_leg, function='swinging', gait='3')
        print('Gait Transition -> Phase 3')
        pass

    ###### Concept: Gait Phase 3 #####  
    @Rule(Orchestration(gait=L(3)) & NOT(Orchestration(event=L('selection'))), AS.swinging_leg << Leg(gait=L(3)) & Leg(function=L('swinging')),
            AS.planted_leg << Leg(gait=L(3)) & Leg(function=L('planted'), side=MATCH.side), salience=0)  # Not fuzzy.
    def gait_phase_3(self, swinging_leg, planted_leg):  # Deterministic Phase 3.
        # Each concept rule needs to mask the gait and flip bits for control concept.
        self.valid_actions[0] = [0, 0, 1]  # gait phase 3
        # Flip bits for joint modifications. (Left Hip, Left Knee, Right Hip, Right Knee) 
        pass 

    ##### Transition: Gait Phase 3 -> 1: Push Off.
    # Heurisitc: This heuristic is related to the size of the supporting base and the x velocity in relation to the speed the gait is validated for.
    @Rule(AS.orchestration << Orchestration(event=L('selection')),
            AS.left_leg << Leg(gait=L(3)) & Leg(side=L('left')), AS.right_leg << Leg(gait=L(3)) & Leg(side=L('right')), salience=1)
    def transition_phase_1(self, orchestration):
        #Each concept rule needs to mask the gait and flip bits for control concept.
        self.valid_actions[0] = [1, 0, 1]  # gait phase 1 or 3
        self.retract(orchestration)
        print('Gait Transition -> Phase 1')
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
        self.engine.valid_actions = mask(self.action_space)
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
        
        # Query the state to pass to def facts.
        sim = self.training_env.env_method('get_state')
        #print(sim[0]['state'][8], sim[0]['state'][13], sim[0]['action'])
        left_leg = {'side': 'left', 'contact': sim[0]['state'][8], 'position': sim[0]['legs'][0].position[0], 'hip_angle': sim[0]['state'][4], 'knee_angle': sim[0]['state'][6], 'hip_height': sim[0]['state'][26], 'knee_height': sim[0]['state'][24]}
        right_leg = {'side': 'right', 'contact': sim[0]['state'][13], 'position': sim[0]['legs'][1].position[0], 'hip_angle': sim[0]['state'][9], 'knee_angle': sim[0]['state'][11], 'hip_height': sim[0]['state'][27], 'knee_height': sim[0]['state'][25]}
        
        self.engine.reset(left_position=left_leg['position'], left_contact=left_leg['contact'], \
            right_position=right_leg['position'], right_contact=right_leg['contact'], \
                left_hip_angle=left_leg['hip_angle'], left_knee_angle=left_leg['knee_angle'], \
                    right_hip_angle=right_leg['hip_angle'], right_knee_angle=right_leg['knee_angle'], \
                        left_hip_height=left_leg['hip_height'], right_hip_height=right_leg['hip_height'], \
                            left_knee_height=left_leg['knee_height'], right_knee_height=right_leg['knee_height'])
        #self.mask(self.action_space)()
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

        self.engine.reset(left_position=left_leg['position'], left_contact=left_leg['contact'], \
            right_position=right_leg['position'], right_contact=right_leg['contact'], \
                left_hip_angle=left_leg['hip_angle'], left_knee_angle=left_leg['knee_angle'], \
                    right_hip_angle=right_leg['hip_angle'], right_knee_angle=right_leg['knee_angle'], \
                        left_hip_height=left_leg['hip_height'], right_hip_height=right_leg['hip_height'], \
                            left_knee_height=left_leg['knee_height'], right_knee_height=right_leg['knee_height'])
        print('joint status: ', right_leg['hip_height'], right_leg['knee_height'], left_leg['hip_height'], left_leg['knee_height'], right_leg['contact'])
        #self.valid_actions = mask(self.action_space)
        #print('Before engine run: ', self.engine.facts)
        #print('Agenda: ', self.engine.agenda)
        self.engine.run()
        #print('After engine run: ', self.engine.facts)
        #print('Agenda: ', self.engine.agenda)
        #print('done run')
        #sys.exit()
        #print('engine run', tf.convert_to_tensor(np.expand_dims(self.engine.valid_actions[3], 0)).get_shape().as_list())
        #self.action_mask = test_mask #self.engine.valid_actions
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