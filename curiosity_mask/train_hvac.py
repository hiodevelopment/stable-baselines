import os
import warnings

import stable_baselines.common.hvac_action_mask as hvac
from stable_baselines import PPO2, A2C
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.callbacks import BaseCallback

from experta import *
from experta.fact import *
import schema

env = DummyVecEnv([hvac.MultiDiscreteActionMaskEnv])

n_steps = 0

class Occupancy(Fact):
   level = Field(schema.Or("low", "normal", "high"), mandatory=True)

class CO2(Fact):
    level = Field(schema.Or("low", "high", "non-compliant"), mandatory=True)

class Outdoor_Temperature(Fact):
    level = Field(schema.Or("normal", "extreme"), mandatory=True)

class HVAC_Machine_Teaching(KnowledgeEngine):
    @DefFacts()
    def enivronment_definitions(self, people, co2, weather):
        """Declare threshold rules."""
        if people > 5:
            yield Occupancy(level='high')
        elif people < 1:
            yield Occupancy(level='low')
        else:
            yield Occupancy(level='normal')

        if weather > 15 or weather < 2:
            yield Outdoor_Temperature(level='extreme')
        else:
            yield Outdoor_Temperature(level='normal')

        if co2 > 900:
            yield CO2(level='high')
        elif co2 < 300:  #300 for first experiment
            yield CO2(level='low')
        elif co2 > 1200:
            yield CO2(level='non-compliant')

        valid_actions1 = [1] * 21
        valid_actions2 = []

        for action in valid_actions1:
            valid_actions2.append([1] * 6)

        valid_actions3 = []
        
        for i in range(21):
            tmp = [] 
            for j in range(6):
                tmp.append([1] * 4)
            valid_actions3.append(tmp)

        self.valid_actions = [valid_actions1, valid_actions2, valid_actions3]
    

    # Heuristic Machine Teaching Strategies
    @Rule(Occupancy(level=L('low')) & CO2(level=L('low')) | Outdoor_Temperature(level='extreme'))
    def recycle_air(self):
        
        # Close the Damper
        valid_actions1 = [1] * 21
        valid_actions2 = []

        #for action in valid_actions1:
        for index, value in enumerate(valid_actions1):
            if index > 1:
                valid_actions2.append([0] * 6)
            if index <= 1:
                valid_actions2.append([1] * 6)

        valid_actions3 = []

        for i in range(21):
            tmp = [] 
            for j in range(6):
                tmp.append([1] * 4)
            valid_actions3.append(tmp)

        valid_actions = [valid_actions1, valid_actions2, valid_actions3]
        
        self.valid_actions = valid_actions

        return valid_actions

    @Rule(Occupancy(level=L('high')) | CO2(level=L('high') | CO2(level=L('non-compliant')) | Outdoor_Temperature(level='normal')))
    def freshen_air(self):
        
        # Open the Damper
        valid_actions1 = [1] * 21
        valid_actions2 = []

        #for action in valid_actions1:
        for index, value in enumerate(valid_actions1):
            if (index < 2 or index == 5):
                valid_actions2.append([0] * 6)
            else:
                valid_actions2.append([1] * 6)

        valid_actions3 = []

        for i in range(21):
            tmp = [] 
            for j in range(6):
                tmp.append([1] * 4)
            valid_actions3.append(tmp)

        valid_actions = [valid_actions1, valid_actions2, valid_actions3]
        #print('found a rule', valid_actions)

        self.valid_actions = valid_actions

        return valid_actions

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
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.engine = HVAC_Machine_Teaching()


    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        
        print('train start')
        sim = self.training_env.env_method('get_state')
        o = sim[0][0]
        h = sim[0][1]
        people = h['n_people']
        co2 = o['CO2']
        weather = h['T_out']
        self.engine.reset(people=people, co2=co2, weather=weather)
         

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
        
        #if self.verbose > 0:
            #print("Num timesteps: {}".format(self.num_timesteps))

        self.engine.run()
        infos = self.training_env.set_attr('valid_actions', self.engine.valid_actions)
        
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

model = PPO2(MlpPolicy, env, verbose=2, tensorboard_log="experiment/", nminibatches=1)  
#model = A2C(MlpPolicy, env, verbose=2)
model.learn(1000000, callback=callback)

# model.save("expert_model")

# Enjoy trained agent
"""
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