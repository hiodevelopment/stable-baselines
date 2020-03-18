import gym
import numpy as np
from gym.spaces import Discrete, MultiDiscrete

import math
from pprint import pprint as pp
from scipy import linalg
from scipy import integrate
from scipy import interpolate

import os
import warnings

import stable_baselines.common.hvac_data.hvac_environment as Environment
import stable_baselines.common.hvac_data.hvac_params as Parameters

# Action 1: Valve
# Action 2: Damper
# Action 3: Fan

#valve: number <-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1>,
#damper: number <0.05, 0.2, 0.4, 0.6, 0.8, 1>

class MultiDiscreteActionMaskEnv(gym.Env):
    metadata = {'render.modes': ['human', 'system', 'none']}

    def __init__(self,
                 sim_params: dict = {},
                 room_params: dict = {},
                 env: Environment = None,
                 n_hist: int = None  # Number of timesteps to keep history (Tin, etc.) @TODO: Add
                 ):

        self.disturbance = False
        # Simulation Parameters
        sim_params = Parameters.get_sim_parameters()
        #print(sim_params)
        self.sim_params = sim_params
        self.h = sim_params['sample_time']  # Must be multiple of self.h_internal
        self.h_internal = 60  # Internal sample rate in seconds
        self.hist_buffer = 30

        # Clock / iteration (k)
        self.k = 0
        self._update_date_time()

        # Memory
        self.damper_old = -1
        self.error_T_hist = []
        self.heat_valve_hist = []
        self.cool_valve_hist = []
        self.damper_hist = []
        self.fan_speed_hist = []
        self.T_set_cool_hist = []
        self.T_set_heat_hist = []
        self.occupancy_hist = []
        self.delta_co2_hist = []

        # Room parameters
        room_params = Parameters.get_room_parameters(selection='random')
        self.room_params = room_params
        #print(room_params)
        # Get the environment simulator
        """
        env = Environment.Environment(sim_params=sim_params,
                              room_params=room_params,
                              weather_selection='deterministic',
                              weather_case=1,
                              weather_data_shift=1 * sim_params['n_days'],
                              occupancy_selection='deterministic',
                              occupancy_case=1,
                              occupancy_data_shift=1 * sim_params['n_days']
                              )
        """
        ### Randomized
        env = Environment.Environment(sim_params=sim_params,
                              room_params=room_params,
                              weather_selection='random',
                              weather_data_shift=np.random.randint(365),
                              occupancy_selection='random',
                              occupancy_data_shift=np.random.randint(365)
                              )
        self.env = env
        #print(env)
        self._update_env()
        self._update_grace_period_remaining(True)

        # Initial CO2
        self.CO2 = room_params['CO2_fresh']
        self._measure_deltaCO2()

        # Temp inside
        self.T_in = room_params['T_in_initial']
        self.T_in_meas = room_params['T_in_initial']
        # self.T_in_hist = deque().append(T_in)
        self.T_wall = room_params['wallQuota'] * self.T_out + (1 - room_params['wallQuota']) * self.T_in
        self._measure_error_T()

        # Energy
        self.energy_heat = 0
        self.energy_cool = 0
        self.energy_elec = 0

        # Dynamics
        # Continous time model equations
        A = np.array([[-1 / (room_params['Ci'] * room_params['Riw']),
                       1 / (room_params['Ci'] * room_params['Riw'])],
                      [1 / (room_params['Cw'] * room_params['Riw']),
                       -1 / room_params['Cw'] * (1 / room_params['Riw'] + 1 / room_params['Rwo'])]])

        B = np.array([[1 / room_params['Ci'], 0.],
                      [0, 1 / (room_params['Cw'] * room_params['Rwo'])]])

        # Conversion to discrete time model
        self.Ad = linalg.expm(A * self.h_internal)

        def f(x, y):
            return lambda t: np.dot(linalg.expm(A * t), B)[x][y]

        def fq(x, y):
            return integrate.quad(f(x, y), 0, self.h_internal)[0]

        self.Bd = np.array([[fq(0, 0), fq(0, 1)],
                            [fq(1, 0), fq(1, 1)]])

        # Temperature sensor response
        self.AdSensor = math.exp(-1 / (9 * 60) * self.h_internal)

        #####
        ##### Machine Teaching 
        self.action_space = MultiDiscrete([21, 6, 4])

        self.observation_shape = (27,)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=np.float16)

        self.counter = 0
        self.valid_actions1 = [1] * 21
        self.valid_actions2 = []

        for action in self.valid_actions1:
            self.valid_actions2.append([1] * 6)

        self.valid_actions3 = []

        for i in range(21):
            tmp = [] 
            for j in range(6):
                tmp.append([1] * 4)
            self.valid_actions3.append(tmp)

        self.valid_actions = [self.valid_actions1, self.valid_actions2, self.valid_actions3]
        #print('finished init')

    def _update_date_time(self):
        self.time = self.k * self.h  # Total time elapesed in seconds
        self.days, remainder = np.divmod(self.time, 24 * 60 * 60)
        self.hours, remainder = np.divmod(remainder, 60 * 60)
        self.minutes = remainder // 60

    def _update_grace_period_remaining(self, reset):
        if reset:
            self.grace_left = self.room_params['grace_period']
        else:
            self.grace_left -= self.sim_params['sample_time']
            if self.grace_left < 0:
                self.grace_left = 0

    def _update_env(self):
        env_signals = self.env.get_env_signals(self.k)
        # Environment signals
        self.T_out = env_signals['T_out']
        self.n_people = env_signals['n_people']
        self.occupancy = env_signals['n_people'] > 0  # Occupancy sensor
        self.int_load = env_signals['int_load']
        # Update remaining time in grace period
        if (self.k == 0) or (self.T_set_cool != env_signals['T_set_cool']) or (
                self.T_set_heat != env_signals['T_set_heat']):
            # Reset it
            self._update_grace_period_remaining(True)
        else:
            # Decrease it. No changes to T-set-cool or T-set-heat
            self._update_grace_period_remaining(False)
        self.T_set_cool = env_signals['T_set_cool']
        self.T_set_heat = env_signals['T_set_heat']

        # Add to history
        self.T_set_cool_hist.append(env_signals['T_set_cool'])
        self.T_set_heat_hist.append(env_signals['T_set_heat'])
        self.occupancy_hist.append(env_signals['n_people'] > 0)

        # State truncation
        if len(self.T_set_cool_hist) > self.hist_buffer:
            self.T_set_cool_hist = self.T_set_cool_hist[-self.hist_buffer:]
        if len(self.T_set_heat_hist) > self.hist_buffer:
            self.T_set_heat_hist = self.T_set_heat_hist[-self.hist_buffer:]
        if len(self.occupancy_hist) > self.hist_buffer:
            self.occupancy_hist = self.occupancy_hist[-self.hist_buffer:]
            
    def _update_temp(self, control, flow, AHU_temp):
        
        # Applied power
        P_vent = flow * 1005 * 1.205 * control['damper'] / 100 * (AHU_temp - self.T_in)
    
        # limited heating and cooling power from valves
        if (self.T_in > 60):
            maxHeatPower = 0
        else:
            maxHeatPower = self.room_params['maxHeatPower']
        if (self.T_in < 10):
            maxCoolPower = 0
        else:
            maxCoolPower = self.room_params['maxCoolPower']

        # Mapping from valve signal to thermal power
        heat_valve_map = max( min( \
            100 * (control['heatValve'] - self.room_params['valve_closed_pos']) / (self.room_params['valve_open_pos'] - self.room_params['valve_closed_pos']) \
            ,100 ), 0)
        cool_valve_map = max( min( \
            100 * (control['coolValve'] - self.room_params['valve_closed_pos']) / (self.room_params['valve_open_pos'] - self.room_params['valve_closed_pos']) \
            ,100 ), 0)    
        
        # Heating and cooling power from valves
        Phi = heat_valve_map * maxHeatPower / 100 \
              - cool_valve_map * maxCoolPower / 100

        # add internal loads and thermal heat from people
        Phi += self.int_load + 100 * self.n_people  # 100W/person
        Phi += P_vent

        # Discrete time thermal equation: Ad * x + Bd * u
        self.T_in, self.T_wall = np.dot(self.Ad,
                                        np.transpose([self.T_in, self.T_wall])) + np.dot(self.Bd,
                                                                                         np.transpose(
                                                                                             [Phi, self.T_out]))
        if self.disturbance:
            self.T_in += np.random.choice([0, -5, 5], p=[0.9998, 0.0001, 0.0001])

    def _update_control_hist(self, control):
        # Add to history
        self.heat_valve_hist.append(control['heatValve'])
        self.damper_hist.append(control['damper'])
        self.cool_valve_hist.append(control['coolValve'])
        self.fan_speed_hist.append(control['fanSpeed'])

        # State truncation
        if len(self.heat_valve_hist) > self.hist_buffer:
            self.heat_valve_hist = self.heat_valve_hist[-self.hist_buffer:]
        if len(self.damper_hist) > self.hist_buffer:
            self.damper_hist = self.damper_hist[-self.hist_buffer:]
        if len(self.cool_valve_hist) > self.hist_buffer:
            self.cool_valve_hist = self.cool_valve_hist[-self.hist_buffer:]
        if len(self.fan_speed_hist) > self.hist_buffer:
            self.fan_speed_hist = self.fan_speed_hist[-self.hist_buffer:]

    def _measure_error_T(self):
        error = 0
        if self.T_in_meas > self.T_set_cool:
            error = self.T_in_meas - self.T_set_cool
        elif self.T_in_meas < self.T_set_heat:
            error = self.T_in_meas - self.T_set_heat
        self.error_T_hist.append(error)

        # State truncation
        if len(self.error_T_hist) > self.hist_buffer:
            self.error_T_hist = self.error_T_hist[-self.hist_buffer:]

        # Also measure the real error
        real_error = 0
        if self.T_in > self.T_set_cool:
            real_error = self.T_in - self.T_set_cool
        elif self.T_in < self.T_set_heat:
            real_error = self.T_in - self.T_set_heat
        self.error_T_real = real_error

    def _measure_temp(self):
        # Low pass filter (9 min) for temperature sensor
        self.T_in_meas = self.T_in_meas * self.AdSensor + (1. - self.AdSensor) * self.T_in

    def _measure_deltaCO2(self):
        deltaCO2 = self.room_params['CO2_limit'] - self.CO2
        self.delta_co2_hist.append(deltaCO2)
        if len(self.delta_co2_hist) > self.hist_buffer:
            self.delta_co2_hist = self.delta_co2_hist[-self.hist_buffer:]

    def _update_co2(self, control, flow):
        # (based on Bonsai pilot 2018)
        # state vector: CO2 concentration (ppm)
        # input vector: CO2new nbrPeople
        A = -flow * control['damper'] / (self.room_params['V'] * 100)
        gen = 1043 * 516 / (24 * 60 * 60)  # CO2 generated by people
        B = np.array([flow * control['damper'] / (self.room_params['V'] * 100), gen / self.room_params['V']])

        # Discretize every step due to the nonlinear property of the equation
        self.AdCO2 = math.exp(A * self.h_internal)

        def f(x):
            return lambda tt: np.dot(math.exp(A * tt), B)[x]

        def fq(x):
            return integrate.quad(f(x), 0, self.h_internal)[0]

        self.BdCO2 = np.array([fq(0), fq(1)])
        self.damper_old = control['damper']

        self.CO2 = self.AdCO2 * self.CO2 + np.dot(self.BdCO2,
                                                  np.transpose([self.room_params['CO2_fresh'], self.n_people]))
        if self.disturbance:
            self.CO2 += np.random.choice([0, 2000], p=[0.9998, 0.0002])

    def _update_energy(self, control, flow, AHU_temp):
        P_AHU = flow * 1005 * 1.205 * control['damper'] * (AHU_temp - self.T_out) / 100
        heat_valve_map = max( min( \
            100 * (control['heatValve'] - self.room_params['valve_closed_pos']) / (self.room_params['valve_open_pos'] - self.room_params['valve_closed_pos']) \
            ,100 ), 0)
        cool_valve_map = max( min( \
            100 * (control['coolValve'] - self.room_params['valve_closed_pos']) / (self.room_params['valve_open_pos'] - self.room_params['valve_closed_pos']) \
            ,100 ), 0)    
        self.energy_heat = (heat_valve_map / 100 * self.room_params['maxHeatPower'] + max(0, P_AHU)) * self.h
        self.energy_cool = (cool_valve_map / 100 * self.room_params['maxCoolPower'] + -1 * min(0,P_AHU)) * self.h
        self.energy_elec = control['fanSpeed'] / 100 * self.room_params['maxFlow'] * 3000 * self.h

    def reset(self):
        #print('in reset')
        self.counter = 0
        self.valid_actions1 = [1] * 21
        self.valid_actions2 = []

        for action in self.valid_actions1:
            self.valid_actions2.append([1] * 6)

        self.valid_actions3 = []
        
        for i in range(21):
            tmp = [] 
            for j in range(6):
                tmp.append([1] * 4)
            self.valid_actions3.append(tmp)
        self.valid_actions = [self.valid_actions1, self.valid_actions2, self.valid_actions3]

        tmp = np.reshape(np.array([*range(27)]), self.observation_shape)
        obs = tmp / 27
        #print(obs)
        return obs

    def step(self, action):
        #print('in step, heres the actions', action)
        actions = {}
        """
        # If using PI Controller
        ### PI Controller
        piCtrl = PIctrl(PIgain = 25,
                PIint = 900,
                sampleTime = self.h,
                deadZone = 0.0,
                maxCtrlSig = 100,
                minCtrlSig = 0,
                CO2Limit = 1200)

        # Execute controller
        ctrlState = PIctrlState()
        envSignals = self.env.get_env_signals(self.time)
        ctrlState.TsetCool = envSignals['T_set_cool']
        ctrlState.TsetHeat = envSignals['T_set_heat']
        ctrlState.TinMeas = self.T_in_meas
        ctrlState.CO2 = self.CO2
        piCtrl.step(ctrlState) # update ctrlState's heatValve, coolValve, damper, fanSpeed
        
         ### If using the PI Controller
        actions['coolValve'] = ctrlState.heatValve
        actions['heatValve'] = ctrlState.coolValve
        actions['damper'] = ctrlState.damper
        actions['fanSpeed'] = ctrlState.fanSpeed

        ############
        """

        valve = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        damper = [0.05, 0.2, 0.4, 0.6, 0.8, 1]
        fan = [0.25, 0.5, 0.75, 1]

        #### If using the BRAIN
        # Transform the actions. 
        if action[0] < 0:  # Cool down the room
            actions['coolValve'] = -valve[action[0]] * 100
            actions['heatValve'] = 0          
        else:  # Warm up the room
            actions['heatValve'] = valve[action[0]] * 100
            actions['coolValve'] = 0

        actions['damper'] = damper[action[1]] * 100
        actions['fanSpeed'] = fan[action[2]] * 100
        #actions['fanSpeed'] = max(actions['damper'], actions['heatValve'], actions['coolValve'])

        #print('heres the transofrmed actions:', actions)
        #print('heres the room parameters:', self.room_params)

        flow = self.room_params['maxFlow'] * actions['fanSpeed'] / 100
        AHU_temp = min(max(self.T_out, 19), 25)  # Pre conditioning by air handler
        for _ in range(int(self.h / self.h_internal)):
            self._update_temp(actions, flow, AHU_temp)
            self._update_co2(actions, flow)
            self._measure_temp()
        self._update_control_hist(actions)
        self._measure_deltaCO2()
        self._measure_error_T()
        self._update_energy(actions, flow, AHU_temp)
        self.k += 1  # Update the iteration number
        self._update_date_time()
        self._update_env()  # This has to be the last step to feed the controller the conditions in the following step

        

        #######
        ####### Machine Teaching
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
        #print(self.valid_actions[0])

        """
        if self.valid_actions[0][actions[0]] == 0:
            raise Exception("Invalid action was selected! Valid actions: {}, "
                            "action taken: {}".format(self.valid_actions, actions))
        else:
            valid_actions1[actions[0]] = 0
        if self.valid_actions[1][actions[0]][actions[1]] == 0:
            raise Exception("Invalid action was selected! Valid actions: {}, "
                            "action taken: {}".format(self.valid_actions, actions))
        else:
            valid_actions2[0][actions[1]] = 0
            valid_actions2[1][actions[1]] = 0
        if self.valid_actions[2][actions[0]][actions[1]][actions[2]] == 0:
            raise Exception("Invalid action was selected! Valid actions: {}, "
                            "action taken: {}".format(self.valid_actions, actions))
        else:
            valid_actions3[0][0][actions[2]] = 0
            valid_actions3[0][1][actions[2]] = 0
            valid_actions3[0][2][actions[2]] = 0
            valid_actions3[1][0][actions[2]] = 0
            valid_actions3[1][1][actions[2]] = 0
            valid_actions3[1][2][actions[2]] = 0
        """

        #self.valid_actions = [valid_actions1, valid_actions2, valid_actions3] # The mask setting is one step behind.
        self.counter += 1

        #print(self.get_brain_state())

        return self.get_brain_state(), self.get_reward(), self.finish(), {'action_mask': self.valid_actions}

    def get_state(self):
        observable_state = {}
        observable_state['iteration'] = self.k
        observable_state['time'] = self.time
        observable_state['days'] = self.days
        observable_state['hours'] = self.hours
        observable_state['minutes'] = self.minutes
        observable_state['grace_left'] = self.grace_left
        observable_state['T_in_meas'] = self.T_in_meas
        observable_state['occupancy'] = self.occupancy
        observable_state['CO2'] = self.CO2
        observable_state['T_set_cool'] = self.T_set_cool
        observable_state['T_set_heat'] = self.T_set_heat
        observable_state['CO2_limit'] = self.room_params['CO2_limit']
        ## Hist
        observable_state['T_set_cool_hist'] = self.T_set_cool_hist
        observable_state['T_set_heat_hist'] = self.T_set_heat_hist
        observable_state['error_T_hist'] = self.error_T_hist
        observable_state['delta_co2_hist'] = self.delta_co2_hist
        # Control Hist
        observable_state['heat_valve_hist'] = self.heat_valve_hist
        observable_state['cool_valve_hist'] = self.cool_valve_hist
        observable_state['damper_hist'] = self.damper_hist
        observable_state['fan_speed_hist'] = self.fan_speed_hist
        observable_state['occupancy_hist'] = self.occupancy_hist

        hidden_state = {}
        hidden_state['T_in'] = self.T_in
        hidden_state['T_out'] = self.T_out
        hidden_state['energy_heat'] = self.energy_heat
        hidden_state['energy_cool'] = self.energy_cool
        hidden_state['energy_elec'] = self.energy_elec
        hidden_state['n_people'] = self.n_people
        hidden_state['T_wall'] = self.T_wall
        hidden_state['int_load'] = self.int_load
        hidden_state['damper_old'] = self.damper_old
        hidden_state['error_T_real'] = self.error_T_real

        return observable_state, hidden_state

    def get_brain_state(self):

        observable_state, hidden_state = self.get_state()

        n_lag = 5
        # DeltaT History
        last_val = None
        for i in range(1, n_lag + 1):
            if len(observable_state['error_T_hist']) >= i:
                observable_state['deltaT_' + str(i)] = observable_state['error_T_hist'][-i] / 10
                last_val = observable_state['error_T_hist'][-i] / 10
            else:
                observable_state['deltaT_' + str(i)] = last_val

        # Delta CO2 history
        last_val = None
        for i in range(1, n_lag + 1):
            if len(observable_state['delta_co2_hist']) >= i:
                observable_state['deltaCO2_' + str(i)] = observable_state['delta_co2_hist'][-i] / 1000
                last_val = observable_state['delta_co2_hist'][-i] / 1000
            else:
                observable_state['deltaCO2_' + str(i)] = last_val

        # Delta Heat Valve Hist
        last_val = 0
        for i in range(1, n_lag + 1):
            if len(observable_state['heat_valve_hist']) >= i:
                observable_state['heatV_' + str(i)] = observable_state['heat_valve_hist'][-i] / 100
                last_val = observable_state['heat_valve_hist'][-i] / 100
            else:
                observable_state['heatV_' + str(i)] = last_val

        # Delta Cool Valve Hist
        last_val = 0
        for i in range(1, n_lag + 1):
            if len(observable_state['cool_valve_hist']) >= i:
                observable_state['coolV_' + str(i)] = observable_state['cool_valve_hist'][-i] / 100
                last_val = observable_state['cool_valve_hist'][-i] / 100
            else:
                observable_state['coolV_' + str(i)] = last_val

        # Delta Damper Valve Hist
        last_val = 0
        for i in range(1, n_lag + 1):
            if len(observable_state['damper_hist']) >= i:
                observable_state['damper_' + str(i)] = observable_state['damper_hist'][-i] / 100
                last_val = observable_state['damper_hist'][-i] / 100
            else:
                observable_state['damper_' + str(i)] = last_val

        # Occupancy
        last_val = 0
        for i in range(1, n_lag + 1):
            if len(observable_state['occupancy_hist']) >= i:
                observable_state['occupancy_' + str(i)] = observable_state['occupancy_hist'][-i] * 1
                last_val = observable_state['occupancy_hist'][-i] * 1
            else:
                observable_state['occupancy_' + str(i)] = last_val

        observable_state['dist2Heat'] = (observable_state['T_in_meas'] - observable_state['T_set_heat']) * (
                observable_state['T_in_meas'] > observable_state['T_set_heat']) / 10
        observable_state['dist2Cool'] = (observable_state['T_set_cool'] - observable_state['T_in_meas']) * (
                observable_state['T_set_cool'] > observable_state['T_in_meas']) / 10

        observable_state['grace_left'] /= 60 # Normalization

        # Pull the generated state vars above into a list as the brain state. 
        
        deltaT_ = [value for key, value in observable_state.items() if 'deltaT_' in key]
        deltaCO2_ = [value for key, value in observable_state.items() if 'deltaCO2_' in key]
        heatV_ = [value for key, value in observable_state.items() if 'heatV_' in key]
        coolV_ = [value for key, value in observable_state.items() if 'coolV_' in key]
        damper_ = [value for key, value in observable_state.items() if 'damper_' in key and not 'hist' in key]
        occupancy_ = [value for key, value in observable_state.items() if 'occupancy_' in key and not 'hist' in key]
        
        brain_state = deltaT_ + [observable_state['dist2Heat']] + [observable_state['dist2Cool']] + deltaCO2_ + heatV_ + coolV_ + damper_ 

        #print('observable state:', observable_state)
        return brain_state


    def get_reward(self):

        obs_state, h_state = self.get_state()

        # Temperature Component
        reward_tg = 0
        if obs_state['grace_left'] <= 0:
            rk = 0.5
            deltaT = np.abs(h_state['error_T_real'])
            reward_t = np.exp(-deltaT * rk)
            # Temperature guardrail
            if deltaT >= 3:
                reward_tg = - deltaT * 0.2
        else:
            reward_t = 1

        # CO2 component
        co2_violation_band = 200
        co2_level = obs_state['CO2']
        co2_limit = obs_state['CO2_limit']
        co2_penalty_coef = -1
        co2_delta = co2_limit - co2_level
        co2_violation = np.amax([0, co2_violation_band - co2_delta])
        co2_reward = co2_penalty_coef * co2_violation / co2_violation_band

        # Energy
        coef = -0.001
        heat_valve = obs_state['heat_valve_hist'][-1]
        cool_valve = obs_state['cool_valve_hist'][-1]
        damper = obs_state['damper_hist'][-1]
        fan = obs_state['fan_speed_hist'][-1] #max(heat_valve, cool_valve, damper)

        reward_en_heat = coef * heat_valve
        reward_en_cool = coef * cool_valve
        reward_en_fan = coef * fan / 5
        reward_en = reward_en_heat + reward_en_cool + reward_en_fan

        # Movement
        if len(obs_state['heat_valve_hist']) < 2:
            reward_move = 0
        else:
            h2, h1 = obs_state['heat_valve_hist'][-2:]
            c2, c1 = obs_state['cool_valve_hist'][-2:]
            d2, d1 = obs_state['damper_hist'][-2:]
            # f2, f1 = obs_state['fan_speed_hist'][-2:]
            mcoef = - (1 / 15)
            valve_move_heat = mcoef * np.abs(h2 - h1) / 100
            valve_move_cool = mcoef * np.abs(c2 - c1) / 100
            damper_move = mcoef * np.abs(d2 - d1) / 100
            reward_move = valve_move_heat + valve_move_cool + damper_move

        reward_t /= 5
        reward_tg /= 5
        co2_reward /= 5

        return reward_t + reward_tg + co2_reward + reward_en + reward_move

    def render(self, mode='human'):
        pass

    def finish(self):
        return self.counter == 250

class PIctrlState:
    def __init__(self):
        self.heatMode = True
        self.I = 0
        self.TinMeas = 0
        self.TsetHeat = 0
        self.TsetCool = 0
        self.heatValve = 0
        self.coolValve = 0
        self.damper = 0
        self.fanSpeed = 0
        # For debugging
        self.ctrlError = 0


class PIctrl:
    def __init__(self, PIgain, PIint, sampleTime, deadZone, maxCtrlSig, minCtrlSig, CO2Limit):
        self.PIgain = PIgain
        self.PIint = PIint
        self.sampleTime = sampleTime
        self.deadZone = deadZone  # Not used!
        self.maxCtrlSig = maxCtrlSig
        self.minCtrlSig = minCtrlSig
        self.CO2Limit = CO2Limit

    def step(self, ctrlState):
        # Switch heating/cooling mode if necessary
        if ctrlState.TinMeas <= ctrlState.TsetHeat:
            if not ctrlState.heatMode:
                ctrlState.I = 0
            ctrlState.heatMode = True
        elif ctrlState.TinMeas >= ctrlState.TsetCool:
            if ctrlState.heatMode:
                ctrlState.I = 0
            ctrlState.heatMode = False

        # Control error based on heating mode
        if ctrlState.heatMode:
            ctrlError = ctrlState.TsetHeat - ctrlState.TinMeas
        else:
            ctrlError = ctrlState.TinMeas - ctrlState.TsetCool

        # PI algorithm
        v = self.PIgain * ctrlError + ctrlState.I
        if (v > self.maxCtrlSig):
            v = self.maxCtrlSig
        elif (v < self.minCtrlSig):
            v = self.minCtrlSig
        else:
            ctrlState.I += (self.PIgain * self.sampleTime / self.PIint) * ctrlError

        # Map control signal to valve positions
        if ctrlState.heatMode:
            ctrlState.heatValve = v
            ctrlState.coolValve = 0
        else:
            ctrlState.heatValve = 0
            ctrlState.coolValve = v

        # Damper control based on CO2, p-controller
        pBand = (self.CO2Limit - 400)  # CO2 interval under limit to apply damper range on
        minDamperPos = 5  # minimum damper position
        ctrlState.damper = np.clip(
            (100 - minDamperPos) / pBand * (ctrlState.CO2 - (self.CO2Limit - pBand)) + minDamperPos,
            minDamperPos, 100)

        # Fan speed set by max need of damper or heating
        ctrlState.fanSpeed = max(ctrlState.coolValve,
                                 ctrlState.heatValve,
                                 ctrlState.damper)

        # For debugging
        ctrlState.ctrlError = ctrlError


class SignalVectors:
    def __init__(self, N):
        self.Tin = np.zeros(N)
        self.TinMeas = np.zeros(N)
        self.Twall = np.zeros(N)
        self.CO2 = np.zeros(N)
        self.heatValve = np.zeros(N)
        self.coolValve = np.zeros(N)
        self.damper = np.zeros(N)
        self.fanSpeed =np.zeros(N)
        self.energyHeat = np.zeros(N)
        self.energyCool = np.zeros(N)
        self.energyElec = np.zeros(N)
        # For debugging
        self.I = np.zeros(N)
        self.ctrlError = np.zeros(N)
        self.heatMode = np.zeros(N)
        self.TsetHeat = np.zeros(N)
        self.TinMeas = np.zeros(N)
        self.TsetCool = np.zeros(N)

    