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

    def __init__(self):

        #####
        ##### Machine Teaching 
        self.action_space = MultiDiscrete([21, 6, 4])

        self.observation_shape = (1, 33, 33)
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
        print('finished init')

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
        print('in reset')
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
        return self.state()

    def step(self, actions):
        print('in step')
        
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

        self.valid_actions = [valid_actions1, valid_actions2, valid_actions3]
        self.counter += 1

        return self.state(), 0, self.finish(), {'action_mask': self.valid_actions}

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

    def render(self, mode='human'):
        pass

    def finish(self):
        return self.counter == 250

    def state(self):
        tmp = np.reshape(np.array([*range(100)]), self.observation_shape)
        obs = tmp / 100
        return obs
