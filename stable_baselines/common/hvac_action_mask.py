import gym
import numpy as np
from gym.spaces import Discrete, MultiDiscrete

import math
from pprint import pprint as pp
from scipy import linalg
from scipy import integrate
from scipy import interpolate

import os

class SimulationParameters:
    def __init__(self):
        # Simulation time period
        self.h = 5 * 60 # 5 minutes 
        self.nbrDays = 5
        self.endTime = self.nbrDays * 24 * 60 * 60
        self.t = np.arange(0, self.endTime, self.h)
        self.tH = self.t / 3600
        self.N = self.t.size

sp = SimulationParameters()

class RoomParams:
    # roomL1: length of outdoor facing wall in meters
    # roomL2: length of perpendicular wall in meters
    # roomHeight: ceiliing height in meters
    # RiwU: Outdoor wall total U value (W/(m^2 * k)). [0.15 - 1.0]
    def __init__(self, roomL1, roomL2, roomHeight, RiwU = 0.5, wallQuota = 0.91, CO2new = 400):
        # Sample time
        self.h = sp.h 
        
        # Room volume in m^3
        self.V = roomL1 * roomL2 * roomHeight
        
        # wallQuota: Inertia placement in outer wall, 0=inside, 1=outside
        self.wallQuota = wallQuota
        
        # Thermal capacistance (J/K) single plaster wall
        self.Ci = 800 * 700 * 2 * (roomL1 + roomL2) * roomHeight*0.01
        
        # Heat loss coefficient (W/K)
        self.Riw = RiwU * roomL1 * roomHeight * wallQuota
        
        # Heat loss coefficient (W/K)
        self.Rwo = RiwU * roomL1 * roomHeight * (1 - wallQuota)
        
        # Envelope thermal capacistance J/K brick wall
        self.Cw = 900 * 1890 * roomL1 * roomHeight * 0.11
        
        # max air flow into room in m^3/s
        self.maxFlow = self.V * 10 / (60 * 60)
        
        # CO2 ppm of fresh air
        self.CO2new = CO2new
        
        # power to cool 30 to 24 deg C open damper full fan speed
        self.maxCoolPower = self.maxFlow * 1005 * 1.205 * (30 - 24)
        
        # power to heat 15 to 21 deg C open damper full fan speed
        self.maxHeatPower = self.maxFlow * 1005 * 1.205 * (21 - 15)

class EnvironmentSignals:
    def __init__(self, weatherFile, occupancyFile):
        self.officeHours = np.logical_and((sp.tH % 24) > 7 , (sp.tH % 24) < 17)
        
        weatherData = np.loadtxt(weatherFile, delimiter=',', skiprows=1)[:,0:2]
        self.Tout = np.interp(sp.t, weatherData[:,0], weatherData[:,1])
        
        occupancyData = np.loadtxt(occupancyFile, delimiter=',', skiprows=1)
        occInterp = interpolate.interp1d(occupancyData[:,0],occupancyData[:,1], kind='nearest', fill_value="extrapolate") # added , fill_value="extrapolate"
        
        self.nbrPeople = np.array([occInterp(ti) for ti in sp.t])
        self.nbrPeople[np.logical_not(self.officeHours)] = 0
        
        self.TsetCool = 28 * np.ones(sp.N)
        self.TsetCool[self.officeHours] = 23
        
        self.TsetHeat = 16 * np.ones(sp.N)
        self.TsetHeat[self.officeHours] = 21
        
        # zero!
        self.intLoad = 0 * 200 * (0.5 * np.sin(1. / 24 * 2 * np.pi * sp.tH) + 0.5)

class SignalVectors:
    def __init__(self):
        self.Tin = np.zeros(sp.N)
        self.TinMeas = np.zeros(sp.N)
        self.Twall = np.zeros(sp.N)
        self.CO2 = np.zeros(sp.N)
        self.heatValve = np.zeros(sp.N)
        self.coolValve = np.zeros(sp.N)
        self.damper = np.zeros(sp.N)
        self.fanSpeed =np.zeros(sp.N)
        self.energyHeat = np.zeros(sp.N)
        self.energyCool = np.zeros(sp.N)
        self.energyElec = np.zeros(sp.N)
        
        # For debugging
        self.I = np.zeros(sp.N)
        self.ctrlError = np.zeros(sp.N)
        self.heatMode = np.zeros(sp.N)
        self.TsetHeat = np.zeros(sp.N)
        self.TsetCool = np.zeros(sp.N)

class RoomSimEnv:
    def __init__(self, Tout, nbrPeople, intLoad):
        self.Tout = Tout
        self.nbrPeople = nbrPeople
        self.intLoad = intLoad

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
        self.deadZone = deadZone # Not used!
        self.maxCtrlSig = maxCtrlSig
        self.minCtrlSig = minCtrlSig
        self.CO2Limit = CO2Limit
    
    def step(self, ctrlState):
        #Switch heating/cooling mode if necessary
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
            ctrlState.I += (self.PIgain * self.sampleTime/self.PIint) * ctrlError

        # Map control signal to valve positions
        if ctrlState.heatMode:
            ctrlState.heatValve = v
            ctrlState.coolValve = 0
        else:
            ctrlState.heatValve = 0
            ctrlState.coolValve = v
            
        # Damper control based on CO2, p-controller
        pBand = 400 # CO2 interval under limit to apply damper range on
        minDamperPos = 5 # minimum damper position
        ctrlState.damper = np.clip((100 - minDamperPos) / pBand * (ctrlState.CO2 - (self.CO2Limit - pBand)) + minDamperPos,
                                   minDamperPos, 100)
        
        # Fan speed set by max need of damper or heating
        ctrlState.fanSpeed = max(ctrlState.coolValve,
                                ctrlState.heatValve,
                                ctrlState.damper)
        
        # For debugging
        ctrlState.ctrlError = ctrlError


class MultiDiscreteUnbalancedActionMaskEnv(gym.Env):
    metadata = {'render.modes': ['human', 'system', 'none']}

    def __init__(self):
        
        """
        Tin = sp.Tin
        roomL1 = sp.roomL1
        roomL2 = sp.roomL2
        roomHeight = sp.roomHeight
        """

        roomL1 = 10
        roomL2 = 5
        roomHeight = 3
        Tin = 20
        sampleTime = 300
        
        piCtrl = PIctrl(PIgain = 25,
                PIint = 900,
                sampleTime = 300,  # sp.h
                deadZone = 0.0,
                maxCtrlSig = 100,
                minCtrlSig = 0,
                CO2Limit = 1200)

        ctrlState = PIctrlState()
        sigV = SignalVectors()
        print(os.getcwd())
        envSignals = EnvironmentSignals(weatherFile='../stable_baselines/common/hvac_data/weather.csv', occupancyFile='../stable_baselines/common/hvac_data/occupancy.csv')

        roomParams = RoomParams(roomL1, roomL2, roomHeight)
        #roomSim = RoomSim(Tin, roomParams = roomParams, envSignals = envSignals, sampleTime = 300)  #sp.h
        
        #Sim Init
        self.h = sampleTime
        self.roomParams = roomParams
        self.Tin = Tin
        self.TinMeas = Tin
        self.CO2 = 400
        self.Twall = roomParams.wallQuota * envSignals.Tout[0]  + (1 - roomParams.wallQuota) * Tin
        self.energyHeat = 0
        self.energyCool = 0
        self.energyElec = 0
        self.occupancy = False
        self.damperOld = -1
        
        # Continous time model equations
        A = np.array([[-1 / (roomParams.Ci * roomParams.Riw), 1 / (roomParams.Ci * roomParams.Riw)],
                      [1 / (roomParams.Cw * roomParams.Riw), -1 / roomParams.Cw * (1 / roomParams.Riw + 1 / roomParams.Rwo)]])
        B = np.array([[1 / roomParams.Ci, 0.], [0, 1 / (roomParams.Cw * roomParams.Rwo)]])
       
        # Conversion to discrete time model
        self.Ad = linalg.expm(A * self.h)

        def f(x, y):
            return lambda t: np.dot(linalg.expm(A * t), B)[x][y]
        
        def fq(x, y):
            return integrate.quad(f(x,y), 0, self.h)[0]

        self.Bd = np.array([[fq(0, 0), fq(0, 1)],
                            [fq(1, 0), fq(1, 1)]])
        
        # Temperature sensor response
        self.AdSensor = math.exp(-1 / (9 * 60) * self.h)

        # Machine Teaching Init
        self.action_space = MultiDiscrete([2, 3, 4])

        self.observation_shape = (1, 10, 10)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=np.float16)

        self.counter = 0
        self.valid_actions = [[1, 1],
                              [1, 1, 1],
                              [1, 1, 1, 1]]


    def step(self, action: int):
        # HVAC Calulations
        # Applied power
        flow = roomParams.maxFlow * ctrlState.fanSpeed / 100
        Pvent = flow * 1005 * 1.205 * ctrlState.damper/100*(env.Tout - self.Tin)

        # Heating and cooling power from valves
        Phi = ctrlState.heatValve * self.roomParams.maxHeatPower / 100  - ctrlState.coolValve*self.roomParams.maxCoolPower/100
        
        # add internal loads and thermal heat from people
        Phi += env.intLoad + 100 * env.nbrPeople # 100W/person
        Phi += Pvent
        
        # Discrete time thermal equation: Ad * x + Bd * u
        self.Tin, self.Twall = np.dot(self.Ad, 
                                      np.transpose([self.Tin, self.Twall])) + np.dot(self.Bd, np.transpose([Phi, env.Tout]))
        
        # Low pass filter (9 min) for temperature sensor
        self.TinMeas = self.TinMeas * self.AdSensor + (1. - self.AdSensor) * self.Tin

        # == CO2 == 
        # (based on Bonsai pilot 2018)
        # state vector: CO2 concentration (ppm)
        # input vector: CO2new nbrPeople
        if ctrlState.damper != self.damperOld:
            A = -flow * ctrlState.damper / (roomParams.V * 100)
            gen = 1043 * 516 / (24 * 60 * 60) # CO2 generated by people
            B = np.array([flow * ctrlState.damper / (roomParams.V * 100), gen/roomParams.V])
            
            # Discretize every step due to the nonlinear property of the equation
            self.AdCO2 = math.exp(A * self.h)
        
            def f(x):
                return lambda tt: np.dot(math.exp(A * tt), B)[x]
        
            def fq(x):
                return integrate.quad(f(x), 0, self.h)[0]

            self.BdCO2 = np.array([fq(0), fq(1)])
            self.damperOld = ctrlState.damper
            
        self.CO2 = self.AdCO2 * self.CO2 + np.dot(self.BdCO2, np.transpose([roomParams.CO2new, env.nbrPeople]))
        
        # == Energy ==
        # in Joule since last sample
        self.energyHeat = ctrlState.heatValve / 100 * roomParams.maxHeatPower * self.h
        self.energyCool = ctrlState.coolValve / 100 * roomParams.maxCoolPower * self.h
        self.energyElec = ctrlState.fanSpeed / 100 * roomParams.maxFlow * 3000 * self.h
        
        # == Occupancy sensor ==
        self.occupancy = env.nbrPeople > 0

        # Run controller and sim
        k = self.counter
        ctrlState.TsetCool = envSignals.TsetCool[k]
        ctrlState.TsetHeat = envSignals.TsetHeat[k]
        ctrlState.TinMeas = roomSim.TinMeas
        ctrlState.CO2 = roomSim.CO2
        #piCtrl.step(ctrlState) # update ctrlState's heatValve, coolValve, damper, fanSpeed
        # Record data
        sigV.Tin[k] = roomSim.Tin
        sigV.TinMeas[k] = roomSim.TinMeas
        sigV.Twall[k] = roomSim.Twall
        sigV.CO2[k] = roomSim.CO2
        sigV.heatValve[k] = ctrlState.heatValve
        sigV.coolValve[k] = ctrlState.coolValve
        sigV.damper[k] = ctrlState.damper
        sigV.fanSpeed[k] = ctrlState.fanSpeed
        sigV.energyHeat[k] = roomSim.energyHeat
        sigV.energyCool[k] = roomSim.energyCool
        sigV.energyElec[k] = roomSim.energyElec
        
        # For debugging
        sigV.I[k] = ctrlState.I
        sigV.ctrlError[k] = ctrlState.ctrlError
        sigV.heatMode[k] = ctrlState.heatMode
        sigV.TsetHeat[k] = ctrlState.TsetHeat
        sigV.TsetCool[k] = ctrlState.TsetCool

        env = RoomSimEnv(Tout = envSignals.Tout[k], 
                        nbrPeople = envSignals.nbrPeople[k], 
                        intLoad = envSignals.intLoad[k])

        # Machine Teaching
        valid_actions = [[1, 1],
                         [1, 1, 1],
                         [1, 1, 1, 1]]
        for i, action in enumerate(actions):
            if self.valid_actions[i][action] == 0:
                raise Exception("Invalid action was selected! Valid actions: {}, "
                                "action taken: {}".format(self.valid_actions, actions))
            valid_actions[i][action] = 0

        self.valid_actions = valid_actions
        self.counter += 1

        return self.state(), 0, self.finish(), {'action_mask': self.valid_actions}

    def reset(self):
        self.counter = 0
        self.valid_actions = [[1, 1],
                              [1, 1, 1],
                              [1, 1, 1, 1]]
        return self.state()

    def render(self, mode='human'):
        pass

    def finish(self):
        return self.counter == 250

    def state(self):
        tmp = np.reshape(np.array([*range(100)]), self.observation_shape)
        obs = tmp / 100
        return obs
