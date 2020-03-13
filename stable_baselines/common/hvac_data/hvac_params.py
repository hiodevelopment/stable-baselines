import numpy as np
import pandas as pd
import os


def get_sim_parameters(sample_time: float = 1, n_days: float = 5):
    sim_params = {}
    sim_params['sample_time'] = sample_time * 60  # h, 5 minutes (in seconds)
    sim_params['n_days'] = n_days
    sim_params['end_time'] = sim_params['n_days'] * 24 * 60 * 60
    sim_params['t'] = np.arange(0, sim_params['end_time'], sim_params['sample_time'])
    sim_params['tH'] = sim_params['t'] / 3600 # Time in hours
    sim_params['N'] = sim_params['t'].size
    return sim_params


def get_room_parameters(selection: str = 'default', # Whether to choose the default, deterministic or a random configuration
                        case: int = None, # If deterministic is selected, a case number needs to be entered
                        data_folder: str = 'data/',
                        room_file: str = 'room.csv',
                        roomL1: float = None,     # length of outdoor facing wall in meters
                        roomL2: float = None,      # length of perpendicular wall in meters
                        room_height: float = None, # ceiliing height in meters
                        RiwU: float = None,      # Inverse of wall total U value (W/(m^2 * k))
                        wallQuota: float = None,
                        CO2_fresh: float = None,
                        CO2_limit: float = None,
                        T_in_initial: float = None,
                        office_start_time: float = None,
                        office_end_time: float = None,
                        T_set_cool_ah: float = None,  # Cooling setpoint for after hours
                        T_set_cool_wh: float = None,  # Cooling setpoint for work hours
                        T_set_heat_ah: float = None,  # Heating setpoint for after hours
                        T_set_heat_wh: float = None,  # Heating setpoint for work hours
                        int_load_coef: float = None,
                        valve_closed_pos: float = None, # position with no heating or cooling
                        valve_open_pos: float = None # position with 100% heating or cooling
                        ):
    """
    The room parameters are read from room.csv file. 'default' selection selects Case 0,
    'random' selection randomly selects a case for each episode (including Case 0) based on the specified weights.
    Overwrite the case if any value is passed to the function.
    """
    # dirname = os.path.dirname(__file__)

    # Intrapolate the randomized parameters

    room_params = {}
    if selection == 'deterministic':
        np.random.seed(case)
    elif selection != 'random':
        raise Exception("Incorrect selection for room parameters: ", selection)

    # Overwrite the case, if any arguments specified
    if roomL1 is not None:
        room_params['roomL1'] = roomL1
    else:
        room_params['roomL1'] = np.random.uniform(5, 15)

    if roomL2 is not None:
        room_params['roomL2'] = roomL2
    else:
        room_params['roomL2'] = np.random.uniform(5, 15)

    if room_height is not None:
        room_params['room_height'] = room_height
    else:
        room_params['room_height'] = 3

    if RiwU is not None:
        room_params['RiwU'] = RiwU
    else:
        room_params['RiwU'] = np.random.uniform(0.2, 6)

    if wallQuota is not None:
        room_params['wallQuota'] = wallQuota    # wallQuota: Inertia placement in outer wall, 0=inside, 1=outside
    else:
        room_params['wallQuota'] = np.random.uniform(0.05, 0.95)

    if CO2_fresh is not None:
        room_params['CO2_fresh'] = CO2_fresh    # CO2 ppm of fresh air
    else:
        room_params['CO2_fresh'] = np.random.uniform(400, 800)

    if CO2_limit is not None:
        room_params['CO2_limit'] = CO2_limit
    else:
        room_params['CO2_limit'] = 1200

    if T_in_initial is not None:
        room_params['T_in_initial'] = T_in_initial  # Initial room temperature
    else:
        room_params['T_in_initial'] = 20

    if office_start_time is not None:
        room_params['office_start_time'] = office_start_time
    else:
        room_params['office_start_time'] = 7

    if office_end_time is not None:
        room_params['office_end_time'] = office_end_time
    else:
        room_params['office_end_time'] = 17

    if T_set_cool_ah is not None:
        room_params['T_set_cool_ah'] = T_set_cool_ah
    else:
        room_params['T_set_cool_ah'] = 28

    if T_set_cool_wh is not None:
        room_params['T_set_cool_wh'] = T_set_cool_wh
    else:
        room_params['T_set_cool_wh'] = 23

    if T_set_heat_ah is not None:
        room_params['T_set_heat_ah'] = T_set_heat_ah
    else:
        room_params['T_set_heat_ah'] = 16

    if T_set_heat_wh is not None:
        room_params['T_set_heat_wh'] = T_set_heat_wh
    else:
        room_params['T_set_heat_wh'] = 21

    if int_load_coef is not None:
        room_params['int_load_coef'] = int_load_coef
    else:
        room_params['int_load_coef'] = 0
        
    if valve_closed_pos is not None:
        room_params['valve_closed_pos'] = valve_closed_pos    
    else:
        room_params['valve_closed_pos'] = 0

    if valve_open_pos is not None:
        room_params['valve_open_pos'] = valve_open_pos
    else:
        room_params['valve_open_pos'] = 100

    # Factors to scale some parameters
    room_params['ci_factor'] = np.random.uniform(0.5, 5)
    room_params['cw_factor'] = np.random.uniform(0.1, 10)
    room_params['max_flow_factor'] = np.random.uniform(0.2, 2)
    room_params['max_cool_factor'] = np.random.uniform(0.6, 4)
    room_params['max_heat_factor'] = np.random.uniform(0.6, 4)


        # Subsequent calculations
    room_params['V'] = room_params['roomL1'] * room_params['roomL2'] * room_params['room_height']    # Room volume in m^3
    room_params['Ci'] = room_params['ci_factor'] * 800 * 700 * 2 * (room_params['roomL1'] + room_params['roomL2']) * room_params['room_height']*0.01    # Thermal capacistance (J/K) single plaster wall
    room_params['Riw'] = room_params['RiwU'] * room_params['wallQuota'] / (room_params['roomL1'] * room_params['room_height'])    # Heat loss coefficient (W/K)
    room_params['Rwo'] = room_params['RiwU'] * (1 - room_params['wallQuota']) / (room_params['roomL1'] * room_params['room_height']) # Heat loss coefficient (W/K)
    room_params['Cw'] = room_params['cw_factor'] * 900 * 1890 * room_params['roomL1'] * room_params['room_height'] * 0.11    # Envelope thermal capacistance J/K brick wall
    room_params['maxFlow'] = room_params['max_flow_factor'] * room_params['V'] * 10 / (60 * 60)  # max air flow into room in m^3/s
    
    # Sizing to heat or cool incoming air by 2 degrees and compensate for -15 and 30 deg C outdoor temp through wall, with 7 people in room in cooling case
    room_params['maxCoolPower'] = room_params['max_cool_factor'] * (room_params['maxFlow'] * 1005 * 1.205 * 2 + ((30 - 23) / (room_params['Riw'] + room_params['Rwo']) + 700))     
    room_params['maxHeatPower'] = room_params['max_heat_factor'] * (room_params['maxFlow'] * 1005 * 1.205 * 2 + (21 - (-15)) / (room_params['Riw'] + room_params['Rwo']))   

    # Time before we start penalizing the agent for not reaching the comfort zone (s)
    room_params['grace_period'] = 60 * 30
    return room_params