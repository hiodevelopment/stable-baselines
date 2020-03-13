import numpy as np
import pandas as pd
from scipy import linalg, integrate, interpolate
import os

class Environment(object):
    '''
    Environment simulator is decoupled from the room simulator to be able to run the same room dynamics with different environment.
    '''
    def __init__(self,
                 sim_params: dict = {},
                 room_params: dict = {},
                 data_folder: str = 'data/',
                 weather_file: str = 'weather.csv',
                 weather_selection: str = 'default', # Default, random or deterministic (which requires specifying the case name)
                 weather_case: int = None,  # Column index, must be >=1
                 weather_data_granularity: float = 3600, # in seconds, default is hourly data
                 weather_data_shift: int = 0, # Shifting the data to increase randomization, in days. Between 0-365.
                                              #  This is because most weather data starts in January 1st
                 occupancy_file: str = 'occupancy.csv',
                 occupancy_selection: str = 'default',  # Default, random or deterministic (which requires specifying the case name)
                 occupancy_case: int = None,
                 occupancy_data_granularity: float = 3600,  # in seconds, default is hourly data
                 occupancy_data_shift: int = 0
                 ):
        dirname = os.path.dirname(__file__)

        self.sim_params = sim_params
        self.office_hours = np.logical_and((sim_params['tH'] % 24) > room_params['office_start_time'],
                                           (sim_params['tH'] % 24) < room_params['office_end_time'])

        # Get the weather data
        weather_path = os.path.join(dirname, data_folder + weather_file)
        df_weather_data = pd.read_csv(weather_path)
        if weather_selection == 'default':
            temperature_col = np.array(df_weather_data['default'])[1:]  # First row is dedicated to weights
        elif weather_selection == 'deterministic':  # Requires specifying the case index
            weather_case_ix = weather_case % len(df_weather_data.columns[1:])
            temperature_col = np.array(df_weather_data.iloc[:, weather_case_ix + 1])[1:]  # First row is dedicated to weights
            #print("Deterministically selected weather data:", df_weather_data.columns[weather_case_ix + 1])
        elif weather_selection == 'random':
            # Randomly select a case
            weights = np.array(df_weather_data.iloc[0,1:])  # Get the weights
            p = weights / weights.sum()     # Weights -> prob
            weather_case = np.random.choice(a = range(len(p)), p=p) + 1 # Randomly select column (first column is index, so shift)
            temperature_col = np.array(df_weather_data.iloc[:, weather_case])[1:]  # First row is dedicated to weights
            #print("Randomly selected weather case:", weather_case, df_weather_data.columns[weather_case])
        else:
            raise Exception("Invalid weather selection:", weather_selection)

        # Shifting the weather data for randomization purposes
        weather_data_shift_mod = (weather_data_shift * 24) % len(temperature_col)
        temperature_col = np.roll(temperature_col, weather_data_shift_mod) # convert days into hours of shift
        #print("Starting data from hour", weather_data_shift_mod)
        weather_time = np.array(range(len(temperature_col))) * weather_data_granularity
        #self.weather_data = np.loadtxt(weather_path, delimiter=',')[:, 0:2]
        self.weather_data = np.transpose(np.concatenate([[weather_time], [temperature_col]], axis=0))
        self.T_out_vec = np.interp(sim_params['t'],
                                   self.weather_data[:, 0],
                                   self.weather_data[:, 1])

        # Get the occupancy data
        occupancy_path = os.path.join(dirname, data_folder + occupancy_file)
        df_occupancy_data = pd.read_csv(occupancy_path)
        if occupancy_selection == 'default':
            occupancy_col = np.array(df_occupancy_data['default'])[1:]  # First row is dedicated to weights
        elif occupancy_selection == 'deterministic':  # Requires specifying the case index
            occupancy_case_ix = occupancy_case % len(df_occupancy_data.columns[1:])
            occupancy_col = np.array(df_occupancy_data.iloc[:, occupancy_case_ix + 1])[1:]  # First row is dedicated to weights
            #print("Deterministically selected occupancy data:", df_occupancy_data.columns[occupancy_case_ix + 1])
        elif occupancy_selection == 'random':
            weights = np.array(df_occupancy_data.iloc[0, 1:])  # Get the weights
            p = weights / weights.sum()  # Weights -> prob
            occupancy_case = np.random.choice(a=range(len(p)),
                                            p=p) + 1  # Randomly select column (first column is index, so shift)
            occupancy_col = np.array(df_occupancy_data.iloc[:, occupancy_case])[1:]  # First row is dedicated to weights
            #print("Randomly selected occupancy case:", occupancy_case, df_occupancy_data.columns[occupancy_case])
        else:
            raise Exception("Invalid occupancy selection:", occupancy_selection)

        occupancy_data_shift_mod = (occupancy_data_shift * 24) % len(occupancy_col)
        #print("Starting from hour", occupancy_data_shift_mod)
        occupancy_col = np.roll(occupancy_col, occupancy_data_shift_mod)  # convert days into hours of shift
        occupancy_time = np.array(range(len(occupancy_col))) * occupancy_data_granularity
        self.occupancy_data = np.transpose(np.concatenate([[occupancy_time], [occupancy_col]], axis=0))
        #self.occupancy_data = np.loadtxt(os.path.join(dirname, data_folder + occupancy_file), delimiter=',')
        self.occInterp = interpolate.interp1d(self.occupancy_data[:, 0],
                                              self.occupancy_data[:, 1],
                                              kind='nearest')
        self.n_people_vec = np.array([self.occInterp(ti) for ti in sim_params['t']])
        self.n_people_vec[np.logical_not(self.office_hours)] = 0

        self.T_set_cool_vec = room_params['T_set_cool_ah'] * np.ones(sim_params['N'])
        self.T_set_cool_vec[self.office_hours] = room_params['T_set_cool_wh']

        self.T_set_heat_vec = room_params['T_set_heat_ah'] * np.ones(sim_params['N'])
        self.T_set_heat_vec[self.office_hours] = room_params['T_set_heat_wh']

        # zero!
        self.intLoad_vec = room_params['int_load_coef'] * 200 * (0.5 * np.sin(1. / 24 * 2 * np.pi * sim_params['tH']) + 0.5)


    def get_env_signals(self, k):
        env_signals = {}
        env_signals['T_out'] = self.T_out_vec[k % len(self.T_out_vec)]
        env_signals['n_people'] = self.n_people_vec[k % len(self.n_people_vec)]
        env_signals['int_load'] = self.intLoad_vec[k % len(self.intLoad_vec)]
        env_signals['T_set_cool'] = self.T_set_cool_vec[k % len(self.T_set_cool_vec)]
        env_signals['T_set_heat'] = self.T_set_heat_vec[k % len(self.T_set_heat_vec)]
        return env_signals