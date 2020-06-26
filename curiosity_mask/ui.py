import sys
import PySimpleGUI as sg

class UI():
    
    def __init__(self):

        sg.set_options(text_justification='right')
        sg.theme('Reddit')

        teach_gait0 = [[sg.Text('Swinging Hip (Min)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=2, key='gait-0-swinging-hip-min'),
                            sg.Text('Swinging Hip (Max)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=2, key='gait-0-swinging-hip-max')],
                            [sg.Text('Swinging Knee (Min)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=-10, key='gait-0-swinging-knee-min'),
                            sg.Text('Swinging Knee (Max)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=-10, key='gait-0-swinging-knee-max')],
                            [sg.Text('Planted Hip (Min)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=-5, key='gait-0-planted-hip-min'),
                            sg.Text('Planted Hip (Max)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=1, key='gait-0-planted-hip-max')],
                            [sg.Text('Planted Knee (Min)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=1, key='gait-0-planted-knee-min'),
                            sg.Text('Planted Knee (Max)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=1, key='gait-0-planted-knee-max')] 
                            ]
        
        teach_gait1 = [[sg.Text('Swinging Hip (Min)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=2, key='gait-1-swinging-hip-min'),
                            sg.Text('Swinging Hip (Max)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=2, key='gait-1-swinging-hip-max')],
                            [sg.Text('Swinging Knee (Min)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=-1, key='gait-1-swinging-knee-min'),
                            sg.Text('Swinging Knee (Max)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=-1, key='gait-1-swinging-knee-max')],
                            [sg.Text('Planted Hip (Min)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=-10, key='gait-1-planted-hip-min'),
                            sg.Text('Planted Hip (Max)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=1, key='gait-1-planted-hip-max')],
                            [sg.Text('Planted Knee (Min)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=1, key='gait-1-planted-knee-min'),
                            sg.Text('Planted Knee (Max)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=6, key='gait-1-planted-knee-max')] 
                            ]

        teach_gait2 = [[sg.Text('Swinging Hip (Min)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=-6, key='gait-2-swinging-hip-min'),
                            sg.Text('Swinging Hip (Max)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=-6, key='gait-2-swinging-hip-max')],
                            [sg.Text('Swinging Knee (Min)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=1, key='gait-2-swinging-knee-min'),
                            sg.Text('Swinging Knee (Max)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=1, key='gait-2-swinging-knee-max')],
                            [sg.Text('Planted Hip (Min)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=-3, key='gait-2-planted-hip-min'),
                            sg.Text('Planted Hip (Max)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=1, key='gait-2-planted-hip-max')],
                            [sg.Text('Planted Knee (Min)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=1, key='gait-2-planted-knee-min'),
                            sg.Text('Planted Knee (Max)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=1, key='gait-2-planted-knee-max')] 
                            ]

        teach_gait3 = [[sg.Text('Swinging Hip (Min)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=10, key='gait-3-swinging-hip-min'),
                            sg.Text('Swinging Hip (Max)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=10, key='gait-3-swinging-hip-max')],
                            [sg.Text('Swinging Knee (Min)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=-10, key='gait-3-swinging-knee-min'),
                            sg.Text('Swinging Knee (Max)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=-10, key='gait-3-swinging-knee-max')],
                            [sg.Text('Planted Hip (Min)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=-3, key='gait-3-planted-hip-min'),
                            sg.Text('Planted Hip (Max)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=2, key='gait-3-planted-hip-max')],
                            [sg.Text('Planted Knee (Min)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=1, key='gait-3-planted-knee-min'),
                            sg.Text('Planted Knee (Max)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=5, key='gait-3-planted-knee-max')] 
                            ]

        layout = [ [sg.StatusBar('Gait Phase: Waiting to Start Training.', key='teaching-button-1')],
                [sg.Frame('Gait Phase 0: Start', teach_gait0, font='Any 12')],
                [sg.Frame('Gait Phase 1: Lift Swinging Leg', teach_gait1, font='Any 12')],
                [sg.Frame('Gait Phase 2: Plant Swinging Leg', teach_gait2, font='Any 12')],
                [sg.Frame('Gait Phase 3: Switch Legs, Swing Leg',  teach_gait3,
                            font='Any 12')],
                [sg.Radio('Lesson 1: Practice first step.', "lessons", key='radio-1', default=True),
    sg.Radio('Lesson 2', "lessons", key='radio-2')],
                [sg.Submit(), sg.Cancel()]]

        sg.set_options(text_justification='left')

        self.window = sg.Window('Machine Teaching Interface - Bipedal Walker',
                        layout, font=("Helvetica", 12))


    


