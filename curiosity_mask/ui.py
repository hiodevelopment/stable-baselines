import sys
import PySimpleGUI as sg

class UI():
    
    def __init__(self):

        sg.set_options(text_justification='right')
        sg.theme('Reddit')
        # Running Gait
        
        teach_gait1 = [[sg.Text('Swinging Hip (Min)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=2, key='slider1-1a'),
                            sg.Text('Swinging Hip (Max)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=2, key='slider1-1b')],
                            [sg.Text('Swinging Knee (Min)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=-10, key='slider1-2a'),
                            sg.Text('Swinging Knee (Max)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=-10, key='slider1-2b')],
                            [sg.Text('Planted Hip (Min)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=-5, key='slider1-3a'),
                            sg.Text('Planted Hip (Max)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=1, key='slider1-3b')],
                            [sg.Text('Planted Knee (Min)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=1, key='slider1-4a'),
                            sg.Text('Planted Knee (Max)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=1, key='slider1-4b')] 
                            ]

        teach_gait2 = [[sg.Text('Swinging Hip (Min)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=-4, key='slider2-1a'),
                            sg.Text('Swinging Hip (Max)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=-4, key='slider2-1b')],
                            [sg.Text('Swinging Knee (Min)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=10, key='slider2-2a'),
                            sg.Text('Swinging Knee (Max)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=10, key='slider2-2b')],
                            [sg.Text('Planted Hip (Min)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=-3, key='slider2-3a'),
                            sg.Text('Planted Hip (Max)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=1, key='slider2-3b')],
                            [sg.Text('Planted Knee (Min)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=1, key='slider2-4a'),
                            sg.Text('Planted Knee (Max)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=1, key='slider2-4b')] 
                            ]

        teach_gait3 = [[sg.Text('Swinging Hip (Min)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=10, key='slider3-1a'),
                            sg.Text('Swinging Hip (Max)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=10, key='slider3-1b')],
                            [sg.Text('Swinging Knee (Min)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=-10, key='slider3-2a'),
                            sg.Text('Swinging Knee (Max)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=-10, key='slider3-2b')],
                            [sg.Text('Planted Hip (Min)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=-10, key='slider3-3a'),
                            sg.Text('Planted Hip (Max)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=2, key='slider3-3b')],
                            [sg.Text('Planted Knee (Min)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=1, key='slider3-4a'),
                            sg.Text('Planted Knee (Max)', size=(15, 1)), sg.Slider(range=(-10,10), orientation='h', size=(7, 10), default_value=1, key='slider3-4b')] 
                            ]

        layout = [ [sg.StatusBar('Gait Phase: Waiting to Start Training.', key='teaching-button-1')],
                [sg.Frame('Gait Phase 1: Lift Swinging Leg', teach_gait1, font='Any 12')],
                [sg.Frame('Gait Phase 2: Plant Swinging Leg', teach_gait2, font='Any 12')],
                [sg.Frame('Gait Phase 3: Switch Legs, Swing Leg',  teach_gait3,
                            font='Any 12')],
                [sg.Radio('Lesson 1', "lessons", key='radio-1', default=True),
    sg.Radio('Lesson 2', "lessons", key='radio-2'), sg.Checkbox('Push Swinging Leg Down', key='push-leg')],
                [sg.Submit(), sg.Button('Reset'), sg.Cancel(), sg.Button('Plant Leg')]]

        sg.set_options(text_justification='left')

        self.window = sg.Window('Machine Teaching Front End',
                        layout, font=("Helvetica", 12))


    


