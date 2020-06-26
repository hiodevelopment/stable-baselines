from gym.spaces import MultiDiscrete

def create_dummy_action_mask(ac_spaces: MultiDiscrete):
    action_mask = []
    for i, space_size in enumerate(ac_spaces.nvec):
        mask = None
        for j, size in enumerate(ac_spaces.nvec[i::-1]):
            if j == 0:
                mask = [0] * size
            else:
                mask = [mask] * size
        action_mask.append(mask)
    return action_mask

def create_negative_action_mask(ac_spaces: MultiDiscrete):
    action_mask = []
    for i, space_size in enumerate(ac_spaces.nvec):
        mask = None
        for j, size in enumerate(ac_spaces.nvec[i::-1]):
            if j == 0:
                mask = [0] * size
            else:
                mask = [mask] * size
        action_mask.append(mask)
    return action_mask

#action_space = MultiDiscrete([3, 2, 3, 2, 2])
#print(create_dummy_action_mask(action_space)[2])

# 0 - Gait [G1, G2, G3]
# 1 - Left Hip for each gait [[G1xLH1, G1xLH2], [G2xLH1, G2xLH2], [G3xLH1, G3xLH2]]
# 2 - Left Knee for each gait x Left Hip [[[G1xLH1xLK1, G1xLH1xLK2], [G1xLH2xLK1, G1xLH2xLK2]], [[G2xLH1xLK1, G2xLH1xLK2], [G2xLH2xLK1, G2xLH2xLK2]], [[G3xLH1xLK1, G3xLH1xLK2], [G3xLH2xLK1, G3xLH2xLK2]]]
# 3 - Right Hip for each gait x Left Hip x Left Knee
# 4 - Right Knee for each gait x Left Hip x Left Knee x Right Hip

left_hip = left_knee = right_hip = right_knee = range(21)

def set_action_mask_gait(gait, ranges, mask):
    for left_hip_index in list(filter(lambda x: x >= ranges['left_hip']['min'] and x <= ranges['left_hip']['max'], left_hip)): 
        mask[1][gait][left_hip_index] = 1
        for left_knee_index in list(filter(lambda x: x >= ranges['left_knee']['min'] and x <= ranges['left_knee']['max'], left_knee)): 
            mask[2][gait][left_hip_index][left_knee_index] = 1
            for right_hip_index in list(filter(lambda x: x >= ranges['right_hip']['min'] and x <= ranges['right_hip']['max'], right_hip)): 
                mask[3][gait][left_hip_index][left_knee_index][right_hip_index] = 1
                for right_knee_index in list(filter(lambda x: x >= ranges['right_knee']['min'] and x <= ranges['right_knee']['max'], right_knee)): 
                    mask[4][gait][left_hip_index][left_knee_index][right_hip_index][right_knee_index] = 1
    return mask


def test_mask(action, state, teaching, swinging_leg):
    
    if state == 'start' or state == 'lift_leg':
        gait = 0
    if state == 'plant_leg':
        gait = 1
    if state == 'switch_leg':
        gait = 2

    gait_ref = str(gait+1)
    
    if swinging_leg == 'left':

            ranges = {'left_hip': {'min': teaching['gait-' + gait_ref + '-swinging-hip-min'], 'max': teaching['gait-' + gait_ref + '-swinging-hip-max']},
                    'left_knee': {'min': teaching['gait-' + gait_ref + '-swinging-knee-min'], 'max': teaching['gait-' + gait_ref + '-swinging-knee-max']}, 
                    'right_hip': {'min': teaching['gait-' + gait_ref + '-planted-hip-min'], 'max': teaching['gait-' + gait_ref + '-planted-hip-max']}, 
                    'right_knee': {'min': teaching['gait-' + gait_ref + '-planted-knee-min'], 'max': teaching['gait-' + gait_ref + '-planted-knee-max']}
                }
    if swinging_leg == 'right':

            ranges = {'left_hip': {'min': teaching['gait-' + gait_ref + '-planted-hip-min'], 'max': teaching['gait-' + gait_ref + '-planted-hip-max']},
                'left_knee': {'min': teaching['gait-' + gait_ref + '-planted-knee-min'], 'max': teaching['gait-' + gait_ref + '-planted-knee-max']}, 
                'right_hip': {'min': teaching['gait-' + gait_ref + '-swinging-hip-min'], 'max': teaching['gait-' + gait_ref + '-swinging-hip-max']}, 
                'right_knee': {'min': teaching['gait-' + gait_ref + '-swinging-knee-min'], 'max': teaching['gait-' + gait_ref + '-swinging-knee-max']}
            }

    action_list = []
    for left_hip_index in list(filter(lambda x: x >= ranges['left_hip']['min'] and x <= ranges['left_hip']['max'], left_hip)):
        for left_knee_index in list(filter(lambda x: x >= ranges['left_knee']['min'] and x <= ranges['left_knee']['max'], left_knee)):
            for right_hip_index in list(filter(lambda x: x >= ranges['right_hip']['min'] and x <= ranges['right_hip']['max'], right_hip)):
                for right_knee_index in list(filter(lambda x: x >= ranges['right_knee']['min'] and x <= ranges['right_knee']['max'], right_knee)):
                        action_list.append((left_hip_index, left_knee_index, right_hip_index, right_knee_index))
    #print(action_list)
    return action in action_list
