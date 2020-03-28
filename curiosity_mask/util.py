from gym.spaces import MultiDiscrete

def create_dummy_action_mask(ac_spaces: MultiDiscrete):
    action_mask = []
    for i, space_size in enumerate(ac_spaces.nvec):
        mask = None
        for j, size in enumerate(ac_spaces.nvec[i::-1]):
            if j == 0:
                mask = [1] * size
            else:
                mask = [mask] * size
        action_mask.append(mask)
    return action_mask
