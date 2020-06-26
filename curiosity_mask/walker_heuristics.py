    # The leg that is further ahead is leading.
    @Rule(AS.leg <<
        Leg(side=L('left'), position=MATCH.p1),
        Leg(side=L('right'), position=MATCH.p2,),
        TEST(lambda p1, p2: p1 if (p1>p2) else p2)
        )
    #@Rule(AS.leg << Leg(side=L('left')))
    def determine_leading_leg(self, leg, p1, p2):
        self.duplicate(leg, orientation='leading')
        #print('leading leg set: ', leg['side'], leg['position'], p1, p2)
        pass
    
    # The leg that is further behind is trailing.
    @Rule(AS.leg << 
        Leg(side=L('right'), position=MATCH.p1),
        Leg(side=L('left'), position=MATCH.p2,),
        TEST(lambda p1, p2: p1 if (p1>p2) else p2)
        )
    #@Rule(AS.leg << Leg(side=L('right')))
    def determine_trailing_leg(self, leg, p1, p2):
        self.duplicate(leg, orientation='trailing')
        #print('trailing leg set: ', leg['side'], leg['position'], p1, p2)
        pass

    # Lift trailing leg (more than swinging) when both feet are on the ground.. 
    @Rule(AS.leg << Leg(orientation=L('trailing')) & Leg(contact=L('1.0')))
    def define_lifting(self, leg):
        self.duplicate(leg, function='lifting')
        #print('lifting leg set: ', leg['side'], leg['position'], leg['orientation'])
        pass
    """
    # Swinging leg is leading and not touching the ground. 
    @Rule(AS.leg << Leg(orientation=L('leading')) & Leg(contact=L('0.0')))
    def define_swinging(self, leg):
        self.duplicate(leg, function='swinging')
        #print('swinging leg set: ', leg['side'], leg['position'], leg['orientation'])
        pass
    """
    # Planted leg is leading and in contact with the ground. , contact=1.0
    @Rule(AS.leg << Leg(orientation=L('leading')))
    def define_planted(self, leg):
        self.duplicate(leg, function='planted')
        #print('planted leg set: ', leg['side'], leg['position'], leg['orientation'])
        pass

    # Plant right leg, lift left leg.
    @Rule(Leg(function='planted', side='right') & Leg(function='lifting', side='left'))
    def right_plant_left_lift(self):
        """
        Right leg planted, left leg lifting. 
        """
        # act on mask
        a = b = c = d = 21
        self.mask_right_hip = [[[1 if action_index > 10 else 0 for action_index in range(c)] for y in range(b)] for x in range(a)]
        self.valid_actions[2] = self.mask_right_hip

        self.mask_left_hip = [1 if action_index <= 6 else 0 for action_index in range(a)] # Equivalent of lift leg higher (than swinging)
        self.valid_actions[0] = self.mask_left_hip

        """
        # Keep the planted leg on the ground.
        def forecast(x,y,z,j):
            return self.possible_states[x][y][z][j][0]['right_leg_contact']
        
        left_hip = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        left_knee = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        right_hip = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        right_knee = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        result = [[[[forecast(x, y, z, j) for j in right_knee] for z in right_hip]for y in left_knee] for x in left_hip]
        print('result of lookup: ', result)
        sys.exit()
        """

        #print('right leg planted, left leg lifting')
        return self.valid_actions
    
    # Plant left leg, lift right leg.
    @Rule(Leg(function='planted', side='left') & Leg(function='lifting', side='right'))
    def left_plant_right_lift(self):
        
        #Left leg planted, right leg lifting. 
        
        # act on mask
        a = b = c = d = 21
        self.mask_left_hip = [1 if action_index == 9 else 0 for action_index in range(a)]
        self.valid_actions[0] = self.mask_left_hip

        self.mask_left_knee = [[1 if action_index == 21 else 0 for action_index in range(b)] for x in range(a)]
        self.valid_actions[1] = self.mask_left_knee

        self.mask_right_hip = [[[1 if action_index > 10 else 0 for action_index in range(c)] for y in range(b)] for x in range(a)] # Equivalent of lift leg higher (than swinging)
        self.valid_actions[2] = self.mask_right_hip
        
        print('left leg planted, right leg lifting')
        return self.valid_actions

    """
    @Rule(Leg(function='planted', side='left'))
    def planted_leg(self):

        #Modify the mask for the planted leg.  Hip rotates forward.  Left hip action mask. 
    
        # act on mask
        a = b = c = d = 21
        self.mask_left_hip = [1 if action_index > 10 else 0 for action_index in range(a)]
        self.valid_actions[0] = self.mask_left_hip
        print('planted left rule')
        return self.valid_actions

    @Rule(Leg(function='planted', side='right'))
    def planted_leg(self):
        
        #Modify the mask for the planted leg.  Hip rotates forward.  Right hip action mask. 
        
        # act on mask
        a = b = c = d = 21
        self.mask_right_hip = [[[1 if action_index > 10 else 0 for action_index in range(c)] for y in range(b)] for x in range(a)]
        self.valid_actions[2] = self.mask_right_hip
        print('planted right rule')
        return self.valid_actions

    @Rule(Leg(function='swinging', side='left'))
    def swinging_leg(self):
        
        #Modify the mask for the swinging leg.  Hip rotates backward. For now, raise.  Left hip action mask.
        
        # act on mask
        a = b = c = d = 21
        self.mask_left_hip = [1 if action_index <= 10 else 0 for action_index in range(a)]
        self.valid_actions[0] = self.mask_left_hip

        return self.valid_actions

    @Rule(Leg(function='swinging', side='right'))
    def swinging_leg(self):
        
        #Modify the mask for the swinging leg.  Hip rotates backward. For now, raise.  Right hip action mask.
        
        # act on mask
        a = b = c = d = 21
        self.mask_right_hip = [[[1 if action_index <= 10 else 0 for action_index in range(c)] for y in range(b)] for x in range(a)]
        self.valid_actions[2] = self.mask_right_hip

        return self.valid_actions

    @Rule(Leg(function='lifting', side='left'))
    def lifting_leg(self):
        
        #Modify the mask for the lifting leg.  Hip rotates backward. For now, raise.  Left hip action mask.
        
        # act on mask
        a = b = c = d = 21
        self.mask_left_hip = [1 if action_index <= 6 else 0 for action_index in range(a)] # Equivalent of lift leg higher (than swinging)
        self.valid_actions[0] = self.mask_left_hip
        print('lifting left rule')
        return self.valid_actions

    @Rule(Leg(function='lifting', side='right'))
    def lifting_leg(self):
        
        #Modify the mask for the planted leg.  Hip rotates backward. For now, raise.  Right hip action mask.
        
        # act on mask
        a = b = c = d = 21
        self.mask_right_hip = [[[1 if action_index <= 6 else 0 for action_index in range(c)] for y in range(b)] for x in range(a)] # Equivalent of lift leg higher (than swinging)
        self.valid_actions[2] = self.mask_right_hip
        print('lifting right rule')
        return self.valid_actions
    """

    ###################################

    # Widen the mask based on reward conditions. 
        """
        # Reward based criteria for widening the mask"
        reward = self.training_env.env_method('get_reward')[0]
        if reward > 60: # 60
            self.engine.mask_width = 2
        elif reward > 62: # 70
            self.engine.mask_width = 3
        elif reward > 65:
            self.engine.mask_width = 4
        """

        ##############################

        # Filter the actions based on the states that will result from taking each of them. 
        """
        # Run through all the actions to generate the resulting state vectors.
        left_hip = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        left_knee = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        right_hip = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        right_knee = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

        #self.engine.possible_states = [[[[self.training_env.env_method('simulate', x, y, z, j) for j in right_knee] for z in right_hip]for y in left_knee] for x in left_hip]
        #print('possible states: ', self.engine.possible_states[0][0][0][0][0])  # x, y, z, j then [0] 
        self.engine.possible_states = []
        for x_index, x_value in enumerate(left_hip):
            for y_index, y_value in enumerate(left_knee):
                for z_index, z_value in enumerate(right_hip):
                    for j_index, j_value in enumerate(right_knee):
                        #print(self.engine.mask_left_hip[x_index])
                        if self.engine.mask_left_hip[x_index] == 1 and self.engine.mask_right_hip[x_index][y_index][z_index] == 1 and self.training_env.env_method('simulate', x_value, y_value, z_value, j_value)[0] == 0.0:
                            self.engine.possible_states.append((y_value,j_value))
        print('possible states: ', len(self.engine.possible_states))
        sys.exit()
        """

        ##################

        # Log variables to Tensorboard as needed. 
        """
        # Log scalar value to Tensorboard
        co2 = o['CO2']
        energy = h['energy_elec']
        comfort = h['error_T_real']
        summary = tf.Summary(value=[tf.Summary.Value(tag='air_quality', simple_value=co2), tf.Summary.Value(tag='energy', simple_value=energy), tf.Summary.Value(tag='comfort', simple_value=comfort)])
        #summary2 = tf.Summary(value=[tf.Summary.Value(tag='energy', simple_value=energy)])
        #summary3 = tf.Summary(value=[tf.Summary.Value(tag='comfort', simple_value=comfort)])
        self.locals['writer'].add_summary(summary, self.num_timesteps)
        """

        ###### Graveyard of old rules

        """
    @Rule(Fact(concept=3), Fact(function='swinging'))
    def test2(self): 
        gait = 2
        self.valid_actions[0] = [0, 1, 0]  # gait phase 1
        self.mask_left_hip = self.valid_actions[1][gait-1] = [1 if action_index == 12 else 0 for action_index in range(21)] # Positive hip motion for planted leg.
        #print(self.mask_left_hip)
        #sys.exit()
        for joint1_value in range(21):
            self.mask_left_knee = self.valid_actions[2][gait-1][joint1_value] = [1 if action_index == 6 else 0 for action_index in range(21)] # Positive hip motion for swinging leg.
        for joint1_value in range(21):
                for joint2_value in range(21):
                    self.mask_right_hip = self.valid_actions[3][gait-1][joint1_value][joint2_value] = [1 if action_index == 12 else 0 for action_index in range(21)] # Slight Positive knee motion for planted leg.
        for joint1_value in range(21):
            for joint2_value in range(21):
                for joint3_value in range(21):
                    self.mask_right_knee = self.valid_actions[4][gait-1][joint1_value][joint2_value][joint3_value] = [1 if action_index == 0 else 0 for action_index in range(21)] # Negative knee motion for swinging leg.
    
        #print('Concept 2 Activated')
        pass
      """  
     
    @Rule(Fact(concept=1), AS.swinging_leg << Fact(function='swinging', side=MATCH.side_swinging),
            AS.planted_leg << Fact(function='planted', side=MATCH.side_planted))  # Not fuzzy.
    def gait_phase_1(self, swinging_leg, planted_leg, side_swinging, side_planted):  # Deterministic Phase 1.
        # Each concept rule needs to mask the gait and flip bits for control concept.
        self.valid_actions[0] = [1, 0, 0]  # gait phase 1
        # Flip bits for joint modifications. (Left Hip, Right Hip, Left Knee, Right Knee)
        
        # action_space[0] is the gait
        # action_space[1] is joint 1 for each gait
        # action_space[2] is joint 2 for each gait, then for each possible value of joint 1
        # action_space[3] is joint 3 for each gait, then then for each possible value of joint 1, then for joint 2
        # action_space[4] is joint 4 for each gait, then then for each possible value of joint 1, then for joint 2, then for joint 3
        gait = 1
        
        """
        if swinging_leg['side'] == 'left':
            self.mask_left_hip = self.valid_actions[1][gait] = [1 if action_index >= 10 else 0 for action_index in range(21)] # Positive hip motion for swinging leg.
            for joint1_value in range(20):
                self.mask_left_knee = self.valid_actions[2][gait][joint1_value] = [1 if action_index < 9 else 0 for action_index in range(21)] # Negative knee motion for swinging leg.
        elif swinging_leg['side'] == 'right':
            for joint1_value in range(20):
                for joint2_value in range(20):
                    self.mask_right_hip = self.valid_actions[3][gait][joint1_value][joint2_value] = [1 if action_index >= 10 else 0 for action_index in range(21)] # Positive hip motion for swinging leg.
            for joint1_value in range(20):
                for joint2_value in range(20):
                    for joint3_value in range(20):
                        self.mask_right_knee = self.valid_actions[4][gait][joint1_value][joint2_value][joint3_value] = [1 if action_index < 9 else 0 for action_index in range(21)] # Negative knee motion for swinging leg.
        """
        #print('Concept 1 Activated', self.valid_actions[0])
        pass
    
    ####### Transtion: Gait Phase 1 -> 2, If planted leg is behind -> put swinging leg down: plant swinging leg
    @Rule(Leg(gait=L(1)) & Leg(function=L('planted')) & Leg(orientation=L('trailing')))
    def define_swinging(self):
        orchestration = self.declare(Orchestration(gait=1, event='selection', condition='fuzzy'))
        print('Fuzzy condition between gait phase 1 and 2: ', orchestration['event'])
        pass

    @Rule(AS.orchestration << Orchestration(gait=L(1)) & Orchestration(event=L('selection')) & Orchestration(condition=L('fuzzy')), salience=1)   # Fuzzy, gait choice only. Let it learn what to do in transition. 
    def transition_phase_2(self, orchestration):
        # Each concept rule needs to mask the gait and flip bits for control concept.
        self.valid_actions[0] = [1, 1, 0]  # gait phase 1 or gait phase 2
        self.retract(orchestration)
        print('Gait Transition -> Phase 2')
        pass

    ###### Concept: Gait Phase 2 #####  
    @Rule(Orchestration(gait=L(2)) & NOT(Orchestration(event=L('selection'))), AS.swinging_leg << Leg(gait=L(2)) & Leg(function=L('swinging')),
            AS.planted_leg << Leg(gait=L(2)) & Leg(function=L('planted'), side=MATCH.side), salience=0)  # Not fuzzy.
    def gait_phase_2(self, swinging_leg, planted_leg):  # Deterministic Phase 1.
        # Each concept rule needs to mask the gait and flip bits for control concept.
        self.valid_actions[0] = [1, 0, 0]  # gait phase 1
        # Flip bits for joint modifications. (Left Hip, Left Knee, Right Hip, Right Knee) 

        if swinging_leg[side] == 'left':
            for mask in range(20):
                self.valid_actions[4][mask][0][0][0] = [1 if action_index >= 10 else 0 for action_index in range(21)] # Negative hip motion for swinging leg.
            for mask in range(20):
                self.valid_actions[4][0][0][mask][0] = [1 if action_index < 9 else 0 for action_index in range(21)] # Positive knee motion for swinging leg.
        elif swinging_leg[side] == 'right':
            for mask in range(20):
                self.valid_actions[4][0][mask][0][0] = [1 if action_index < 9 else 0 for action_index in range(21)] # Negative hip motion for swinging leg.
            for mask in range(20):
                self.valid_actions[4][0][0][0][mask] = [1 if action_index >= 10 else 0 for action_index in range(21)] # Positive knee motion for swinging leg.
        print('Concept 2 Activated')
        pass

    
    ##### Transition: Gait Phase 2 -> 3: Push off planted leg, if swinging leg makes contact with the ground.
    @Rule(Leg(gait=L(2)) & Leg(function=L('swinging')) & Leg(contact=L(True)))
    def define_swinging(self):
        orchestration = self.declare(Orchestration(gait=2, event='selection', condition='fuzzy'))
        print('Fuzzy condition between gait phase 2 and 3: ', orchestration['event'])
        pass
 
    @Rule(AS.swinging_leg << Leg(gait=L(2)) & Leg(function=L('swinging')) & Leg(contact=L(True)),
            AS.planted_leg << Leg(gait=L(2)) & Leg(function=L('planted')), salience=1)
    def transition_phase_3(self, swinging_leg, planted_leg):
        # Each concept rule needs to mask the gait and flip bits for control concept.
        self.valid_actions[0] = [0, 1, 1]  # gait phase 2 or 3
        self.modify (swinging_leg, function='planted', gait='3') # Remember to check for modify effects. 
        self.modify (planted_leg, function='swinging', gait='3')
        print('Gait Transition -> Phase 3')
        pass

    ###### Concept: Gait Phase 3 #####  
    @Rule(Orchestration(gait=L(3)) & NOT(Orchestration(event=L('selection'))), AS.swinging_leg << Leg(gait=L(3)) & Leg(function=L('swinging')),
            AS.planted_leg << Leg(gait=L(3)) & Leg(function=L('planted'), side=MATCH.side), salience=0)  # Not fuzzy.
    def gait_phase_3(self, swinging_leg, planted_leg):  # Deterministic Phase 3.
        # Each concept rule needs to mask the gait and flip bits for control concept.
        self.valid_actions[0] = [0, 0, 1]  # gait phase 3
        # Flip bits for joint modifications. (Left Hip, Left Knee, Right Hip, Right Knee) 
        pass 

    ##### Transition: Gait Phase 3 -> 1: Push Off.
    # Heurisitc: This heuristic is related to the size of the supporting base and the x velocity in relation to the speed the gait is validated for.
    @Rule(AS.orchestration << Orchestration(event=L('selection')),
            AS.left_leg << Leg(gait=L(3)) & Leg(side=L('left')), AS.right_leg << Leg(gait=L(3)) & Leg(side=L('right')), salience=1)
    def transition_phase_1(self, orchestration):
        #Each concept rule needs to mask the gait and flip bits for control concept.
        self.valid_actions[0] = [1, 0, 1]  # gait phase 1 or 3
        self.retract(orchestration)
        print('Gait Transition -> Phase 1')
        pass


###################

"""
    def set_mask_start(self, event): 
        
        gait = 1
        teaching = self.teaching

        # Flip bits to teach the strategy for this gait. (Left Hip, Left Knee, Right, Hip, Right Knee)
        if self.swinging_leg == 'left':
            self.action_mask[1][gait-1] = [1 if (action_index >= teaching['slider1-1a'] and action_index <= teaching['slider1-1b']) else 0 for action_index in range(21)] # Positive hip motion for planted leg.
            for joint1_value in range(21):
                self.action_mask[2][gait-1][joint1_value] = [1 if (action_index >= teaching['slider1-2a'] and action_index <= teaching['slider1-2b']) else 0 for action_index in range(21)] # Negative hip motion for swinging leg.
            for joint1_value in range(21):
                    for joint2_value in range(21):
                        self.action_mask[3][gait-1][joint1_value][joint2_value] = [1 if (action_index >= teaching['slider1-3a'] and action_index <= teaching['slider1-3b']) else 0 for action_index in range(21)] # Slight Positive knee motion for planted leg.
            for joint1_value in range(21):
                for joint2_value in range(21):
                    for joint3_value in range(21):
                        self.action_mask[4][gait-1][joint1_value][joint2_value][joint3_value] = [1 if (action_index >= teaching['slider1-4a'] and action_index <= teaching['slider1-4b']) else 0 for action_index in range(21)] # Negative knee motion for swinging leg.

        if self.swinging_leg == 'right':
            self.action_mask[1][gait-1] = [1 if (action_index >= teaching['slider1-3a'] and action_index <= teaching['slider1-3b']) else 0 for action_index in range(21)] # Positive hip motion for planted leg.
            for joint1_value in range(21):
                self.action_mask[2][gait-1][joint1_value] = [1 if (action_index >= teaching['slider1-4a'] and action_index <= teaching['slider1-4b']) else 0 for action_index in range(21)] # Negative hip motion for swinging leg.
            for joint1_value in range(21):
                    for joint2_value in range(21):
                        self.action_mask[3][gait-1][joint1_value][joint2_value] = [1 if (action_index >= 20 and action_index <= 20) else 0 for action_index in range(21)] # Slight Positive knee motion for planted leg.
            for joint1_value in range(21):
                for joint2_value in range(21):
                    for joint3_value in range(21):
                        self.action_mask[4][gait-1][joint1_value][joint2_value][joint3_value] = [1 if (action_index >= teaching['slider1-2a'] and action_index <= teaching['slider1-2b']) else 0 for action_index in range(21)] # Negative knee motion for swinging leg.
        pass

    def set_mask_gait_1(self, event): 
        
        gait = 1
        teaching = self.teaching
        
        # Flip bits to teach this gait strategy. (Left Hip, Left Knee, Right, Hip, Right Knee)
        if self.swinging_leg == 'left':

            self.action_mask[1][gait-1] = [1 if (action_index >= teaching['slider1-1a'] and action_index <= teaching['slider1-1b']) else 0 for action_index in range(21)] # Positive hip motion for planted leg.
            for joint1_value in range(21):
                self.action_mask[2][gait-1][joint1_value] = [1 if (action_index >= teaching['slider1-2a'] and action_index <= teaching['slider1-2b']) else 0 for action_index in range(21)] # Negative hip motion for swinging leg.
            for joint1_value in range(21):
                    for joint2_value in range(21):
                        self.action_mask[3][gait-1][joint1_value][joint2_value] = [1 if (action_index >= teaching['slider1-3a'] and action_index <= teaching['slider1-3b']) else 0 for action_index in range(21)] # Slight Positive knee motion for planted leg.
            for joint1_value in range(21):
                for joint2_value in range(21):
                    for joint3_value in range(21):
                        self.action_mask[4][gait-1][joint1_value][joint2_value][joint3_value] = [1 if (action_index >= teaching['slider1-4a'] and action_index <= teaching['slider1-4b']) else 0 for action_index in range(21)] # Negative knee motion for swinging leg.
            
        if self.swinging_leg == 'right':

            self.action_mask[1][gait-1] = [1 if (action_index >= teaching['slider1-3a'] and action_index <= teaching['slider1-3b']) else 0 for action_index in range(21)] # Positive hip motion for planted leg.
            for joint1_value in range(21):
                self.action_mask[2][gait-1][joint1_value] = [1 if (action_index >= teaching['slider1-4a'] and action_index <= teaching['slider1-4b']) else 0 for action_index in range(21)] # Negative hip motion for swinging leg.
            for joint1_value in range(21):
                    for joint2_value in range(21):
                        self.action_mask[3][gait-1][joint1_value][joint2_value] = [1 if (action_index >= teaching['slider1-1a'] and action_index <= teaching['slider1-1b']) else 0 for action_index in range(21)] # Slight Positive knee motion for planted leg.
            for joint1_value in range(21):
                for joint2_value in range(21):
                    for joint3_value in range(21):
                        self.action_mask[4][gait-1][joint1_value][joint2_value][joint3_value] = [1 if (action_index >= teaching['slider1-2a'] and action_index <= teaching['slider1-2b']) else 0 for action_index in range(21)] # Negative knee motion for swinging leg.
            

    def set_mask_gait_2(self, event): 
        
        gait = 2
        teaching = event.kwargs.get('teaching')

        # Flip bits to teach this gait strategy. (Left Hip, Left Knee, Right, Hip, Right Knee)
        if self.swinging_leg == 'left':
            self.action_mask[1][gait-1] = [1 if (action_index >= teaching['slider2-1a'] and action_index <= teaching['slider2-1b']) else 0 for action_index in range(21)] # Positive hip motion for planted leg.
            for joint1_value in range(21):
                self.action_mask[2][gait-1][joint1_value] = [1 if (action_index >= teaching['slider2-2a'] and action_index <= teaching['slider2-2b']) else 0 for action_index in range(21)] # Negative hip motion for swinging leg.
            for joint1_value in range(21):
                    for joint2_value in range(21):
                        self.action_mask[3][gait-1][joint1_value][joint2_value] = [1 if (action_index >= teaching['slider2-3a'] and action_index <= teaching['slider2-3b']) else 0 for action_index in range(21)] # Slight Positive knee motion for planted leg.
            for joint1_value in range(21):
                for joint2_value in range(21):
                    for joint3_value in range(21):
                        self.action_mask[4][gait-1][joint1_value][joint2_value][joint3_value] = [1 if (action_index >= teaching['slider2-4a'] and action_index <= teaching['slider2-4b']) else 0 for action_index in range(21)] # Negative knee motion for swinging leg.
            
        if self.swinging_leg == 'right':
            self.action_mask[1][gait-1] = [1 if (action_index >= teaching['slider2-3a'] and action_index <= teaching['slider2-3b']) else 0 for action_index in range(21)] # Positive hip motion for planted leg.
            for joint1_value in range(21):
                self.action_mask[2][gait-1][joint1_value] = [1 if (action_index >= teaching['slider2-4a'] and action_index <= teaching['slider2-4b']) else 0 for action_index in range(21)] # Negative hip motion for swinging leg.
            for joint1_value in range(21):
                    for joint2_value in range(21):
                        self.action_mask[3][gait-1][joint1_value][joint2_value] = [1 if (action_index >= teaching['slider2-1a'] and action_index <= teaching['slider2-1b']) else 0 for action_index in range(21)] # Slight Positive knee motion for planted leg.
            for joint1_value in range(21):
                for joint2_value in range(21):
                    for joint3_value in range(21):
                        self.action_mask[4][gait-1][joint1_value][joint2_value][joint3_value] = [1 if (action_index >= teaching['slider2-2a'] and action_index <= teaching['slider2-2b']) else 0 for action_index in range(21)] # Negative knee motion for swinging leg.
        pass

    def set_mask_gait_3(self, event): 
        
        gait = 3
        teaching = event.kwargs.get('teaching')

        # Flip bits for to teach this gait strategy.  (Left Hip, Left Knee, Right, Hip, Right Knee)
        if self.swinging_leg == 'left':
            self.action_mask[1][gait-1] = [1 if (action_index >= teaching['slider3-1a'] and action_index <= teaching['slider3-1b']) else 0 for action_index in range(21)] # Positive hip motion for planted leg.
            for joint1_value in range(21):
                self.action_mask[2][gait-1][joint1_value] = [1 if (action_index >= teaching['slider3-2a'] and action_index <= teaching['slider3-2b']) else 0 for action_index in range(21)] # Negative hip motion for swinging leg.
            for joint1_value in range(21):
                    for joint2_value in range(21):
                        self.action_mask[3][gait-1][joint1_value][joint2_value] = [1 if (action_index >= teaching['slider3-3a'] and action_index <= teaching['slider3-3b']) else 0 for action_index in range(21)] # Slight Positive knee motion for planted leg.
            for joint1_value in range(21):
                for joint2_value in range(21):
                    for joint3_value in range(21):
                        self.action_mask[4][gait-1][joint1_value][joint2_value][joint3_value] = [1 if (action_index >= teaching['slider3-4a'] and action_index <= teaching['slider3-4b']) else 0 for action_index in range(21)] # Negative knee motion for swinging leg.
        
        if self.swinging_leg == 'right':
            self.action_mask[1][gait-1] = [1 if (action_index >= teaching['slider3-3a'] and action_index <= teaching['slider3-3b']) else 0 for action_index in range(21)] # Positive hip motion for planted leg.
            for joint1_value in range(21):
                self.action_mask[2][gait-1][joint1_value] = [1 if (action_index >= teaching['slider3-4a'] and action_index <= teaching['slider3-4b']) else 0 for action_index in range(21)] # Negative hip motion for swinging leg.
            for joint1_value in range(21):
                    for joint2_value in range(21):
                        self.action_mask[3][gait-1][joint1_value][joint2_value] = [1 if (action_index >= teaching['slider3-1a'] and action_index <= teaching['slider3-1b']) else 0 for action_index in range(21)] # Slight Positive knee motion for planted leg.
            for joint1_value in range(21):
                for joint2_value in range(21):
                    for joint3_value in range(21):
                        self.action_mask[4][gait-1][joint1_value][joint2_value][joint3_value] = [1 if (action_index >= teaching['slider3-2a'] and action_index <= teaching['slider3-2b']) else 0 for action_index in range(21)] # Negative knee motion for swinging leg.
        pass
    """


    ##### Options Framework Working for everything except resettable simulator #####

    def set_mask_filter(self, event):

        if self.options_framework and self.num_timesteps > 0:
            
            # Apply the options framework.
            left_hip = left_knee = right_hip = right_knee = range(21)

            total_possible_actions = []
            possible_actions = []
            possible_states = []

            if event.kwargs.get('action') is not None: # and len(event.kwargs.get('action')) == 5:
                gait = event.kwargs.get('action')[0]
            else:
                gait = 0

            teaching = self.teaching # event.kwargs.get('teaching')

            # Flip bits to teach the strategy for this gait. (Left Hip, Left Knee, Right, Hip, Right Knee)
            if self.swinging_leg == 'left':

                ranges = {'left_hip': {'min': teaching['gait-1-swinging-hip-min'], 'max': teaching['gait-1-swinging-hip-max']},
                        'left_knee': {'min': teaching['gait-1-swinging-knee-min'], 'max': teaching['gait-1-swinging-knee-max']}, 
                        'right_hip': {'min': teaching['gait-1-planted-hip-min'], 'max': teaching['gait-1-planted-hip-max']}, 
                        'right_knee': {'min': teaching['gait-1-planted-knee-min'], 'max': teaching['gait-1-planted-knee-max']}
                    }

            if self.swinging_leg == 'right':

                ranges = {'left_hip': {'min': teaching['gait-1-planted-hip-min'], 'max': teaching['gait-1-planted-hip-max']},
                    'left_knee': {'min': teaching['gait-1-planted-knee-min'], 'max': teaching['gait-1-planted-knee-max']}, 
                    'right_hip': {'min': teaching['gait-1-swinging-hip-min'], 'max': teaching['gait-1-swinging-hip-max']}, 
                    'right_knee': {'min': teaching['gait-1-swinging-knee-min'], 'max': teaching['gait-1-swinging-knee-max']}
                }

            #if self.num_timesteps == 0:
            #    self.recorded_actions.append([0, 5, 11, 12, 0])
            """
            if np.all(event.kwargs.get('action')==0):
                self.recorded_actions.append([10, 10, 10, 10])
            else:
                self.recorded_actions.append(event.kwargs.get('action'))
            """
            
            if self.training_env is not None:
                sim_env = copy.deepcopy(self.training_env.envs[0]) 
            else: 
                sim_env = copy.deepcopy(env.envs[0])

            print('recorded actions: ', self.recorded_actions, round(sim_env.get_state()['state'][9], 4))
            for action in self.recorded_actions:
                replay_state, _, _, _ = sim_env.step(action)
                print('replay: ', replay_state[9], round(self.training_env.envs[0].get_state()['state'][9], 4))
                

            #print(round(self.training_env.envs[0].get_state()['state'][9], 4))
            
            """
            for x_index, x_value in enumerate(list(filter(lambda x: x >= ranges['left_hip']['min'] and x <= ranges['left_hip']['max'], left_hip))):
                for y_index, y_value in enumerate(list(filter(lambda x: x >= ranges['left_knee']['min'] and x <= ranges['left_knee']['max'], left_knee))):
                    for z_index, z_value in enumerate(list(filter(lambda x: x >= ranges['right_hip']['min'] and x <= ranges['right_hip']['max'], right_hip))):
                        #for j_index, j_value in enumerate(list(filter(lambda x: x >= 0 and x <= 20, right_knee))):
                        for j_index, j_value in enumerate(list(filter(lambda x: x >= ranges['right_knee']['min'] and x <= ranges['right_knee']['max'], right_knee))):
            """
            for left_hip_index in list(filter(lambda x: x >= ranges['left_hip']['min'] and x <= ranges['left_hip']['max'], left_hip)): # (range(ranges['left_hip']['min'], ranges['left_hip']['max'])):
                for left_knee_index in list(filter(lambda x: x >= ranges['left_knee']['min'] and x <= ranges['left_knee']['max'], left_knee)): # range(ranges['left_knee']['min'], ranges['left_knee']['max']+1):
                    for right_hip_index in list(filter(lambda x: x >= ranges['right_hip']['min'] and x <= ranges['right_hip']['max'], right_hip)): # range(ranges['right_hip']['min'], ranges['right_hip']['max']+1):
                        for right_knee_index in list(filter(lambda x: x >= ranges['right_knee']['min'] and x <= ranges['right_knee']['max'], right_knee)): # range(ranges['right_knee']['min'], ranges['right_knee']['max']+1):
                            #"""
                            test_copy = None
                            test_copy = copy.deepcopy(sim_env)
                            
                            """
                            for action in self.recorded_actions:
                                print('replay: ', round(test_copy.get_state()['state'][9], 4))
                                test_copy.step(action)
                            """
                            #print('deep copy env: ', test_copy.state[4])  # <----

                            #test_copy = copy.copy(self.training_env) #copy.deepcopy(self.training_env.envs[0])
                            #print(test_copy, round(test_copy.legs[1].position[0], 4))
                            total_possible_actions.append((left_hip_index, left_knee_index, right_hip_index, right_knee_index))
                            #snapshot = self.snapshot
                            if self.num_timesteps < 1:
                                #possible_state = test_copy.simulate(10, 10, 10, 10)
                                possible_state, _, _, _ = test_copy.step([0, 10, 10, 10, 10])
                                print('Options Framework: ', [10, 10, 10, 10], event.kwargs.get('action'), possible_state[9], round(event.kwargs.get('right_hip_angle'), 4)) #test_copy.state[4], test_copy.state[8], test_copy.state[13], self.training_env.env_method('get_state')[0]['state'][8], self.training_env.env_method('get_state')[0]['state'][13])

                            else: 
                                #possible_state = test_copy.simulate(left_hip_index, left_knee_index, right_hip_index, right_knee_index)
                                possible_state, _, _, _ = test_copy.step([gait, left_hip_index, left_knee_index, right_hip_index, right_knee_index])
                                print('Options Framework: ', [left_hip_index, left_knee_index, right_hip_index, right_knee_index], event.kwargs.get('action'), possible_state[9], round(event.kwargs.get('right_hip_angle'), 4)) #test_copy.state[4], test_copy.state[8], test_copy.state[13], self.training_env.env_method('get_state')[0]['state'][8], self.training_env.env_method('get_state')[0]['state'][13])
                            #"""
                            #possible_states.append(possible_state[0])
                            #if not possible_state:
                                #possible_actions.append((x_value, y_value, z_value, j_value))
                #test_copy.render()
            #print('possible states: ', len(possible_actions), possible_actions, possible_states)
            
            # Grab all possible actions by joint. 
            left_hip_actions = [item[0] for item in possible_actions]
            left_knee_actions = [item[1] for item in possible_actions]
            right_hip_actions = [item[2] for item in possible_actions]
            right_knee_actions = [item[3] for item in possible_actions]

        """ 
            if action_list[step-1] != self.action:
                raise Exception("Invalid action was selected: ", action_list[step], self.action)
        """

        self.gait.save_point_env = None
        self.gait.simulated_env = None
        self.gait.training_env = None
        self.gait.snapshot = None
        self.gait.recorded_actions = []
        self.gait.recorded_states = []

        self.gait.options_framework = False

        self.gait.recorded_actions.append(self.training_env.env_method('get_state')[0]['action'])
        """
        #print(sim[0]['state'][8], sim[0]['state'][13], sim[0]['action'])
        """
        def simulate(self, a, b, c, d):

        masked_action = [a, b, c, d]
        print('in simulate: ', self.joints[2].angle)
        
        left_hip = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        left_knee = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        right_hip = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        right_knee = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        
        action = [left_hip[masked_action[0]], left_knee[masked_action[1]], right_hip[masked_action[2]], right_knee[masked_action[3]]]
        #action = [left_hip[masked_action[1]], left_knee[masked_action[2]], right_hip[masked_action[3]], right_knee[masked_action[4]]]
        """
        if self.counter == 0 or len(masked_action) == 4:
        #print(masked_action)
        #if np.all(masked_action==0):
            action = [left_hip[masked_action[0]], left_knee[masked_action[1]], right_hip[masked_action[2]], right_knee[masked_action[3]]]
        else:
            action = [left_hip[masked_action[1]], left_knee[masked_action[2]], right_hip[masked_action[3]], right_knee[masked_action[4]]]
        """
        # Modify the joints with the simulated action. 
        control_speed = False  # Should be easier as well
        if control_speed:
            self.joints[0].motorSpeed = float(SPEED_HIP  * np.clip(action[0], -1, 1))
            self.joints[1].motorSpeed = float(SPEED_KNEE * np.clip(action[1], -1, 1))
            self.joints[2].motorSpeed = float(SPEED_HIP  * np.clip(action[2], -1, 1))
            self.joints[3].motorSpeed = float(SPEED_KNEE * np.clip(action[3], -1, 1))
        else:
            self.joints[0].motorSpeed     = float(SPEED_HIP     * np.sign(action[0]))
            self.joints[0].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action[0]), 0, 1))
            self.joints[1].motorSpeed     = float(SPEED_KNEE    * np.sign(action[1]))
            self.joints[1].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action[1]), 0, 1))
            self.joints[2].motorSpeed     = float(SPEED_HIP     * np.sign(action[2]))
            self.joints[2].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action[2]), 0, 1))
            self.joints[3].motorSpeed     = float(SPEED_KNEE    * np.sign(action[3]))
            self.joints[3].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action[3]), 0, 1))


        # Step the world.         
        self.world.Step(1.0/FPS, 6*30, 2*30)

        state = {
            'action': action,
            'hull': self.hull.angle,        # Normal angles up to 0.5 here, but sure more is possible.
            'left_hip_angle': self.joints[0].angle,   # This will give 1.1 on high up, but it's still OK (and there should be spikes on hiting the ground, that's normal too)
            'left_knee_angle': self.joints[1].angle + 1.0,
            'left_leg_contact': self.legs[1].ground_contact,
            'right_hip_angle': self.joints[2].angle,
            'right_knee_angle': self.joints[3].angle + 1.0,
            'right_leg_contact': self.legs[3].ground_contact
        }
        #print(state)

        return [not state['left_leg_contact'] and not state['right_leg_contact'], action, state['left_leg_contact'], state['right_leg_contact'], round(state['right_hip_angle'], 4)] #state
    
    # Old crouch and spring

    # Start Crouch
            # Goal: Hull tilts down, Rules: Both legs touching the ground, Action: Flex hip
            if self.state_machine.state == 'start_crouch':

                start_crouch_done = False

                if state[0] > 0.5 : #  < state[0] -0.1 or 
                    start_crouch_done = done = True
                    reason_code = 'hull angle too steep'
                    print('terminal condition: hull angle too steep ', state[0], reward)
                
                if state[8] == 0 and state[13] == 0: 
                    
                    if self.state_machine.swinging_leg == 'left':
                        start_crouch_done = done = True
                        reason_code = 'neither leg in contact with the ground'
                        print('terminal condition: neither leg in contact with the ground ', state[8], state[13], reward)
                    if self.state_machine.swinging_leg == 'right':
                        start_crouch_done = done = True
                        reason_code = 'neither leg in contact with the ground'
                        print('terminal condition: neither leg in contact with the ground ', state[8], state[13], reward, self.action)

                # Reward, 10 points at hull angle = -0.05.
                if start_crouch_done or state[0] < -0.05:
                    reward += 10*math.exp(-20*abs(state[0]+0.05)) - 9
                    print('crouch terminal penalty: ', reward)

                # Reward, 10 points at 0 velocity or above
                if spring_forward_done or (state[2] > 0 and masked_action[0] == 1):
                    reward += 10*math.exp(10*(state[2] - abs(state[2]))) - 9
                    print('spring forward terminal penalty: ', reward)

                gait_kpi = state[0]

            # Start Spring Forward
            # Goal: Maximize forward motion, Rules: Planted leg touching the ground, Action: Extend planted leg (extend hip, flex knee)
            if self.state_machine.state == 'start_spring_forward':

                spring_forward_done = False

                init_x = TERRAIN_STEP*TERRAIN_STARTPAD/2 # starting x position
                #reward += pos[0] - init_x # This should be a measure of x distance traveled during this phase, normalized to 0 to 1
                self.state_machine.start_spring_forward_reward += pos[0] - init_x

                if state[8] == 0 and state[13] == 0: 
                    if self.state_machine.swinging_leg == 'left':
                        spring_forward_done = done = True
                        reason_code = 'neither leg in contact with the ground'
                        print('terminal condition: neither leg in contact with the ground ', state[8], state[13], reward)
                    if self.state_machine.swinging_leg == 'right':
                        spring_forward_done = done = True
                        reason_code = 'neither leg in contact with the ground'
                        print('terminal condition: neither leg in contact with the ground ', state[8], state[13], reward, self.action)
                """
                if self.state_machine.num_timesteps > 2 and pos[0] <= self.position_history[4]:  # No forward progress 
                    if self.state_machine.num_timesteps <= 1 and not self.state_machine.step_flag:  # don't enforce terminal when swinging leg plants. 
                        spring_forward_done = done = True
                        reason_code = 'no forward progress'
                        print('terminal condition: no forward progress ', pos[0] < self.position_history[0], reward)
                
                if abs(state[4] - state[9]) < abs(self.state_history[0][4] - self.state_history[0][9]):  # Legs are not moving apart
                    done = True
                    reason_code = 'legs not moving apart'
                    print('terminal condition: legs not moving apart', self.state_machine.state, abs(state[4] - state[9]), abs(self.state_history[0][4] - self.state_history[0][9]), reward)
                """
                #print('velocity: ', state[2])

                # Reward, 10 points at 0 velocity or above
                if spring_forward_done or (state[2] > 0 and masked_action[0] == 1):
                    reward += 10*math.exp(10*(state[2] - abs(state[2]))) - 9
                    print('spring forward terminal penalty: ', reward)

                gait_kpi = state[2]

    # Old code for legs moving toward each other during switch leg

    # Reward, 10 points for swinging leg in front of planted leg
    if self.state_machine.swinging_leg == 'left' and masked_action[0] == 0:
        reward += 10*math.exp(10*(self.legs[2].position[0] - self.legs[0].position[0]) - abs(self.legs[2].position[0] - self.legs[0].position[0])) 
        print('plant leg reward: ', reward)
        pass
    if self.state_machine.swinging_leg == 'right' and masked_action[0] == 0:
        reward += 10*math.exp(10*(self.legs[0].position[0] - self.legs[2].position[0]) - abs(self.legs[0].position[0] - self.legs[2].position[0]))
        print('plant leg reward: ', reward)
        pass

    if self.state_machine.swinging_leg == 'left':
        gait_kpi = self.legs[2].position[0] - self.legs[0].position[0]
    if self.state_machine.swinging_leg == 'right':
        gait_kpi = self.legs[0].position[0] - self.legs[2].position[0]

