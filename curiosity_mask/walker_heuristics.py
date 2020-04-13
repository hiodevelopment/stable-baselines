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
