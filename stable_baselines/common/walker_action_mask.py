import sys
import math
import copy
import csv
import time

import numpy as np
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym.spaces import Discrete, MultiDiscrete
from gym.utils import colorize, seeding, EzPickle

from collections import deque
from curiosity_mask.util import create_dummy_action_mask as mask

# This is simple 4-joints walker robot environment.
#
# There are two versions:
#
# - Normal, with slightly uneven terrain.
#
# - Hardcore with ladders, stumps, pitfalls.
#
# Reward is given for moving forward, total 300+ points up to the far end. If the robot falls,
# it gets -100. Applying motor torque costs a small amount of points, more optimal agent
# will get better score.
#
# Heuristic is provided for testing, it's also useful to get demonstrations to
# learn from. To run heuristic:
#
# python gym/envs/box2d/bipedal_walker.py
#
# State consists of hull angle speed, angular velocity, horizontal speed, vertical speed,
# position of joints and joints angular speed, legs contact with ground, and 10 lidar
# rangefinder measurements to help to deal with the hardcore version. There's no coordinates
# in the state vector. Lidar is less useful in normal version, but it works.
#
# To solve the game you need to get 300 points in 1600 time steps.
#
# To solve hardcore version you need 300 points in 2000 time steps.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

# Action 1: Left Hip
# Action 2: Left Knee
# Action 3: Right Hip
# Action 4: Right Knee

FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

MOTORS_TORQUE = 80
SPEED_HIP     = 4
SPEED_KNEE    = 6
LIDAR_RANGE   = 160/SCALE

INITIAL_RANDOM = 5

HULL_POLY =[
    (-30,+9), (+6,+9), (+34,+1),
    (+34,-8), (-30,-8)
    ]
LEG_DOWN = -8/SCALE
LEG_W, LEG_H = 8/SCALE, 34/SCALE

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP   = 14/SCALE
TERRAIN_LENGTH = 200     # in steps
TERRAIN_HEIGHT = VIEWPORT_H/SCALE/4
TERRAIN_GRASS    = 10    # low long are grass spots, in steps
TERRAIN_STARTPAD = 20    # in steps
FRICTION = 2.5

HULL_FD = fixtureDef(
                shape=polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in HULL_POLY ]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0020,
                maskBits=0x001,  # collide only with ground
                restitution=0.0) # 0.99 bouncy

LEG_FD = fixtureDef(
                    shape=polygonShape(box=(LEG_W/2, LEG_H/2)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)

LOWER_FD = fixtureDef(
                    shape=polygonShape(box=(0.8*LEG_W/2, LEG_H/2)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        if self.env.hull==contact.fixtureA.body or self.env.hull==contact.fixtureB.body:
            self.env.game_over = True
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = True
        
        # Detect Forces on Contact
        contact_force = 0
        if self.env.state_machine is not None:
            if self.env.state_machine.state == 'plant_leg' and (self.env.legs[1] in [contact.fixtureA.body, contact.fixtureB.body] or self.env.legs[3] in [contact.fixtureA.body, contact.fixtureB.body]):
                for i, leg in enumerate([self.env.legs[1], self.env.legs[3]]):
                    for c in leg.contacts:
                        if c.contact.manifold.pointCount > 0:
                            #print('contact force: ', i, c.contact.manifold.points[0].normalImpulse/(1.0/FPS))
                            if c.contact.manifold.points[0].normalImpulse/(1.0/FPS) > contact_force:
                                contact_force = c.contact.manifold.points[0].normalImpulse/(1.0/FPS)
                                self.env.leg_force = contact_force  
                                self.env.state_machine.step_flag = True

    def EndContact(self, contact):
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = False


class BipedalWalker(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    hardcore = False

    def __init__(self):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World()
        self.terrain = None
        self.hull = None
        self.leg_force = 0

        
        #ts = str(int(time.time()))
        #print(ts)
        #with open('walker_teaching_log_' + ts + '.csv', 'w', newline='') as file:
        with open('walker_teaching_log.csv', 'w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['iteration', 'episode', 'gait_iteration', 'gait_phase', 'gait_action', 'left_hip', 'left_knee', 'right_hip', 'right_knee', 'terminal', 'gait_kpi', 'reward', 'reason_code'])

        self.prev_shaping = None

        self.fd_polygon = fixtureDef(
                        shape = polygonShape(vertices=
                        [(0, 0),
                         (1, 0),
                         (1, -1),
                         (0, -1)]),
                        friction = FRICTION)

        self.fd_edge = fixtureDef(
                    shape = edgeShape(vertices=
                    [(0, 0),
                     (1, 1)]),
                    friction = FRICTION,
                    categoryBits=0x0001,
                )

        

        high = np.array([np.inf] * 28)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        self.action_space = MultiDiscrete([3, 21, 21, 21, 21])

        self.valid_actions = []
        self.state_machine = None
        self.reset()
        self.terminal = False
        self.counter = 0
        self.episodes = 0
    
        #print('env init', len(self.valid_actions[0]), len(self.valid_actions[1]), len(self.valid_actions[2]), len(self.valid_actions[3]))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_state(self):
        return {'state': self.state, 'action': self.action, 'joints': self.joints, 'hull': self.hull, 'legs': self.legs, 'world': self.world}

    def set_infos(self, infos):
        self.valid_actions = infos

    def set_state_machine(self, machine):
        self.state_machine = machine

    def set_terminal(self, terminal):
        self.terminal = terminal

    def _destroy(self):
        if not self.terrain: return
        self.world.contactListener = None
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []
        self.world.DestroyBody(self.hull)
        self.hull = None
        for leg in self.legs:
            self.world.DestroyBody(leg)
        self.legs = []
        self.joints = []

    def _generate_terrain(self, hardcore):
        GRASS, STUMP, STAIRS, PIT, _STATES_ = range(5)
        state    = GRASS
        velocity = 0.0
        y        = TERRAIN_HEIGHT
        counter  = TERRAIN_STARTPAD
        oneshot  = False
        self.terrain   = []
        self.terrain_x = []
        self.terrain_y = []
        for i in range(TERRAIN_LENGTH):
            x = i*TERRAIN_STEP
            self.terrain_x.append(x)

            if state==GRASS and not oneshot:
                velocity = 0.8*velocity + 0.01*np.sign(TERRAIN_HEIGHT - y)
                if i > TERRAIN_STARTPAD: velocity += self.np_random.uniform(-1, 1)/SCALE   #1
                y += velocity

            elif state==PIT and oneshot:
                counter = self.np_random.randint(3, 5)
                poly = [
                    (x,              y),
                    (x+TERRAIN_STEP, y),
                    (x+TERRAIN_STEP, y-4*TERRAIN_STEP),
                    (x,              y-4*TERRAIN_STEP),
                    ]
                self.fd_polygon.shape.vertices=poly
                t = self.world.CreateStaticBody(
                    fixtures = self.fd_polygon)
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)

                self.fd_polygon.shape.vertices=[(p[0]+TERRAIN_STEP*counter,p[1]) for p in poly]
                t = self.world.CreateStaticBody(
                    fixtures = self.fd_polygon)
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)
                counter += 2
                original_y = y

            elif state==PIT and not oneshot:
                y = original_y
                if counter > 1:
                    y -= 4*TERRAIN_STEP

            elif state==STUMP and oneshot:
                counter = self.np_random.randint(1, 3)
                poly = [
                    (x,                      y),
                    (x+counter*TERRAIN_STEP, y),
                    (x+counter*TERRAIN_STEP, y+counter*TERRAIN_STEP),
                    (x,                      y+counter*TERRAIN_STEP),
                    ]
                self.fd_polygon.shape.vertices=poly
                t = self.world.CreateStaticBody(
                    fixtures = self.fd_polygon)
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)

            elif state==STAIRS and oneshot:
                stair_height = +1 if self.np_random.rand() > 0.5 else -1
                stair_width = self.np_random.randint(4, 5)
                stair_steps = self.np_random.randint(3, 5)
                original_y = y
                for s in range(stair_steps):
                    poly = [
                        (x+(    s*stair_width)*TERRAIN_STEP, y+(   s*stair_height)*TERRAIN_STEP),
                        (x+((1+s)*stair_width)*TERRAIN_STEP, y+(   s*stair_height)*TERRAIN_STEP),
                        (x+((1+s)*stair_width)*TERRAIN_STEP, y+(-1+s*stair_height)*TERRAIN_STEP),
                        (x+(    s*stair_width)*TERRAIN_STEP, y+(-1+s*stair_height)*TERRAIN_STEP),
                        ]
                    self.fd_polygon.shape.vertices=poly
                    t = self.world.CreateStaticBody(
                        fixtures = self.fd_polygon)
                    t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                    self.terrain.append(t)
                counter = stair_steps*stair_width

            elif state==STAIRS and not oneshot:
                s = stair_steps*stair_width - counter - stair_height
                n = s/stair_width
                y = original_y + (n*stair_height)*TERRAIN_STEP

            oneshot = False
            self.terrain_y.append(y)
            counter -= 1
            if counter==0:
                counter = self.np_random.randint(TERRAIN_GRASS/2, TERRAIN_GRASS)
                if state==GRASS and hardcore:
                    state = self.np_random.randint(1, _STATES_)
                    oneshot = True
                else:
                    state = GRASS
                    oneshot = True

        self.terrain_poly = []
        for i in range(TERRAIN_LENGTH-1):
            poly = [
                (self.terrain_x[i],   self.terrain_y[i]),
                (self.terrain_x[i+1], self.terrain_y[i+1])
                ]
            self.fd_edge.shape.vertices=poly
            t = self.world.CreateStaticBody(
                fixtures = self.fd_edge)
            color = (0.3, 1.0 if i%2==0 else 0.8, 0.3)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (0.4, 0.6, 0.3)
            poly += [ (poly[1][0], 0), (poly[0][0], 0) ]
            self.terrain_poly.append( (poly, color) )
        self.terrain.reverse()

    def _generate_clouds(self):
        # Sorry for the clouds, couldn't resist
        self.cloud_poly   = []
        for i in range(TERRAIN_LENGTH//20):
            x = self.np_random.uniform(0, TERRAIN_LENGTH)*TERRAIN_STEP
            y = VIEWPORT_H/SCALE*3/4
            poly = [
                (x+15*TERRAIN_STEP*math.sin(3.14*2*a/5)+self.np_random.uniform(0,5*TERRAIN_STEP),
                 y+ 5*TERRAIN_STEP*math.cos(3.14*2*a/5)+self.np_random.uniform(0,5*TERRAIN_STEP) )
                for a in range(5) ]
            x1 = min( [p[0] for p in poly] )
            x2 = max( [p[0] for p in poly] )
            self.cloud_poly.append( (poly,x1,x2) )

    def reset(self):
        self.valid_actions = mask(self.action_space)
        self.valid_actions[0] = [1, 0, 0]
        self.terminal = False
        self._destroy()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.game_over = False
        self.prev_shaping = None
        self.scroll = 0.0
        self.lidar_render = 0

        W = VIEWPORT_W/SCALE
        H = VIEWPORT_H/SCALE

        self._generate_terrain(self.hardcore)
        self._generate_clouds()

        init_x = TERRAIN_STEP*TERRAIN_STARTPAD/2
        init_y = TERRAIN_HEIGHT+2*LEG_H
        self.hull = self.world.CreateDynamicBody(
            position = (init_x, init_y),
            fixtures = HULL_FD
                )
        self.hull.color1 = (0.5,0.4,0.9)
        self.hull.color2 = (0.3,0.3,0.5)
        self.hull.ApplyForceToCenter((self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), 0), True)

        self.legs = []
        self.joints = []
        for i in [-1,+1]: # Original position is right leg slightly forward, left leg slightly back. 
            leg = self.world.CreateDynamicBody(
                position = (init_x, init_y - LEG_H/2 - LEG_DOWN),
                angle = (i*0.15),  # *0.05
                fixtures = LEG_FD
                )
            leg.color1 = (0.6-i/10., 0.3-i/10., 0.5-i/10.)
            leg.color2 = (0.4-i/10., 0.2-i/10., 0.3-i/10.)
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=leg,
                localAnchorA=(0, LEG_DOWN),
                localAnchorB=(0, LEG_H/2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed = i,
                lowerAngle = -0.8,
                upperAngle = 1.1,
                )
            self.legs.append(leg)
            self.joints.append(self.world.CreateJoint(rjd))

            lower = self.world.CreateDynamicBody(
                position = (init_x, init_y - LEG_H*3/2 - LEG_DOWN),
                angle = (i*0.15),  # *0.05
                fixtures = LOWER_FD
                )
            lower.color1 = (0.6-i/10., 0.3-i/10., 0.5-i/10.)
            lower.color2 = (0.4-i/10., 0.2-i/10., 0.3-i/10.)
            rjd = revoluteJointDef(
                bodyA=leg,
                bodyB=lower,
                localAnchorA=(0, -LEG_H/2),
                localAnchorB=(0, LEG_H/2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed = 1,
                lowerAngle = -1.6,
                upperAngle = -0.1,
                )
            lower.ground_contact = False
            self.legs.append(lower)
            self.joints.append(self.world.CreateJoint(rjd))

        self.drawlist = self.terrain + self.legs + [self.hull]

        self.counter = 0

        class LidarCallback(Box2D.b2.rayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0:
                    return -1
                self.p2 = point
                self.fraction = fraction
                return fraction
        self.lidar = [LidarCallback() for _ in range(10)]

        self.action = []
        self.state = {}
        empty_observation = [init_x,init_x,init_x,init_x,init_x]
        self.position_history = deque(empty_observation)
        self.state_history = deque([None, None, None, None, None])
        self.leg_history = deque([(self.legs[0].position[0], self.legs[2].position[0]), (self.legs[0].position[0], self.legs[2].position[0]), (self.legs[0].position[0], self.legs[2].position[0]), (self.legs[0].position[0], self.legs[2].position[0]), (self.legs[0].position[0], self.legs[2].position[0])])
        if self.state_machine is not None and self.state_machine.state != 'start':
            self.state_machine.reset() # reset the state of the state machine. 
            self.state_machine.num_timesteps = 0 # reset the iteration counter. 
            self.state_machine.swinging_leg = 'right'
            self.state_machine.start = True
            self.state_machine.action_mask[0] = [1, 0, 0]
            #print('reset ', self.state_machine.num_timesteps)
        elif self.state_machine is not None:
            self.state_machine.num_timesteps = 0 # reset the iteration counter.
            self.state_machine.swinging_leg = 'right'
            #print('reset ', self.state_machine.num_timesteps)
        return self.step(np.array([0,10,10,10,10]))[0]

    def step(self, masked_action):
        #self.hull.ApplyForceToCenter((0, 20), True) # Uncomment this to receive a bit of stability help
        #print(self.world.joints[0].GetReactionForce(1.0/FPS), self.world.joints[1].GetReactionForce(1.0/FPS))
        #print(self.hull.mass, self.hull.massData, self.hull.inertia, self.legs[0].mass, self.legs[1].mass)
        #print(self.world.GetProfile)
        """
        for i, leg in enumerate(self.legs):
            if self.legs[i].contacts:
                for i, value in enumerate(self.legs[0].contacts):
                    print(self.legs[0].contacts[i].contact.manifold.points)
        """
        #sys.exit()

        #print('in step', self.valid_actions[1][0])
        #print('in step, action:', masked_action)

        left_hip = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        left_knee = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        right_hip = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        right_knee = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        
        action = [left_hip[masked_action[1]], left_knee[masked_action[2]], right_hip[masked_action[3]], right_knee[masked_action[4]]]
        
        if np.all(masked_action==0):  # The reset action of the env [0,0,0,0] should not be translated to [-1,-1,-1,-1].
            self.action = masked_action
        else:
            self.action = masked_action

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

        self.world.Step(1.0/FPS, 6*30, 2*30)

        pos = self.hull.position
        vel = self.hull.linearVelocity

        for i in range(10):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(1.5*i/10.0)*LIDAR_RANGE,
                pos[1] - math.cos(1.5*i/10.0)*LIDAR_RANGE)
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

        state = [
            self.hull.angle,        # Normal angles up to 0.5 here, but sure more is possible.
            2.0*self.hull.angularVelocity/FPS,
            0.3*vel.x*(VIEWPORT_W/SCALE)/FPS,  # Normalized to get -1..1 range
            0.3*vel.y*(VIEWPORT_H/SCALE)/FPS,
            self.joints[0].angle,   # This will give 1.1 on high up, but it's still OK (and there should be spikes on hiting the ground, that's normal too)
            self.joints[0].speed / SPEED_HIP,
            self.joints[1].angle + 1.0,
            self.joints[1].speed / SPEED_KNEE,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.joints[2].angle,
            self.joints[2].speed / SPEED_HIP,
            self.joints[3].angle + 1.0,
            self.joints[3].speed / SPEED_KNEE,
            1.0 if self.legs[3].ground_contact else 0.0
            ]
        #print(self.legs[0].joints[0].joint.anchorB[0], self.joints[0].anchorB[0])
        #print('knee angles: ', state[6], state[11])

        state += [l.fraction for l in self.lidar]
        state += [self.joints[0].anchorB[1], self.joints[2].anchorB[1], self.joints[1].anchorB[1], self.joints[3].anchorB[1]] # left hip, right hip, left knee height, right knee height
        assert len(state)==28

        self.scroll = pos.x - VIEWPORT_W/SCALE/5

        done = False
        gait_kpi = None
        reason_code = 'none'
        shaping = 130*pos[0]/SCALE   # moving forward is a way to receive reward (normalized to get 300 on completion)
        #shaping -= 5.0*abs(state[0])  # keep head straight, other than that and falling, any behavior is unpunished

        """
        for a in action:
            reward -= 0.00035 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
            # normalized to about -50.0 using heuristic, more optimal agent should spend less
        """

        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        #### Environment Rules: Don't fall backwards or fall on your head

        if self.game_over or pos[0] < 0: # or state[6] < 0 or state[11] < 0: # Hyperextension of the knee is injury
            reward = -100
            done   = True
            reason_code = 'fell backward'
        if pos[0] > (TERRAIN_LENGTH-TERRAIN_GRASS)*TERRAIN_STEP:
            done   = True

        #if state[6] < 0 or state[11] < 0:
        if state[6] < 0:
            print('terminal condition: hyperextended planted knee ', state[6], state[11], reward)

        #### Teaching Strategies: Gait Phase has unique goals and rules expressed as rewards and terminals. #####
        if self.state_machine is not None and not done:

            if self.state_machine.num_timesteps > 100: # Stuck in one phase for more than 50 timesteps.
                done = True
                reason_code = 'stuck'
                reward = -100

            # Start 
            # Goal 1: Hull tilts down, Rules: Both legs touching the ground, Action: Flex hip
            # Goal 2: Maximize forward motion, Rules: Planted leg touching the ground, Action: Extend planted leg (extend hip, flex knee)

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

            # Plant Leg
            # Goal: lean forward and take a step that supports weight, Rules: planted leg touching the ground, swinging leg swings (moves forward)
            if self.state_machine.state == 'plant_leg':

                plant_leg_done = False

                # Add a stepwise reward for leaning forward (hull center of mass is in front of the planted leg), self.hull.massData.center[0], velocity?
                if self.state_machine.step_flag:  # only give out the position reward on taking a step.

                    # Reward
                    reward += 10*math.exp(-0.07*abs(self.leg_force - (self.hull.mass + self.legs[0].mass + self.legs[1].mass)*9.81)) - 9
                    #print('plant leg reward: ', reward)
                    
                    if self.state_machine.swinging_leg == 'left':
                        if self.leg_force >= (self.hull.mass + self.legs[0].mass + self.legs[1].mass)*9.81: # step can support weight of vaulting over planted leg. 
                            reason_code = 'step taken'
                            print('step taken: ', (self.hull.mass + self.legs[0].mass + self.legs[1].mass)*9.81, self.leg_force)
                        else: 
                            plant_leg_done = done = True  # terminate on first step if it can't support the weight of the robot.
                            reason_code = 'step cannot support robot weight'
                            print('terminal condition: step cannot support robot weight ', (self.hull.mass + self.legs[0].mass + self.legs[1].mass)*9.81, self.leg_force, reward)
                        
                    if self.state_machine.swinging_leg == 'right':
                        if self.leg_force >= (self.hull.mass + self.legs[0].mass + self.legs[1].mass)*9.81: # step can support weight of vaulting over planted leg. 
                            reason_code = 'step taken'
                            print('step taken: ', (self.hull.mass + self.legs[0].mass + self.legs[1].mass)*9.81, self.leg_force)
                        else: 
                            plant_leg_done = done = True  # terminate on first step if it can't support the weight of the robot.
                            reason_code = 'step cannot support robot weight'
                            print('terminal condition: step cannot support robot weight ', (self.hull.mass + self.legs[0].mass + self.legs[1].mass)*9.81, self.leg_force, reward)
                        

                if state[8] == 0 and state[13] == 1: 
                    if self.state_machine.swinging_leg == 'left': 
                        #done = True
                        reason_code = 'neither leg in contact with the ground'
                        print('terminal condition: neither leg in contact with the ground ', state[8], state[13], reward)
                    if self.state_machine.swinging_leg == 'right':
                        #done = True
                        reason_code = 'neither leg in contact with the ground'
                        print('terminal condition: neither leg in contact with the ground ', state[8], state[13], reward, self.action)

                #self.state_machine.step_flag = False

                gait_kpi = self.leg_force - (self.hull.mass + self.legs[0].mass + self.legs[1].mass)*9.81
                
            if self.state_machine.state == 'switch_legs':

                switch_leg_done = False

                if self.state_machine.num_timesteps <= 1 and not self.state_machine.step_flag and not (self.state_machine.state == 'switch_leg' and self.state_machine.num_timesteps < 3):  
                    if self.state_machine.swinging_leg == 'left':
                        if self.legs[0].position[0] < self.leg_history[0][0]:
                            switch_leg_done = done = True
                            reason_code = 'swinging leg not moving forward'
                            print('terminal condition: swinging leg not moving forward ', self.leg_history[0][0], self.legs[0].position[0], reward)
                
                if self.state_machine.swinging_leg == 'right':
                    if self.legs[2].position[0] < self.leg_history[0][1]:
                        switch_leg_done = done = True
                        reason_code = 'swinging leg not moving forward'
                        print('terminal condition: swinging leg not moving forward ', self.leg_history[0][1], self.legs[2].position[0], reward)
                
                # Are the legs moving toward each other. 
                if abs(state[4] - state[9]) > abs(self.state_history[0][4] - self.state_history[0][9]):  # Legs are not moving toward each other
                    switch_leg_done = done = True
                    reason_code = 'legs not moving toward each other'
                    print('terminal condition: legs not moving toward each other', self.state_machine.state, abs(state[4] - state[9]), abs(self.state_history[0][4] - self.state_history[0][9]), reward)

                # Reward, 10 points for swinging leg in front of planted leg
                if self.state_machine.swinging_leg == 'left' and masked_action[0] == 0:
                    reward += 10*math.exp(10*(self.legs[2].position[0] - self.legs[0].position[0]) - abs(self.legs[2].position[0] - self.legs[0].position[0])) - 9
                    print('plant leg reward: ', reward)
                    pass
                if self.state_machine.swinging_leg == 'right' and masked_action[0] == 0:
                    reward += 10*math.exp(10*(self.legs[0].position[0] - self.legs[2].position[0]) - abs(self.legs[0].position[0] - self.legs[2].position[0])) - 9
                    print('plant leg reward: ', reward)
                    pass

                if self.state_machine.swinging_leg == 'left':
                    gait_kpi = self.legs[2].position[0] - self.legs[0].position[0]
                if self.state_machine.swinging_leg == 'right':
                    gait_kpi = self.legs[0].position[0] - self.legs[2].position[0]

                if state[0] > 0.5 : #  < state[0] -0.1 or 
                    done = True
                    reason_code = 'hull angle too steep'
                    print('terminal condition: hull angle too steep ', state[0], reward)

            if self.state_machine.state == 'lift_leg':

                lift_leg_done = False

                if abs(state[4] - state[9]) < abs(self.state_history[0][4] - self.state_history[0][9]):  # Legs are not moving apart
                        lift_leg_done = done = True
                        reason_code = 'legs not moving apart'
                        print('terminal condition: legs not moving apart', self.state_machine.state, abs(state[4] - state[9]), abs(self.state_history[0][4] - self.state_history[0][9]), reward)
                
                if state[0] > 0.5:
                    lift_leg_done = done = True
                    reason_code = 'hull angle too steep'
                    print('terminal condition: hull angle too steep ', state[0], reward)
                
                if state[8] == 0 and state[13] == 0: 
                    if self.state_machine.swinging_leg == 'left':
                        lift_leg_done = done = True
                        reason_code = 'neither leg in contact with the ground'
                        print('terminal condition: neither leg in contact with the ground ', state[8], state[13], reward)
                    if self.state_machine.swinging_leg == 'right':
                        lift_leg_done = done = True
                        reason_code = 'neither leg in contact with the ground'
                        print('terminal condition: neither leg in contact with the ground ', state[8], state[13], reward, self.action)

                # Reward
                if lift_leg_done or state[0] < -0.05:
                    reward += 10*math.exp(-20*abs(state[0]+0.05)) - 9
                    print('lift leg terminal penalty: ', reward)

                gait_kpi = state[0]

        self.state = state
        self.leg_history.appendleft((self.legs[0].position[0], self.legs[2].position[0]))
        self.leg_history.pop()

        self.position_history.appendleft(pos[0])
        self.position_history.pop()

        self.state_history.appendleft(state)
        self.state_history.pop()
        #self.render()
        self.counter += 1
        #print('set action mask in env')

        if done:
            self.episodes += 1
            print('episode: ', self.episodes)

        #"""
        # Write CSV iteration here. 
        if self.state_machine is not None:
            with open('walker_teaching_log.csv', 'a', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow([self.counter, self.episodes, self.state_machine.num_timesteps, self.state_machine.state, masked_action[0], masked_action[1], masked_action[2], masked_action[3], masked_action[4], 'gait phase complete' if done else 'running', gait_kpi, reward, reason_code])
        #"""

        return np.array(state), reward, done, {'action_mask': self.valid_actions}
        #return np.array(state), reward, done, {}

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
        self.viewer.set_bounds(self.scroll, VIEWPORT_W/SCALE + self.scroll, 0, VIEWPORT_H/SCALE)

        self.viewer.draw_polygon( [
            (self.scroll,                  0),
            (self.scroll+VIEWPORT_W/SCALE, 0),
            (self.scroll+VIEWPORT_W/SCALE, VIEWPORT_H/SCALE),
            (self.scroll,                  VIEWPORT_H/SCALE),
            ], color=(0.9, 0.9, 1.0) )
        for poly,x1,x2 in self.cloud_poly:
            if x2 < self.scroll/2: continue
            if x1 > self.scroll/2 + VIEWPORT_W/SCALE: continue
            self.viewer.draw_polygon( [(p[0]+self.scroll/2, p[1]) for p in poly], color=(1,1,1))
        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll: continue
            if poly[0][0] > self.scroll + VIEWPORT_W/SCALE: continue
            self.viewer.draw_polygon(poly, color=color)

        self.lidar_render = (self.lidar_render+1) % 100
        i = self.lidar_render
        if i < 2*len(self.lidar):
            l = self.lidar[i] if i < len(self.lidar) else self.lidar[len(self.lidar)-i-1]
            self.viewer.draw_polyline( [l.p1, l.p2], color=(1,0,0), linewidth=1 )

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        flagy1 = TERRAIN_HEIGHT
        flagy2 = flagy1 + 50/SCALE
        x = TERRAIN_STEP*3
        self.viewer.draw_polyline( [(x, flagy1), (x, flagy2)], color=(0,0,0), linewidth=2 )
        f = [(x, flagy2), (x, flagy2-10/SCALE), (x+25/SCALE, flagy2-5/SCALE)]
        self.viewer.draw_polygon(f, color=(0.9,0.2,0) )
        self.viewer.draw_polyline(f + [f[0]], color=(0,0,0), linewidth=2 )

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

class BipedalWalkerHardcore(BipedalWalker):
    hardcore = True
