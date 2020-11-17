import sys
sys.path.insert(0, '../..')

import libs_agents
import numpy
import time

import gym
from pybullet_envs.bullet import minitaur_gym_env
from pybullet_envs.bullet import minitaur_env_randomizer

randomizer = (minitaur_env_randomizer.MinitaurEnvRandomizer())
env = minitaur_gym_env.MinitaurBulletEnv(   render=True,
                                            leg_model_enabled=False,
                                            motor_velocity_limit=numpy.inf,
                                            pd_control_enabled=True,
                                            accurate_motor_model_enabled=True,
                                            motor_overheat_protection=True,
                                            env_randomizer=randomizer,
                                            hard_reset=False)

class Wrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        action_ = numpy.pi*(0.1*action + 0.5)
        return self.env.step(action_)

env = Wrapper(env)

observation = env.reset()

state_shape    = env.observation_space.shape
actions_count  = env.action_space.shape[0]

print("state_shape      = ", state_shape)
print("actions_count    = ", actions_count)

print("state = \n", observation)



x = 0.0

phase = numpy.zeros(8)

for i in range(8):
    phase[i] = numpy.pi*i/8.0

while True:

    x+= 0.2
    action = numpy.sin(x + phase)

    _, _, done, _ = env.step(action)

    if done:
        env.reset()

    time.sleep(0.01)
