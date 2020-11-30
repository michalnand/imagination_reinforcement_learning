import gym
from pybullet_envs.bullet import minitaur_gym_env
from pybullet_envs.bullet import minitaur_env_randomizer

import numpy


class Wrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.iterations_max = 1000
        self.iterations     = 0

    def step(self, action):
        action_ = numpy.pi*(0.2*action + 0.5)
        obs, reward, done, info = self.env.step(action_)
        
        self.iterations+= 1
        if self.iterations > self.iterations_max:
            self.iterations = 0
            done = True

        return obs, reward, done, info


def env_create(render_eanbled = False):
    randomizer = (minitaur_env_randomizer.MinitaurEnvRandomizer())
    env = minitaur_gym_env.MinitaurBulletEnv(   render=render_eanbled,
                                                leg_model_enabled=False,
                                                motor_velocity_limit=numpy.inf,
                                                pd_control_enabled=True,
                                                accurate_motor_model_enabled=True,
                                                motor_overheat_protection=True,
                                                env_randomizer=randomizer,
                                                hard_reset=False)

    env = Wrapper(env)
    return env
