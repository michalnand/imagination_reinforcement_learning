import sys
sys.path.insert(0, '..')

import agents
import time
import gym_2048
import gym
import numpy

from common.Game2048Wrapper import *


path = "models/game2048_dqn_a/"

env = gym.make("2048-v0")
#env = gym.make("Tiny2048-v0")
env = Game2048Wrapper(env, 4)
env.reset()


obs_shape       =   env.observation_space.shape
actions_count   =   env.action_space.n


print("observation shape ", obs_shape)
print("actions count ", actions_count)
print("observation\n", env.observation_space)


 
agent = agents.AgentRandom(env)

while True:
    agent.main()
    env.render()
    print("\n\n")
    time.sleep(0.01)
