import gym_2048
import gym
import numpy
import time

import sys
sys.path.insert(0, '../..')

import libs_agents
from libs_common.Training import *
from libs_common.Game2048Wrapper import *

import models.dqn_a.src.model            as Model
import models.dqn_a.src.config           as Config

path = "models/dqn_a/" 

env = gym.make("2048-v0")
env = Game2048Wrapper(env, 4)
env.reset()

agent = libs_agents.AgentDQN(env, Model, Config)

max_iterations = 60*(10**6)
#trainig = TrainingIterations(env, agent, max_iterations, path, 10000)
#trainig.run() 


agent.load(path) 
agent.disable_training()
agent.iterations = 0
while True:
    reward, done = agent.main()

    if done:
        print(env.stats)
        print(env.stats_norm)
        print("\n")
