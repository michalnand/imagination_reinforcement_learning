import gym
import pybulletgym
import numpy
import time

import sys
sys.path.insert(0, '../..')

import libs_agents
from libs_common.Training import *

import models.a2c.model.src.model    as Model
import models.a2c.model.src.config   as Config

path = "models/a2c/model/"




env = gym.make("AntPyBulletEnv-v0")
#env.render()

agent = libs_agents.AgentA2CContinuous(env, Model, Config)

max_iterations = 4*(10**6)
trainig = TrainingIterations(env, agent, max_iterations, path, 10000)
trainig.run() 

'''
agent.load(path)
agent.disable_training()
while True:
    agent.main()
    env.render()
    time.sleep(0.01)
'''